import numpy as np
import scipy.sparse as sp
import skimage.transform as skimt
import os

from pyxu.info import ptype as pxt
import pyxu.operator as pyxop
import pyxu.abc as pxa
from pyxu.abc.operator import LinOp
import pyxu_diffops.operator as px_diffops

from src.tomo_fusion.tools import plotting_fcts as plt_tools


dirname = os.path.dirname(__file__)


class _ExplicitLinOpSparseMatrix(LinOp):
    def __init__(self, dim_shape, mat):
        assert len(mat.shape) == 2, "Matrix `mat` must be a 2-dimensional array"
        super().__init__(dim_shape=dim_shape, codim_shape=mat.shape[0])
        self.mat = mat
        self.num_pixels = np.prod(dim_shape[-2:])

    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        arr = arr.reshape(*arr.shape[:-2], self.num_pixels)
        y = self.mat.dot(arr.T).T
        y = y.reshape(*y.shape[:-1], *self.codim_shape)
        return y

    def adjoint(self, arr: pxt.NDArray) -> pxt.NDArray:
        arr = arr.reshape(*arr.shape[:-1], *self.codim_shape)
        y = self.mat.T.dot(arr.T).T
        y = y.reshape(*y.shape[:-1], *self.dim_shape[1:])
        return y


class _ClipToROI(pxa.ProxFunc):
    r"""
    Clips signal to a region of interest provided at initialization time.
    """
    def __init__(self, dim_shape: pxt.NDArrayShape, clipping_mask: pxt.NDArray):
        super().__init__(dim_shape=dim_shape, codim_shape=1)
        self.clipping_mask = clipping_mask

    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        in_set = (arr*~self.clipping_mask == 0).all()
        out = np.where(in_set, 0, np.inf).astype(arr.dtype)
        return out

    def prox(self, arr: pxt.NDArray, tau: pxt.Real) -> pxt.NDArray:
        out = arr * self.clipping_mask
        return out


class _PositiveClipToROI(pxa.ProxFunc):
    r"""
    Clips signal to a region of interest provided at initialization time adn constrains it to be positive.
    """
    def __init__(self, dim_shape: pxt.NDArrayShape, clipping_mask: pxt.NDArray):
        super().__init__(dim_shape=dim_shape, codim_shape=1)
        self.clipping_mask = clipping_mask

    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        in_set = (arr*~self.clipping_mask == 0).all() and (arr >= 0).all()
        out = np.where(in_set, 0, np.inf).astype(arr.dtype)
        print(out)
        return out

    def prox(self, arr: pxt.NDArray, tau: pxt.Real) -> pxt.NDArray:
        arr0 = arr.clip(0, None)
        out = arr0 * self.clipping_mask
        return out



def _DataFidelityFunctional(dim_shape: pxt.NDArrayShape, noisy_tomo_data: pxt.NDArray,
                           sigma_err: pxt.NDArray, grid: str = "coarse", geometry_matrix: pxt.NDArray = None) -> pxt.OpT:
    if geometry_matrix is None:
        dir_geom_mats = os.path.join(dirname, 'forward_model/geometry_matrices')
        if grid == "coarse":
            geometry_matrix = sp.load_npz(dir_geom_mats+"/sparse_geometry_matrix_sxr.npz")
        elif grid == "fine":
            geometry_matrix = sp.load_npz(dir_geom_mats+"/sparse_geometry_matrix_sxr_fine_grid.npz")
    if isinstance(sigma_err, float) or (isinstance(sigma_err, np.ndarray) and sigma_err.size == 1):
        # define explicit LinOp from geometry matrix
        forward_model_linop = _ExplicitLinOpSparseMatrix(dim_shape=dim_shape, mat=geometry_matrix)
        forward_model_linop.lipschitz = sp.linalg.norm(geometry_matrix, 2)
        op = 1 / (2 * sigma_err ** 2) * pyxop.SquaredL2Norm(dim_shape=(noisy_tomo_data.size,)).argshift(-noisy_tomo_data.flatten()) * forward_model_linop
    elif (isinstance(sigma_err, list) and len(sigma_err) == 2) or (isinstance(sigma_err, np.ndarray) and sigma_err.size == 2):
        sigma_err_vec = np.zeros(noisy_tomo_data.size)
        sigma_err_vec = sigma_err[0] + sigma_err[1] * noisy_tomo_data
        normalized_noisy_tomo_data = noisy_tomo_data.flatten() / sigma_err_vec
        normalized_geometry_matrix = geometry_matrix / np.hstack([normalized_noisy_tomo_data.reshape(-1,1)]*geometry_matrix.shape[1])
        normalized_geometry_matrix = sp.csr_matrix(normalized_geometry_matrix)
        normalized_forward_model_linop = _ExplicitLinOpSparseMatrix(dim_shape=dim_shape, mat=normalized_geometry_matrix)
        normalized_forward_model_linop.lipschitz = sp.linalg.norm(normalized_geometry_matrix, 2)
        op = 1 / 2 * pyxop.SquaredL2Norm(dim_shape=(noisy_tomo_data.size,)).argshift(-normalized_noisy_tomo_data.flatten()) * normalized_forward_model_linop
        forward_model_linop = normalized_forward_model_linop
    # define data-fidelity functional
    #op = 1 / (2 * sigma_err ** 2) * pyxop.SquaredL2Norm(dim_shape=(noisy_tomo_data.size,)).argshift(-noisy_tomo_data.flatten()) * forward_model_linop
    op.sigma_err = sigma_err
    op.noisy_tomo_data = noisy_tomo_data
    op.forward_model_linop = forward_model_linop
    return op


def define_loglikelihood_and_logprior(ground_truth, psi,
                                      reconstruction_shape=(1, 120, 40),
                               sigma_err=1e-2,
                               reg_fct_type="coherence_enhancing",
                               alpha=1e-2, sampling=0.0125,
                               seed=0, plot=False):
    # Define reconstruction grid finesse (coarse grid)
    dim_shape_coarse = (1, 120, 40)
    # Load geometry matrix
    if ground_truth.shape == (120, 40):
        fwd_matrix = sp.load_npz(os.path.join(dirname, "forward_model/geometry_matrices/sparse_geometry_matrix_sxr.npz"))
    elif ground_truth.shape == (240, 80):
        fwd_matrix = sp.load_npz(os.path.join(dirname, "forward_model/geometry_matrices/sparse_geometry_matrix_sxr_fine_grid.npz"))
    else:
        raise ValueError("Ground truth shape must be `(120, 40)` or `(240, 80)`")
    # Reshape magnetic equilibrium if necessary
    if psi.shape != reconstruction_shape[1:]:
        psi = skimt.resize(psi, reconstruction_shape[1:], anti_aliasing=False, mode='edge')

    # Compute noisy data
    np.random.seed(seed)
    tomo_data = fwd_matrix.dot(ground_truth.flatten())
    if isinstance(sigma_err, float) or (isinstance(sigma_err, np.ndarray) and sigma_err.size == 1):
        noisy_tomo_data = tomo_data + sigma_err * np.random.randn(*tomo_data.shape)
    elif (isinstance(sigma_err, list) and len(sigma_err) == 2) or (isinstance(sigma_err, np.ndarray) and sigma_err.size == 2):
        noisy_tomo_data = (tomo_data
                           + sigma_err[0] * np.random.randn(*tomo_data.shape)
                           + sigma_err[1] * tomo_data * np.random.randn(*tomo_data.shape))
    if plot:
        plt_tools.plot_tomo_data(tomo_data, noisy_tomo_data)

    # Define data-fidelity term
    f = _DataFidelityFunctional(dim_shape=reconstruction_shape, noisy_tomo_data=noisy_tomo_data, sigma_err=sigma_err, grid="coarse")
    f.tomo_data = tomo_data

    # Define regularization functional
    if reg_fct_type == "coherence_enhancing":
        g = px_diffops.AnisCoherenceEnhancingDiffusionOp(dim_shape=reconstruction_shape,
                                                        alpha=alpha,
                                                        m=1,
                                                        sigma_gd_st=1*sampling,
                                                        smooth_sigma_st=2*sampling,
                                                        freezing_arr=psi,
                                                        sampling=sampling,
                                                        matrix_based_impl=True)
    elif reg_fct_type == "anisotropic":
        g = px_diffops.AnisDiffusionOp(dim_shape=reconstruction_shape,
                                       alpha=alpha,
                                       diff_method_struct_tens="fd",
                                       freezing_arr=psi,
                                       sampling=sampling,
                                       matrix_based_impl=True)
    elif reg_fct_type == "MFI":
        raise ValueError("reg_fct_type `MFI` not available yet")
    elif reg_fct_type == "anisMFI":
        raise ValueError("reg_fct_type `AnisMFI` not available yet")

    return f, g


def define_loglikelihood_cv(f, cv_type="CV_single", cv_strategy="random", seed=0):
    if cv_type == "CV_single":
        # select randomly 80 LoS. Remaining 20 will be used for tuning
        np.random.seed(seed)
        if cv_strategy == "random":
            cv_idx = np.sort(np.random.choice(np.arange(0, f.forward_model_linop.codim_size),
                                          int(0.8*f.forward_model_linop.codim_size), False))
        elif cv_strategy == "by_camera":
            cv_idx = np.delete(np.arange(0, 100),
                               (20 * np.random.randint(0,5)) + np.arange(0, 20)
                               )
        else:
            raise ValueError("cv_strategy {} not available".format(cv_strategy))
        # create cv data fidelity functional
        f_cv = _DataFidelityFunctional(dim_shape=f.forward_model_linop.dim_shape,
                                              noisy_tomo_data=f.noisy_tomo_data[cv_idx],
                                              sigma_err=f.sigma_err,
                                              geometry_matrix=f.forward_model_linop.mat[cv_idx, :])
        f_cv.tomo_data = f.tomo_data[cv_idx]
        f_cv.cv_idx = cv_idx
        f_cv.cv_test_idx = np.delete(np.arange(0, 100), cv_idx)

    elif cv_type == "CV_full":
        # initialize list of data fidelity functionals
        f_cv = []
        idxs = np.arange(0, 100)
        np.random.seed(seed)
        if cv_strategy == "random":
            np.random.shuffle(idxs)
        # for each cross-validation fold, assemblate data fidelity functional
        for i in range(5):
            cv_idx = np.delete(idxs, idxs[i * 20: (i + 1) * 20])
            # create cv data fidelity functional
            f_cv_ = _DataFidelityFunctional(dim_shape=f.forward_model_linop.dim_shape,
                                                  noisy_tomo_data=f.noisy_tomo_data[cv_idx],
                                                  sigma_err=f.sigma_err,
                                                  geometry_matrix=f.forward_model_linop.mat[cv_idx, :])
            f_cv_.tomo_data = f.tomo_data[cv_idx]
            f_cv_.cv_idx = cv_idx
            f_cv_.cv_test_idx = idxs[i * 20: (i + 1) * 20]
            # append i-th functional to the list
            f_cv.append(f_cv_)

    else:
        raise ValueError("cv_type must be `CV_single` or `CV_full`")

    return f_cv
