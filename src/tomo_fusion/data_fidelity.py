import numpy as np
from pyxu.info import ptype as pxt
import scipy.sparse as sp
import pyxu.operator as pyxop
from pyxu.abc.operator import LinOp
import os


dirname = os.path.dirname(__file__)


class ExplicitLinOpSparseMatrix(LinOp):
    def __init__(self, dim_shape, mat):
        assert len(mat.shape) == 2, "Matrix `mat` must be a 2-dimensional array"
        super().__init__(dim_shape=dim_shape, codim_shape=mat.shape[0])
        self.mat = mat
        self.num_pixels = np.prod(dim_shape[-2:])

    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        arr = arr.reshape(*arr.shape[:-2], self.num_pixels)
        y = self.mat.dot(arr.T).T
        y = y.reshape(*y.shape[:-1], self.codim_shape)
        return y

    def adjoint(self, arr: pxt.NDArray) -> pxt.NDArray:
        arr = arr.reshape(*arr.shape[:-1], self.codim_shape)
        y = self.mat.T.dot(arr.T).T
        y = y.reshape(*y.shape[:-1], *self.dim_shape[1:])
        return y


def DataFidelityFunctional(dim_shape: pxt.NDArrayShape, tomo_data: pxt.NDArray, sigma_err: pxt.NDArray, grid: str = "coarse") -> pxt.OpT:
    if grid == "coarse":
        geometry_matrix = sp.load_npz("../tomo_fusion/forward_model/matrices/sparse_geometry_matrix_sxr.npz")
    elif grid == "fine":
        geometry_matrix = sp.load_npz("../tomo_fusion/forward_model/matrices/sparse_geometry_matrix_sxr_fine_grid.npz")
    # define explicit LinOp from geometry matrix
    forward_model_linop = ExplicitLinOpSparseMatrix(dim_shape=dim_shape, mat=geometry_matrix)
    forward_model_linop.lipschitz = np.linalg.norm(geometry_matrix.toarray(), 2)
    # define data-fidelity functional
    op = 1 / (2 * sigma_err ** 2 ) * pyxop.SquaredL2Norm(dim_shape=(tomo_data.size,)).argshift(-tomo_data.ravel()) * forward_model_linop
    return op
