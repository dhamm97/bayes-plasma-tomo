import matplotlib.pyplot as plt
import numpy as np
import src.tomo_fusion.helpers as tools
import skimage.transform as skimt
import diffusion_ops.operator.diffusion_op._diffusion_op as diffop_private
import diffusion_ops.operator.diffusion_op.diffusion_op as diffop
import pyxu.abc as pxa
import scipy.sparse as sp

arg_shape = (120,40)
N = int(np.prod(arg_shape))


def define_clipping_mask(psi, xpoint_idx_basi_psi=90, trim_values_x=[0, 120], psi_threshold=0):
    mask = np.where(psi<psi_threshold)
    xpoint_loc = int(120 * (xpoint_idx_basi_psi - trim_values_x[0]) / (trim_values_x[1] - trim_values_x[0]) )
    mask_idx_above_xpoint_x = mask[0][mask[0] < xpoint_loc]
    mask_idx_above_xpoint_y = mask[1][mask[0] < xpoint_loc]
    # mask_core = (mask_idx_above_xpoint_x, mask_idx_above_xpoint_y)
    # mask_outside_core = np.ones(arg_shape, dtype=bool)
    # mask_outside_core[mask_core] = False
    # return mask_outside_core
    mask_core = np.zeros(arg_shape, dtype=bool)
    mask_core[mask_idx_above_xpoint_x, mask_idx_above_xpoint_y] = True
    return mask_core


def generate_sxr_samples(Bfield, arg_shape=(120, 40), nsamples=1e3, seed=0, save=False, clip_each_iter=False, clipping_outside_core=False, psi_upperbound_source_rel=0.4,
  psi_lowerbound_source_rel=1,
  psi_threshold_center_rel=1, xpoint_idx_base_psi=90,
  diff_method_struct_tens="gd"):
    # define bounds for psi defining allowed source locations
    psi_upperbound_source = psi_upperbound_source_rel * np.min(Bfield)
    psi_lowerbound_source = psi_lowerbound_source_rel * np.min(Bfield)
    psi_threshold_center = psi_threshold_center_rel * np.min(Bfield)
    # instantiate lists for generated samples
    sxr_samples = []
    psis = []
    # instantiate coherence enhancing diffusion operator, used to generate samples
    # base version, to instantiate gradietn and diffusion coefficient
    sampling = 0.0125
    reg_fct_matrixfree = diffop.AnisCoherenceEnhancingDiffusionOp(arg_shape=arg_shape,
                                                                  alpha=1e-2,
                                                                  m=1,
                                                                  sigma_gd_st=1*sampling,
                                                                  smooth_sigma_st=2*sampling,
                                                                  sampling=sampling,
                                                                  diff_method_struct_tens=diff_method_struct_tens)
    reg_fct_matrixfree *= (sampling ** 2)
    Dx = - np.diag(np.ones(arg_shape[0])) + np.diag(np.ones(arg_shape[0] - 1), 1)
    Dx[-1, -1] = 0  # symmetric boundary conditions, no flux
    Dy = - np.diag(np.ones(arg_shape[1])) + np.diag(np.ones(arg_shape[1] - 1), 1)
    Dy[-1, -1] = 0  # symmetric boundary conditions, no flux
    # define gradient matrix
    D = np.vstack((np.kron(Dx, np.eye(arg_shape[1])), np.kron(np.eye(arg_shape[0]), Dy)))

    # fix seed and sample/initialize random quantities to generate samples
    np.random.seed(seed)
    alpha_random_values = np.zeros(nsamples)
    # trimming limits
    upper_t=37
    lower_t=27
    hfs_t=5#4
    lfs_t=3#2
    trimming_values = np.zeros((nsamples, 4))
    source_loc = np.zeros((nsamples, 2), dtype=int)
    psis_at_source = np.zeros((nsamples))
    diffusive_steps = np.zeros(nsamples, dtype=int)
    # generate samples
    for i in range(nsamples):
        # generate equilibrium, based on a stretched version of base version
        upper_trim = int(upper_t*np.random.rand())
        lower_trim = int(120-lower_t*np.random.rand())
        hfs_trim = int(hfs_t*np.random.rand())
        lfs_trim = int(40-lfs_t*np.random.rand())
        trimming_values[i, :] = np.array([upper_trim,lower_trim,hfs_trim,lfs_trim])
        psi = skimt.resize(Bfield[upper_trim:lower_trim,hfs_trim:lfs_trim], (120,40), anti_aliasing=True)
        # if we need to clip outside core, define core mask_core
        if clipping_outside_core:
            mask_core = define_clipping_mask(psi, trim_values_x=[trimming_values[i, 0], trimming_values[i, 1]],
                                             psi_threshold=0).flatten()

        # sample random location for the source
        allowed_source_points = np.zeros((120,40))
        mask_locs = np.where(np.logical_or(
            np.logical_and(psi <= psi_upperbound_source,
                           psi >= psi_lowerbound_source),
            psi <= psi_threshold_center))
        # remove points falling below the x-point
        xpoint_loc = int(120 * (xpoint_idx_base_psi - trimming_values[i, 0]) / (trimming_values[i, 1] - trimming_values[i, 0]) )
        mask_locs = (mask_locs[0][mask_locs[0] < xpoint_loc],
                     mask_locs[1][mask_locs[0] < xpoint_loc])
        allowed_source_points[mask_locs] = 1
        # size of matrix is number of pixels belonging to mask
        nonzero_elems = int(np.sum(allowed_source_points))
        nsources = 1
        sources_idx = np.random.randint(0, nonzero_elems-1, size=(nsources, ))

        # create source image
        x0 = np.zeros((120, 40))
        for j in range(nsources):
            x0[mask_locs[0][sources_idx[j]], mask_locs[1][sources_idx[j]]] = 1
        source_loc[i, :] = np.array(mask_locs[0][sources_idx[0]], mask_locs[1][sources_idx[0]])

        # Assume one source, assign different alpha values depending on source's position
        psi_at_source = psi[mask_locs[0][sources_idx[0]], mask_locs[1][sources_idx[0]]]
        psis_at_source[i] = psi_at_source

        #if psi_at_source < psi_threshold_center:
        #    alpha_random_values[i] = np.exp(np.random.uniform(np.log(5e-2),np.log(5e-1)))
        #    diffusive_steps[i] = np.random.randint(int(5e3), int(1e4))

        source_loc_bins = np.linspace(psi_lowerbound_source,psi_upperbound_source, 6, endpoint=True)
        alpha_bounds = np.array([[np.log(1e-2), np.log(5e-1)],
                                 [np.log(5e-3), np.log(5e-2)],
                                 [np.log(5e-3), np.log(5e-2)],#1e-2
                                 [np.log(1e-3), np.log(5e-2)],#5e-3
                                 [np.log(5e-4), np.log(5e-2)]])#5e-3
        diffusive_steps_bounds = np.array([[int(5e3), int(1e4)],
                                           [int(1e3), int(1e4)],
                                           [int(5e2), int(5e3)],
                                           [int(5e2), int(5e3)],
                                           [int(5e2), int(5e3)]])
        for bin in range(source_loc_bins.size-1):
            if psi_at_source>=source_loc_bins[bin] and psi_at_source<source_loc_bins[bin+1]:
                alpha_random_values[i] = np.exp(np.random.uniform(alpha_bounds[bin, 0],
                                                                  alpha_bounds[bin, 1]))
                if alpha_random_values[i] < 5e-3:
                    lower_bound_diffusive_steps = int(3e3)
                else:
                    lower_bound_diffusive_steps = diffusive_steps_bounds[bin, 0]
                #diffusive_steps[i] = lower_bound_diffusive_steps+np.abs(ss.laplace.rvs(0, scale=50000, size=1))
                diffusive_steps[i] = np.random.randint(lower_bound_diffusive_steps,
                                                       diffusive_steps_bounds[bin, 1])

        reg_fct_matrixfree._op.diffusion_coefficient.alpha = alpha_random_values[i]
        u, e = reg_fct_matrixfree._op.diffusion_coefficient._eigendecompose_struct_tensor(psi.reshape(1, -1))
        lambdas = reg_fct_matrixfree._op.diffusion_coefficient._compute_intensities(e)
        tensors = reg_fct_matrixfree._op.diffusion_coefficient._assemble_tensors(u, lambdas)
        W = np.diag(np.hstack((tensors[:, 0, 0], tensors[:, 1, 1]))) + \
            np.diag(tensors[:, 0, 1], N) + \
            np.diag(tensors[:, 1, 0], -N)
        L = D.T @ W @ D
        # use it to define a QuadraticFunc for PGD
        Q_linop_sparse = pxa.LinOp.from_array(A=sp.csr_matrix(L))
        reg_fct = pxa.operator.QuadraticFunc(shape=(1, N), Q=Q_linop_sparse)

        # print information
        print("Phantom {}: ".format(i))
        print(psi_at_source, alpha_random_values[i], nsources, diffusive_steps[i])
        print("\n")

        # generate sample, adapting number of diffusive steps for "extreme" initialization locations
        x0 = x0.reshape(1, -1)
        for s in range(diffusive_steps[i]):
            # 2/L with L=8 seems enough, L=8 good estimate of diff_lipschitz constant of reg_fct
            x0-=0.95*0.25*reg_fct.grad(x0)
            if clip_each_iter:
                x0[x0<0] = 0
            if clipping_outside_core:
                x0 *= mask_core
        x0[x0<0] = 0
        if clipping_outside_core:
            x0 *= mask_core
        # normalize sample to 1
        x0 /= np.max(x0)
        # append sample and magnetic equilibrium
        sxr_samples.append(x0.reshape(arg_shape))
        psis.append(psi)

        # # save information
        if save:
            import os
            save_dir = 'pnp_samples_solps_new'
            if not os.path.isdir(save_dir):
                os.mkdir(save_dir)
            np.save(save_dir+'/sxr_samples.npy', sxr_samples)
            np.save(save_dir+'/psis.npy', psis)
            np.save(save_dir+'/diffusive_steps.npy', diffusive_steps)
            np.save(save_dir+'/alpha_random_values.npy', alpha_random_values)
            np.save(save_dir+'/psis_at_source.npy', psis_at_source)
            #np.save(save_dir+'/source_intensities.npy')
            np.save(save_dir+'/trimming_values.npy', trimming_values)
            np.save(save_dir+'/source_loc.npy', source_loc)

    return sxr_samples, psis


B_solps = np.load('../../tomo_fusion/phantoms/Bfield_saved/psi_eq_SOLPS.npy')

samples_solps, psis_solps = generate_sxr_samples(100*B_solps, arg_shape=(120, 40),
                                                nsamples=int(2e3), seed=0, save=True,
                                                clip_each_iter=True, clipping_outside_core=True,
                                                diff_method_struct_tens="gd")

import matplotlib
matplotlib.use('TkAgg')

# fig,ax=plt.subplots(4,5, figsize=(6,16))
# for i in range(4):
#     for j in range(5):
#         im=ax[i,j].imshow(samples_solps[i*5+j])
#         ax[i, j].contour(psis_solps[i*5+j], origin="lower", levels=15, antialiased=True, colors="r", linewidths=0.2)
#         #plt.colorbar(im, ax=ax[i,j])
#         ax[i,j].set_title("{}".format(i*5+j))
#         ax[i,j].axis('off')
# plt.show()