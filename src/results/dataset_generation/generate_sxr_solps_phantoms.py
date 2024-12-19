import numpy as np
import skimage.transform as skimt
from pyxu_diffops.operator import AnisCoherenceEnhancingDiffusionOp, AnisDiffusionOp
import argparse

from src.tomo_fusion.tools.helpers import define_core_mask


def add_gaussian_background(phantom, psi, mask, peak_value_gaussian=0., sigma_gaussian_normalized_radius=0.5):
    gaussian_profile = peak_value_gaussian * np.exp(-((psi-np.min(psi))/(np.min(psi)))**2 / (2*sigma_gaussian_normalized_radius**2))
    gaussian_profile *= mask
    phantom_new = phantom + gaussian_profile
    phantom_new /= np.max(phantom_new)
    return phantom_new


def generate_sxr_samples(psi_fct, dim_shape=(1, 120, 40), sampling=0.0125,
                                                  reg_fct_type="coherence_enhancing",
                                                  sigma_gd_st_factor=1, smooth_sigma_st_factor=2,
                                                  steps_nb_factor=1, nsamples=1e3, seed=0, save=False, save_dir='sxr_samples',
                                                  clip_each_iter=False, clipping_outside_core=False,
                                                  psi_upperbound_source_rel=0.4,
                                                  psi_lowerbound_source_rel=1, xpoint_idx_base_psi=90,
                                                  diff_method_struct_tens="gd", nsources=1, gpu=False,
                                                  peak_values_gaussian_background=None):
    if gpu:
        import cupy as cp
    if peak_values_gaussian_background is None:
        # if no peak values for gaussian background are provided, initialize to zero (no background)
        peak_values_gaussian_background = np.zeros(nsamples)
    # define bounds for psi defining allowed source locations
    psi_upperbound_source = psi_upperbound_source_rel * np.min(psi_fct)
    psi_lowerbound_source = psi_lowerbound_source_rel * np.min(psi_fct)
    # instantiate lists for generated samples
    sxr_samples = []
    sxr_samples_with_background = []
    psis = []

    # fix seed and sample/initialize random quantities to generate samples
    np.random.seed(seed)
    alpha_random_values = np.zeros(nsamples)
    # trimming limits
    upper_t = 37
    lower_t = 27
    hfs_t = 5  # 4
    lfs_t = 3  # 2
    trimming_values = np.zeros((nsamples, 4))
    source_loc = np.zeros((nsamples, 2), dtype=int)
    psis_at_source = np.zeros((nsamples))
    diffusive_steps = np.zeros(nsamples, dtype=int)
    # generate samples
    for i in range(nsamples):
        # generate equilibrium, based on a stretched version of base version
        upper_trim = int(upper_t * np.random.rand())
        lower_trim = int(120 - lower_t * np.random.rand())
        hfs_trim = int(hfs_t * np.random.rand())
        lfs_trim = int(40 - lfs_t * np.random.rand())
        trimming_values[i, :] = np.array([upper_trim, lower_trim, hfs_trim, lfs_trim])

        # resize psi function to match prescribed discretization
        psi = skimt.resize(psi_fct[upper_trim:lower_trim, hfs_trim:lfs_trim], dim_shape[1:], anti_aliasing=True,
                           mode='edge')

        # sample random location for the source
        allowed_source_points = np.zeros(dim_shape[1:])
        mask_locs = np.where(
            np.logical_and(psi <= psi_upperbound_source,
                           psi >= psi_lowerbound_source))
        # remove points falling below the x-point
        xpoint_loc = int(dim_shape[1] * (xpoint_idx_base_psi - trimming_values[i, 0]) / (
                    trimming_values[i, 1] - trimming_values[i, 0]))
        mask_locs = (mask_locs[0][mask_locs[0] < xpoint_loc],
                     mask_locs[1][mask_locs[0] < xpoint_loc])
        allowed_source_points[mask_locs] = 1
        # size of matrix is number of pixels belonging to mask
        nonzero_elems = int(np.sum(allowed_source_points))
        nsources = nsources
        sources_idx = np.random.randint(0, nonzero_elems - 1, size=(nsources,))

        # create source image
        x0 = np.zeros((dim_shape[1], dim_shape[2]), dtype=psi_fct.dtype)
        for j in range(nsources):
            x0[mask_locs[0][sources_idx[j]]:mask_locs[0][sources_idx[j]] + 2,
            mask_locs[1][sources_idx[j]]:mask_locs[1][sources_idx[j]] + 2] = 1
        source_loc[i, :] = np.array(mask_locs[0][sources_idx[0]], mask_locs[1][sources_idx[0]])

        # Assume one source, assign different alpha values depending on source's position
        psi_at_source = psi[mask_locs[0][sources_idx[0]], mask_locs[1][sources_idx[0]]]
        psis_at_source[i] = psi_at_source

        source_loc_bins = np.linspace(psi_lowerbound_source, psi_upperbound_source, 6, endpoint=True)
        alpha_bounds = np.array([[np.log(1e-2), np.log(5e-1)],
                                 [np.log(5e-3), np.log(5e-2)],
                                 [np.log(5e-3), np.log(5e-2)],  # 1e-2
                                 [np.log(1e-3), np.log(5e-2)],  # 5e-3
                                 [np.log(5e-4), np.log(5e-2)]])  # 5e-3
        diffusive_steps_bounds = steps_nb_factor * np.array([[int(5e3), int(1e4)],
                                           [int(1e3), int(1e4)],
                                           [int(5e2), int(5e3)],
                                           [int(5e2), int(5e3)],
                                           [int(5e2), int(5e3)]])
        for bin in range(source_loc_bins.size - 1):
            if psi_at_source >= source_loc_bins[bin] and psi_at_source < source_loc_bins[bin + 1]:
                alpha_random_values[i] = np.exp(np.random.uniform(alpha_bounds[bin, 0],
                                                                  alpha_bounds[bin, 1]))
                if alpha_random_values[i] < 5e-3:
                    lower_bound_diffusive_steps = steps_nb_factor * int(3e3)
                else:
                    lower_bound_diffusive_steps = diffusive_steps_bounds[bin, 0]
                diffusive_steps[i] = np.random.randint(lower_bound_diffusive_steps,
                                                       diffusive_steps_bounds[bin, 1])

                # if we need to clip outside core, define core mask_core
        if clipping_outside_core:
            mask_core = define_core_mask(psi, dim_shape[1:],
                                             trim_values_x=[trimming_values[i, 0], trimming_values[i, 1]])
            mask_core = np.expand_dims(mask_core, 0)
            # plt.imshow(mask_core.squeeze())
            # plt.colorbar()
            # plt.contour(psi, origin="lower", levels=15, antialiased=True, colors="r", linewidths=0.2)
            # plt.show()

        if gpu:
            psi = cp.array(psi)

        #breakpoint()
        #st = time.time()
        if reg_fct_type == "coherence_enhancing":
            reg_fct = AnisCoherenceEnhancingDiffusionOp(dim_shape=dim_shape, alpha=alpha_random_values[i], m=1,
                                                        sigma_gd_st=sigma_gd_st_factor * sampling, smooth_sigma_st=smooth_sigma_st_factor * sampling,
                                                        sampling=sampling, diff_method_struct_tens=diff_method_struct_tens,
                                                        freezing_arr=psi, matrix_based_impl=True, gpu=gpu, dtype=psi_fct.dtype)
        elif reg_fct_type == "anisotropic":
            reg_fct = AnisDiffusionOp(dim_shape=dim_shape,
                                           alpha=alpha_random_values[i],
                                           diff_method_struct_tens=diff_method_struct_tens,
                                           freezing_arr=psi,
                                           sampling=sampling,
                                           matrix_based_impl=True,
                                           gpu=gpu, dtype=psi_fct.dtype)

        #print(reg_fct._grad_matrix_based.mat.dtype, type(reg_fct._grad_matrix_based.mat))
        #reg_fct._grad_matrix_based.mat = reg_fct._grad_matrix_based.mat.astype(Bfield.dtype)
        #reg_fct._grad_matrix_based.mat = sp.csr_matrix(reg_fct._grad_matrix_based.mat).

        #print("instantiation-time", time.time()-st)
        #st=time.time()

        # print information
        print("Phantom {}: ".format(i))
        print(psi_at_source, alpha_random_values[i], nsources, diffusive_steps[i])
        print("\n")

        # generate sample, adapting number of diffusive steps for "extreme" initialization locations
        #x0 = np.expand_dims(x0, 0)
        x0 = x0.flatten()
        if clipping_outside_core:
            mask_core = mask_core.flatten()
        if gpu:
            x0 = cp.array(x0)
            if clipping_outside_core:
                mask_core = cp.array(mask_core)
        #print("loading-time", time.time()-st)
        for s in range(diffusive_steps[i]):
            # 2/L with L=8 seems enough, L=8 good estimate of diff_lipschitz constant of reg_fct in principle..
            #x0 -= 0.95 * (0.25 * (sampling ** 2)) * reg_fct.grad(x0)
            x0 -= 0.95 * (0.25 * (sampling ** 2)) * reg_fct._grad_matrix_based.mat.dot(x0)
            #x0 -= 0.95 * (0.25 * (sampling ** 2)) * reg_fct._grad_matrix_based.mat@x0
            if clip_each_iter:
                x0[x0 < 0] = 0
            if clipping_outside_core:
                x0 *= mask_core
            # if s % 1000 == 0:
            #     plt.imshow(x0.squeeze())
            #     plt.colorbar()
            #     plt.contour(psi, origin="lower", levels=15, antialiased=True, colors="r", linewidths=0.2)
            #     plt.title("it {}".format(s))
            #     plt.show()
        x0[x0 < 0] = 0
        if clipping_outside_core:
            x0 *= mask_core

        #print("loop-time", time.time() - st)
        # normalize sample to 1
        x0 /= x0.max()
        # append sample and magnetic equilibrium
        if gpu:
            x0 = x0.get()
            psi = psi.get()

        sxr_samples.append(x0.reshape(dim_shape).squeeze())
        # add gaussian background
        x0_plus_background = add_gaussian_background(x0.reshape(dim_shape).squeeze(), psi,
                                                     mask_core.reshape(dim_shape[1], dim_shape[2]), peak_values_gaussian_background[i])
        sxr_samples_with_background.append(x0_plus_background)
        psis.append(psi)
        #print("loop-get-time", time.time()-st)

        # # save information
        if save:
            import os
            save_dir = save_dir
            if not os.path.isdir(save_dir):
                os.mkdir(save_dir)
            np.save(save_dir + '/sxr_samples.npy', sxr_samples)
            np.save(save_dir + '/sxr_samples_with_background.npy', sxr_samples_with_background)
            np.save(save_dir + '/psis.npy', psis)
            np.save(save_dir + '/diffusive_steps.npy', diffusive_steps)
            np.save(save_dir + '/alpha_random_values.npy', alpha_random_values)
            np.save(save_dir + '/psis_at_source.npy', psis_at_source)
            np.save(save_dir + '/trimming_values.npy', trimming_values)
            np.save(save_dir + '/source_loc.npy', source_loc)
            np.save(save_dir + '/peak_values_gaussian_background.npy', peak_values_gaussian_background)
        #print("loop-get-save-time", time.time()-st)
        #breakpoint()
    return sxr_samples, psis


if __name__ == '__main__':
    # parse -gpu argument
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', action='store_true', default=False, help='Use GPU')
    args = parser.parse_args()
    print(args.gpu)

    psi_eq = np.load('magnetic_equilibrium.npy')

    nsamples = int(1e3)

    # define random peak values of gaussian background to be added to perturbations
    np.random.seed(0)
    min_peak_value = 0.2
    peak_vals = min_peak_value + (1-min_peak_value) * np.random.rand(nsamples)

    _, _ = generate_sxr_samples(psi_eq, dim_shape=(1, 240, 80),
                                                         reg_fct_type="coherence_enhancing",
                                                         sigma_gd_st_factor=0, smooth_sigma_st_factor=2,
                                                         sampling=0.00625, steps_nb_factor=2,
                                                    nsamples=nsamples, seed=0, save=True, save_dir="sxr_samples_fine_discretization_new",
                                                    clip_each_iter=True, clipping_outside_core=True,
                                                    diff_method_struct_tens="fd", gpu=args.gpu,
                                                    peak_values_gaussian_background=peak_vals)

    _, _ = generate_sxr_samples(psi_eq, dim_shape=(1, 120, 40),
                                                         reg_fct_type="coherence_enhancing",
                                                         sigma_gd_st_factor=0, smooth_sigma_st_factor=1,
                                                         sampling=0.0125,
                                                    nsamples=nsamples, seed=0, save=True, save_dir="sxr_samples_new",
                                                    clip_each_iter=True, clipping_outside_core=True,
                                                    diff_method_struct_tens="fd", gpu=args.gpu,
                                                    peak_values_gaussian_background=peak_vals)

    _, _ = generate_sxr_samples(psi_eq, dim_shape=(1, 240, 80),
                                                         reg_fct_type="anisotropic",
                                                         sampling=0.00625, steps_nb_factor=2,
                                                    nsamples=nsamples, seed=0, save=True, save_dir="sxr_samples_fine_discretization_anisotropic_new",
                                                    clip_each_iter=True, clipping_outside_core=True,
                                                    diff_method_struct_tens="fd", gpu=args.gpu,
                                                    peak_values_gaussian_background=peak_vals)

    _, _ = generate_sxr_samples(psi_eq, dim_shape=(1, 120, 40),
                                                         reg_fct_type="anisotropic",
                                                         sampling=0.0125,
                                                    nsamples=nsamples, seed=0, save=True, save_dir="sxr_samples_anisotropic_new",
                                                    clip_each_iter=True, clipping_outside_core=True,
                                                    diff_method_struct_tens="fd", gpu=args.gpu,
                                                    peak_values_gaussian_background=peak_vals)




# fig,ax=plt.subplots(4,5, figsize=(6,16))
# for i in range(4):
#     for j in range(5):
#         im=ax[i,j].imshow(samples_solps[i*5+j])
#         ax[i, j].contour(psis_solps[i*5+j], origin="lower", levels=15, antialiased=True, colors="r", linewidths=0.2)
#         #plt.colorbar(im, ax=ax[i,j])
#         ax[i,j].set_title("{}".format(i*5+j))
#         ax[i,j].axis('off')
# plt.show()