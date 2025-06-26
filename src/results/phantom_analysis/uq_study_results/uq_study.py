import scipy.sparse as sp
import numpy as np
import copy
import matplotlib.pyplot as plt
import os
import time
import skimage.transform as skimt
import sys

import src.tomo_fusion.tools.helpers as tomo_helps
import src.tomo_fusion.functionals_definition as fct_def
import src.tomo_fusion.hyperparameter_tuning as hyper_tune
import src.tomo_fusion.bayesian_computations as bcomp


def run_uq_study(sigma_err, reg_param, phantom_indices, phantom_dir, saving_dir):
    # Load phantom data
    psis = np.load(phantom_dir+'/psis.npy')
    sxr_samples = np.load(phantom_dir+'/sxr_samples_with_background.npy')
    trim_val = np.load(phantom_dir+'/trimming_values.npy')

    reconstruction_shape = (120, 40)

    for i, idx in enumerate(phantom_indices):
        print("phantom ", idx)
        ground_truth = copy.deepcopy(sxr_samples[idx, :, :].squeeze())
        psi = psis[idx, :, :]
        trim_val_ = trim_val[idx, :]
        mask_core = tomo_helps.define_core_mask(psi=psi, dim_shape=reconstruction_shape, trim_values_x=trim_val_)

        # Define functionals
        f, g = fct_def.define_loglikelihood_and_logprior(ground_truth=ground_truth, psi=psi,
                                                         sigma_err=sigma_err, reg_fct_type="anisotropic",
                                                         alpha=1e-2, plot=False,
                                                         seed=idx)
        start_time_anis_param_tuning = time.time()
        # tune anisotropic parameter
        anis_param_data = hyper_tune.anis_param_tuning(f, g, reg_param,
                                                       tuning_techniques=["CV_full"],
                                                       ground_truth=ground_truth,
                                                       with_pos_constraint=True, clipping_mask=mask_core,
                                                       cv_strategy=["random"],
                                                       anis_params=np.logspace(-4, 0, 13), plot=False)
        time_anis_param_tuning = time.time() - start_time_anis_param_tuning
        anis_param_data["time"] = time_anis_param_tuning

        g = hyper_tune._redefine_anis_param_logprior(g, anis_param_data["CV_full_random"][
            0, np.argmin(anis_param_data["CV_full_random"][1, :])])

        start_time_uq_study = time.time()
        uq_data = bcomp.run_ula(f, g, reg_param, psi, trim_val_,
                                 with_pos_constraint=True,
                                 clip_iterations="core",
                                 compute_stats_wrt_MAP=True,
                                 estimate_quantiles=True,
                                quantile_marks=[0.005, 0.025, 0.05, 0.16, 0.25, 0.5, 0.75, 0.84, 0.95, 0.975, 0.995],
                                samples=int(1e5))
        time_uq_study = time.time() - start_time_uq_study
        uq_data["time"] = time_uq_study

        np.save(saving_dir+'anis_data_'+ str(idx)+'.npy', np.array(anis_param_data))
        np.save(saving_dir + 'uq_data_' + str(idx)+'.npy', np.array(uq_data))

    return


def postprocess_uq_data(uq_data_dir):
    # Load phantom data
    psis = np.load('../dataset_generation/sxr_samples/psis.npy')
    sxr_samples = np.load('../dataset_generation/sxr_samples/sxr_samples_with_background.npy')
    trim_val = np.load('../dataset_generation/sxr_samples/trimming_values.npy')

    # pixel value stats
    pixels_within_quantiles = np.zeros((900, 5))
    pixels_within_n_stds = np.zeros((900, 3))
    mse_map, mse_mean = np.zeros(900), np.zeros(900)
    mape_map, mape_mean = np.zeros(900), np.zeros(900)
    # peak location stats
    true_peak_loc = np.zeros((900, 2))
    peak_within_n_stds = np.zeros((900, 3))
    peak_distance_from_mean = np.zeros((900, 2))
    # prad stats
    true_prad_core = np.zeros(900)
    prad_within_n_stds = np.zeros((900, 3))
    prad_std_values = np.zeros(900)
    prad_rel_error_map, prad_rel_error_mean = np.zeros(900), np.zeros(900)
    prad_within_quantiles = np.zeros((900, 5))

    for idx in range(900):
        recon_shape = (120, 40)

        uq_data_idx = np.load(uq_data_dir + '/uq_data_' + str(idx) + '.npy', allow_pickle=True).item()
        quantile_marks = uq_data_idx['quantile_marks']

        ground_truth = sxr_samples[idx, :, :].squeeze()
        psi = psis[idx, :, :]

        mask_core = tomo_helps.define_core_mask(psi=psi, dim_shape=recon_shape, trim_values_x=trim_val[idx, :])

        # compute pixels withing quantile bounds stats
        ground_truth_downsampled = skimt.resize(ground_truth, uq_data_idx["im_MAP"].shape, anti_aliasing=False,
                                                mode='edge')  # * mask_core
        core_pixels_number = ground_truth_downsampled[ground_truth_downsampled > 1e-3].size
        for j in range(5):
            # qunatile-related
            idxs = np.where((ground_truth_downsampled >= uq_data_idx["empirical_quantiles"][j, :, :]) & (
                        ground_truth_downsampled <= uq_data_idx["empirical_quantiles"][-(j + 1), :, :]))
            frac_pixels_within_quantiles_core = (idxs[0].size - (4800 - np.sum(mask_core))) / np.sum(mask_core)
            pixels_within_quantiles[idx, j] = frac_pixels_within_quantiles_core
        for j, nb_stds in enumerate(np.arange(1, 4)):
            # standard deviation-related
            pixels_within_n_stds_idxs = np.where((ground_truth_downsampled >= (uq_data_idx["mean"] - nb_stds * np.sqrt(uq_data_idx["var"]))) &
                                    (ground_truth_downsampled <= (uq_data_idx["mean"] + nb_stds * np.sqrt(uq_data_idx["var"]))))
            frac_pixels_within_n_stds_core = (pixels_within_n_stds_idxs[0].size - (4800 - core_pixels_number)) / core_pixels_number
            pixels_within_n_stds[idx, j] = frac_pixels_within_n_stds_core
        mse_map[idx] = np.mean((ground_truth_downsampled - uq_data_idx["im_MAP"]) ** 2)
        mse_mean[idx] = np.mean((ground_truth_downsampled - uq_data_idx["mean"]) ** 2)
        mape_map[idx] = np.mean(np.abs((ground_truth_downsampled[mask_core] - uq_data_idx["im_MAP"][mask_core]) / ground_truth_downsampled[mask_core]))
        mape_mean[idx] = np.mean(np.abs((ground_truth_downsampled[mask_core] - uq_data_idx["mean"][mask_core])/ground_truth_downsampled[mask_core]))

        # compute peak location stats
        true_peak_loc[idx, :] = np.array(np.where(ground_truth == ground_truth.max())).reshape(2) / 2
        peak_distance_from_mean[idx, :] = (uq_data_idx["mean_peak_loc"] - true_peak_loc[idx, :]) * uq_data_idx["sampling"][0]
        for j, nb_stds in enumerate(np.arange(1, 4)):
            peak_within_n_stds[idx, j] = (
                    (np.abs(uq_data_idx["mean_peak_loc"][0] - true_peak_loc[idx, 0]) < nb_stds * np.sqrt(uq_data_idx["var_peak_loc"][0]))
                    and
                    (np.abs(uq_data_idx["mean_peak_loc"][1] - true_peak_loc[idx, 1]) < nb_stds * np.sqrt(uq_data_idx["var_peak_loc"][1])))

        # compute prad stats
        true_prad_core[idx] = tomo_helps.compute_radiated_power(ground_truth_downsampled, mask_core, uq_data_idx["sampling"])
        for j, nb_stds in enumerate(np.arange(1, 4)):
            prad_within_n_stds[idx, j] = np.abs(uq_data_idx["mean_prad_core"] - true_prad_core[idx]) < nb_stds * np.sqrt(uq_data_idx["var_prad_core"])
        prad_std_values[idx] = np.sqrt(uq_data_idx["var_prad_core"])
        prad_rel_error_mean[idx] = (uq_data_idx["mean_prad_core"] - true_prad_core[idx]) / true_prad_core[idx]
        prad_rel_error_map[idx] = (uq_data_idx["prad_map_core"] - true_prad_core[idx]) / true_prad_core[idx]
        for j in range(5):
            prad_within_quantiles[idx, j] = (
                    (np.quantile(uq_data_idx["prads_core"], quantile_marks[-(j + 1)]) > true_prad_core[idx])
                    and
                    (np.quantile(uq_data_idx["prads_core"], quantile_marks[j]) < true_prad_core[idx])
            )
        postprocessed_uq_data = {"pixels_within_quantiles": pixels_within_quantiles, "pixels_within_n_stds": pixels_within_n_stds, "mse_map": mse_map,
                              "mse_mean": mse_mean, "mape_map": mape_map, "mape_mean": mape_mean,
                              "true_peak_loc": true_peak_loc, "peak_within_n_stds": peak_within_n_stds, "peak_distance_from_mean": peak_distance_from_mean,
                              "true_prad_core": true_prad_core, "prad_within_n_stds": prad_within_n_stds, "prad_std_values": prad_std_values,
                              "prad_rel_error_map": prad_rel_error_map, "prad_rel_error_mean": prad_rel_error_mean, "prad_within_quantiles": prad_within_quantiles,
                              "quantile_marks": quantile_marks}
        np.save(uq_data_dir + "/postprocessed_data.npy", postprocessed_uq_data)


if __name__ == '__main__':

    argv = sys.argv
    if len(argv) == 1:
        phantom_indices = np.arange(0, 900)
    elif len(argv) == 3:
        phantom_indices = np.arange(int(argv[1]), int(argv[2]))
        print("Running pipeline on phantoms {}-{}".format(int(argv[1]), int(argv[2])))
    else:
        raise ValueError("Number of passed arguments must be either 1 or 3")

    # define directory where phantoms are stored
    phantom_dir = '../../dataset_generation/sxr_samples'

    # Noise model N1, noise level 5%
    sigma_err = np.load('../hyperparameter_tuning/prior_hyperparameters_tuning/tuning_data/reg_param_tuning_sigma005/sigma_err.npy')
    reg_param_median = 0.1
    # saving directory
    saving_dir = 'sigma005/'
    if not os.path.isdir(saving_dir):
        os.mkdir(saving_dir)
    run_uq_study(sigma_err=sigma_err, reg_param=reg_param_median,
                 phantom_indices=phantom_indices,
                 phantom_dir=phantom_dir, saving_dir=saving_dir)

    # Noise model N2, noise level 10%
    sigma_err = np.load('../hyperparameter_tuning/prior_hyperparameters_tuning/tuning_data/reg_param_tuning_sigma01/sigma_err.npy')
    reg_param_median = 0.1
    # saving directory
    saving_dir = 'sigma01/'
    if not os.path.isdir(saving_dir):
        os.mkdir(saving_dir)
    run_uq_study(sigma_err=sigma_err, reg_param=reg_param_median,
                 phantom_indices=phantom_indices,
                 phantom_dir=phantom_dir, saving_dir=saving_dir)

    # Noise model N3, noise level 5% + 5% signal-dependent
    sigma_err = np.load('../hyperparameter_tuning/prior_hyperparameters_tuning/tuning_data/reg_param_tuning_sigma005005/sigma_err.npy')
    reg_param_median = 0.1
    # saving directory
    saving_dir = 'sigma005005/'
    if not os.path.isdir(saving_dir):
        os.mkdir(saving_dir)
    run_uq_study(sigma_err=sigma_err, reg_param=reg_param_median,
                 phantom_indices=phantom_indices,
                 phantom_dir=phantom_dir, saving_dir=saving_dir)
