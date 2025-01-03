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


if __name__ == '__main__':
    # run reg_param tuning routine on training phantoms

    argv = sys.argv
    if len(argv) == 1:
        phantom_indices = np.arange(0, 900)
    elif len(argv) == 3:
        phantom_indices = np.arange(int(argv[1]), int(argv[2]))
        print("Running pipeline on phantoms {}-{}".format(int(argv[1]), int(argv[2])))
    else:
        raise ValueError("Number of passed arguments must be either 1 or 3")

    # define directory where phantoms are stored
    phantom_dir = 'sxr_samples_fine_anisotropic_new_bounds'

    # # Noise level 5%
    # sigma_level = 0.05
    # saving_dir_sigma005 = 'hyperparam_tuning/reg_param_tuning_fine_anisotropic_sigma005/'
    # reg_param_tuning_train_phantoms(sigma_level, phantom_dir, saving_dir_sigma005)

    # # Noise level 10%
    # sigma_level = 0.1
    # saving_dir_sigma01 = 'hyperparam_tuning/reg_param_tuning_fine_anisotropic_sigma01/'
    # reg_param_tuning_train_phantoms(sigma_level, phantom_dir, saving_dir_sigma01)

    # reg_param_tuning directory
    # reg_param_tuning_dir = 'hyperparam_tuning/reg_param_tuning_fine_anisotropic_newbounds_sigma005/'
    # sigma_err = np.load('hyperparam_tuning/reg_param_tuning_fine_anisotropic_newbounds_sigma005/sigma_err.npy')
    # reg_param_median = np.load('hyperparam_tuning/reg_param_tuning_fine_anisotropic_newbounds_sigma005/reg_param_median.npy')
    # # saving directory
    # saving_dir = 'uq_study_results/sigma005/'
    # if not os.path.isdir(saving_dir):
    #     os.mkdir(saving_dir)
    # run_uq_study(sigma_err=sigma_err, reg_param=reg_param_median,
    #              phantom_indices=phantom_indices,
    #              phantom_dir=phantom_dir, saving_dir=saving_dir)

    reg_param_tuning_dir = 'hyperparam_tuning/reg_param_tuning_fine_anisotropic_newbounds_sigma01/'
    sigma_err = np.load('hyperparam_tuning/reg_param_tuning_fine_anisotropic_newbounds_sigma01/sigma_err.npy')
    reg_param_median = np.load('hyperparam_tuning/reg_param_tuning_fine_anisotropic_newbounds_sigma01/reg_param_median.npy')
    # saving directory
    saving_dir = 'uq_study_results/sigma01/'
    if not os.path.isdir(saving_dir):
        os.mkdir(saving_dir)
    run_uq_study(sigma_err=sigma_err, reg_param=reg_param_median,
                 phantom_indices=phantom_indices,
                 phantom_dir=phantom_dir, saving_dir=saving_dir)
