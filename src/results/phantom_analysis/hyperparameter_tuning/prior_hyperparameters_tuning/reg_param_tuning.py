import scipy.sparse as sp
import numpy as np
import copy
import matplotlib.pyplot as plt
import os
import skimage.transform as skimt
import sys

import src.tomo_fusion.tools.helpers as tomo_helps
import src.tomo_fusion.functionals_definition as fct_def
import src.tomo_fusion.hyperparameter_tuning as hyper_tune
import src.tomo_fusion.bayesian_computations as bcomp


def reg_param_tuning_train_phantoms(sigma_level, phantom_dir, saving_dir):
    # training indices
    train_indices = np.arange(900, 1000)
    # Load phantom data
    psis = np.load(phantom_dir+'/psis.npy')
    sxr_samples = np.load(phantom_dir+'/sxr_samples_with_background.npy')
    alphas = np.load(phantom_dir+'/alpha_random_values.npy')
    trim_val = np.load(phantom_dir+'/trimming_values.npy')

    # load forward model
    fwd_model = sp.load_npz(
        '../../../../tomo_fusion/forward_model/geometry_matrices/sparse_geometry_matrix_sxr_fine_grid.npz')
    reconstruction_shape = (120,40)

    # compute average of tomographic measurements on training phantoms
    tomo_data_mean = np.zeros(fwd_model.shape[0])
    for i in train_indices:
        tomo_data_mean += fwd_model.dot(sxr_samples[i,:,:].flatten()) / train_indices.size

    # define noise intensity as a function of average tomographic measurement (5%, 10%, ...)
    if isinstance(sigma_level, float):
        sigma_err = sigma_level * np.mean(tomo_data_mean)
    elif isinstance(sigma_level, list) and len(sigma_level) == 2:
        sigma_err = [sigma_level[0] * np.mean(tomo_data_mean), sigma_level[1]]

    reg_param_tuning_data = []

    for idx in train_indices:
        print("Phantom ", idx)
        ground_truth = copy.deepcopy(sxr_samples[idx, :, :].squeeze())
        psi = psis[idx, :, :]
        alpha = alphas[idx]
        trim_val_ = trim_val[idx, :]
        mask_core = tomo_helps.define_core_mask(psi=psi, dim_shape=reconstruction_shape, trim_values_x=trim_val_)

        # anisotropic regularization functional
        reg_fct_type = "anisotropic"
        # Define functionals
        f, g = fct_def.define_loglikelihood_and_logprior(ground_truth=ground_truth, psi=psi,
                                                         sigma_err=sigma_err, reg_fct_type=reg_fct_type,
                                                         alpha=alpha, plot=False,
                                                         seed=idx)
        # tune hyperparameters
        reg_param_data_idx = hyper_tune.reg_param_tuning(f, g, tuning_techniques=["GT"], ground_truth=ground_truth,
                                                         with_pos_constraint=True, clipping_mask=mask_core,
                                                         cv_strategy="random",
                                                         reg_params=np.logspace(-4, 1, 21), plot=False)

        reg_param_tuning_data.append(reg_param_data_idx)

    best_performing_hyper_params = np.zeros(len(reg_param_tuning_data))
    for i in range(best_performing_hyper_params.size):
        mse_argmin = np.argmin(reg_param_tuning_data[i]['GT'][2, :])
        best_performing_hyper_params[i] = reg_param_tuning_data[i]['GT'][0, mse_argmin]

    nb_occurrences = np.zeros(reg_param_tuning_data[0]["GT"].shape[1])
    for i in range(len(reg_param_tuning_data)):
        j = np.argmin(reg_param_tuning_data[i]['GT'][2, :])
        nb_occurrences[j] += 1

    plt.figure()
    plt.plot(nb_occurrences)
    plt.xticks([0,4,8,12,16,20], [r"$10^{-4}$", r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$", r"$10^{0}$", r"$10^{1}$"])
    plt.title(r"Regularization parameter $\lambda$ minimizing MSE (anisotropic)")
    plt.xlabel(r"$\lambda$")
    plt.ylabel("# occurrences")
    plt.show()

    print("Statistics on best performing regularization parameter (anisotropic)\n")
    print("Mean ", np.mean(best_performing_hyper_params))
    print("Median ", np.median(best_performing_hyper_params))
    print("Standard deviation ", np.std(best_performing_hyper_params))

    # Define regularization parameter as average/median best performing value
    reg_param_mean = np.mean(best_performing_hyper_params)
    reg_param_median = np.median(best_performing_hyper_params)

    # ratio of MSE with average regularization parameter vs best MSE
    factors_average_wrt_best = np.zeros(train_indices.size)
    # ratio of MSE with median regularization parameter vs best MSE
    factors_median_wrt_best = np.zeros(train_indices.size)

    # compute MSE for average value of regularization parameter
    for i, idx in enumerate(train_indices):
        ground_truth = copy.deepcopy(sxr_samples[idx, :, :].squeeze())
        psi = psis[idx, :, :]
        alpha = alphas[idx]
        trim_val_ = trim_val[idx, :]
        mask_core = tomo_helps.define_core_mask(psi=psi, dim_shape=reconstruction_shape, trim_values_x=trim_val_)

        # anisotropic regularization functional
        reg_fct_type = "anisotropic"
        # Define functionals
        f, g = fct_def.define_loglikelihood_and_logprior(ground_truth=ground_truth, psi=psi,
                                                         sigma_err=sigma_err, reg_fct_type=reg_fct_type,
                                                         alpha=alpha, plot=False,
                                                         seed=idx)

        # compute MSE with reg_param fixed to average value
        map = bcomp.compute_MAP(f, g, reg_param_mean, with_pos_constraint=True, clipping_mask=mask_core)
        mse_avg_reg_param = np.mean((map-skimt.resize(ground_truth, f.dim_shape[1:], anti_aliasing=False, mode='edge'))**2)
        # compute MSE with reg_param fixed to median value
        map = bcomp.compute_MAP(f, g, reg_param_median, with_pos_constraint=True, clipping_mask=mask_core)
        mse_median_reg_param = np.mean((map-skimt.resize(ground_truth, f.dim_shape[1:], anti_aliasing=False, mode='edge'))**2)

        # compute factors
        factors_average_wrt_best[i] = mse_avg_reg_param / reg_param_tuning_data[i]['GT'][2, np.argmin(reg_param_tuning_data[i]['GT'][2, :])]
        factors_median_wrt_best[i] = mse_median_reg_param / reg_param_tuning_data[i]['GT'][2, np.argmin(reg_param_tuning_data[i]['GT'][2, :])]

    # save all results
    np.save(saving_dir+'tuning_data.npy', np.array(reg_param_tuning_data))
    np.save(saving_dir+'best_hyperparams.npy', best_performing_hyper_params)
    np.save(saving_dir+'nb_occurrences.npy', nb_occurrences)
    np.save(saving_dir+'factors_avg_wrt_best.npy', factors_average_wrt_best)
    np.save(saving_dir + 'factors_median_wrt_best.npy', factors_median_wrt_best)
    np.save(saving_dir+'reg_param_mean.npy', reg_param_mean)
    np.save(saving_dir + 'reg_param_median.npy', reg_param_median)
    np.save(saving_dir + 'sigma_level.npy', sigma_level)
    np.save(saving_dir + 'sigma_err.npy', sigma_err)


if __name__ == '__main__':
    # run reg_param tuning routine on training phantoms

    # define directory where phantoms are stored
    phantom_dir = '../../../dataset_generation/sxr_samples'

    # # Noise level 5%
    # sigma_level = 0.05
    # saving_dir = 'hyperparam_tuning/reg_param_tuning_fine_anisotropic_newbounds_sigma005/'

    # Noise level 10%
    # sigma_level = 0.1
    # saving_dir = 'hyperparam_tuning/reg_param_tuning_fine_anisotropic_newbounds_sigma01/'

    # Noise level 5% plus 5% signal-dependent
    sigma_level = [0.05, 0.05]
    saving_dir = 'tuning_data/reg_param_tuning_fine_anisotropic_newbounds_sigma005005new/'

    if not os.path.isdir(saving_dir):
        os.mkdir(saving_dir)
    # analyze training phantoms
    reg_param_tuning_train_phantoms(sigma_level, phantom_dir, saving_dir)