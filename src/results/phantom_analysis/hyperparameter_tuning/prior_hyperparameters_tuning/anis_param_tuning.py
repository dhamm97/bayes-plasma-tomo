import scipy.sparse as sp
import numpy as np
import copy
import matplotlib.pyplot as plt
import os
import skimage.transform as skimt

import src.tomo_fusion.tools.helpers as tomo_helps
import src.tomo_fusion.functionals_definition as fct_def
import src.tomo_fusion.hyperparameter_tuning as hyper_tune
import src.tomo_fusion.bayesian_computations as bcomp


def anis_param_tuning_train_phantoms(sigma_err, phantom_dir, saving_dir, reg_param, cv_strategy="random"):
    # training indices
    train_indices = np.arange(900, 1000)
    # Load phantom data
    psis = np.load(phantom_dir+'/psis.npy')
    sxr_samples = np.load(phantom_dir+'/sxr_samples_with_background.npy')
    alphas = np.load(phantom_dir+'/alpha_random_values.npy')
    trim_val = np.load(phantom_dir+'/trimming_values.npy')

    reconstruction_shape = (120,40)

    anis_param_tuning_data = []

    # initialize performance factors
    factors_cv_wrt_best_random, factors_true_wrt_best_random, factors_cv_wrt_true_random, factors_cv_wrt_avg_random = (
        np.zeros(train_indices.size), np.zeros(train_indices.size), np.zeros(train_indices.size), np.zeros(train_indices.size)
    )
    factors_cv_wrt_best_random_full, factors_true_wrt_best_random_full, factors_cv_wrt_true_random_full, factors_cv_wrt_avg_random_full = (
        np.zeros(train_indices.size), np.zeros(train_indices.size), np.zeros(train_indices.size), np.zeros(train_indices.size)
    )
    factors_cv_wrt_best_camera, factors_true_wrt_best_camera, factors_cv_wrt_true_camera, factors_cv_wrt_avg_camera = (
        np.zeros(train_indices.size), np.zeros(train_indices.size), np.zeros(train_indices.size), np.zeros(train_indices.size)
    )
    factors_cv_wrt_best_camera_full, factors_true_wrt_best_camera_full, factors_cv_wrt_true_camera_full, factors_cv_wrt_avg_camera_full = (
        np.zeros(train_indices.size), np.zeros(train_indices.size), np.zeros(train_indices.size), np.zeros(train_indices.size)
    )
    mse_best_alpha, mse_true_alpha, mse_avg_alpha = np.zeros(train_indices.size), np.zeros(train_indices.size), np.zeros(train_indices.size)
    mse_cv_random, mse_cv_random_full, mse_cv_camera, mse_cv_camera_full = (
        np.zeros(train_indices.size), np.zeros(train_indices.size), np.zeros(train_indices.size), np.zeros(train_indices.size)
    )

    for i, idx in enumerate(train_indices):
        print("phantom ", idx)
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

        # compute MSE
        map = bcomp.compute_MAP(f, g, reg_param, with_pos_constraint=True, clipping_mask=mask_core)
        mse_true_alpha[i] = np.mean((map-skimt.resize(ground_truth, f.dim_shape[1:], anti_aliasing=False, mode='edge'))**2)

        # tune anisotropic parameter
        anis_param_data = hyper_tune.anis_param_tuning(f, g, reg_param,
                                                       tuning_techniques=["GT", "CV_single", "CV_full"],
                                                       ground_truth=ground_truth,
                                                       with_pos_constraint=True, clipping_mask=mask_core,
                                                       cv_strategy=cv_strategy,
                                                       anis_params=np.logspace(-4, 0, 13), plot=False)

        anis_param_tuning_data.append(anis_param_data)

        # compute stats for average value of anisotropic parameter alpha=1e-2
        g = hyper_tune._redefine_anis_param_logprior(g, alpha_new=1e-2)
        # compute MSE
        map = bcomp.compute_MAP(f, g, reg_param, with_pos_constraint=True, clipping_mask=mask_core)
        mse_avg_alpha[i] = np.mean((map - skimt.resize(ground_truth, f.dim_shape[1:], anti_aliasing=False, mode='edge')) ** 2)

        # store MSE corresponding to cv selection
        mse_cv_random[i] = anis_param_tuning_data[i]['GT'][2, np.argmin(anis_param_tuning_data[i]['CV_single_random'][1, :])]
        mse_cv_random_full[i] = anis_param_tuning_data[i]['GT'][2, np.argmin(anis_param_tuning_data[i]['CV_full_random'][1, :])]
        mse_cv_camera[i] = anis_param_tuning_data[i]['GT'][2, np.argmin(anis_param_tuning_data[i]['CV_single_by_camera'][1, :])]
        mse_cv_camera_full[i] = anis_param_tuning_data[i]['GT'][2, np.argmin(anis_param_tuning_data[i]['CV_full_by_camera'][1, :])]

        # compute factors
        mse_best_alpha[i] = anis_param_tuning_data[i]['GT'][2, np.argmin(anis_param_tuning_data[i]['GT'][2, :])]
        # CV single random
        factors_cv_wrt_best_random[i] = anis_param_tuning_data[i]['GT'][2, np.argmin(anis_param_tuning_data[i]['CV_single_random'][1, :])] / mse_best_alpha[i]
        factors_true_wrt_best_random[i] = mse_true_alpha[i] / mse_best_alpha[i]
        factors_cv_wrt_true_random[i] = anis_param_tuning_data[i]['GT'][2, np.argmin(anis_param_tuning_data[i]['CV_single_random'][1, :])] / mse_true_alpha[i]
        factors_cv_wrt_avg_random[i] = anis_param_tuning_data[i]['GT'][2, np.argmin(anis_param_tuning_data[i]['CV_single_random'][1, :])] / mse_avg_alpha[i]
        # CV full random
        factors_cv_wrt_best_random_full[i] = anis_param_tuning_data[i]['GT'][2, np.argmin(anis_param_tuning_data[i]['CV_full_random'][1, :])] / mse_best_alpha[i]
        factors_true_wrt_best_random_full[i] = mse_true_alpha[i] / mse_best_alpha[i]
        factors_cv_wrt_true_random_full[i] = anis_param_tuning_data[i]['GT'][2, np.argmin(anis_param_tuning_data[i]['CV_full_random'][1, :])] / mse_true_alpha[i]
        factors_cv_wrt_avg_random_full[i] = anis_param_tuning_data[i]['GT'][2, np.argmin(anis_param_tuning_data[i]['CV_full_random'][1, :])] / mse_avg_alpha[i]
        # by camera single random
        factors_cv_wrt_best_camera[i] = anis_param_tuning_data[i]['GT'][2, np.argmin(anis_param_tuning_data[i]['CV_single_by_camera'][1, :])] / mse_best_alpha[i]
        factors_true_wrt_best_camera[i] = mse_true_alpha[i] / mse_best_alpha[i]
        factors_cv_wrt_true_camera[i] = anis_param_tuning_data[i]['GT'][2, np.argmin(anis_param_tuning_data[i]['CV_single_by_camera'][1, :])] / mse_true_alpha[i]
        factors_cv_wrt_avg_camera[i] = anis_param_tuning_data[i]['GT'][2, np.argmin(anis_param_tuning_data[i]['CV_single_by_camera'][1, :])] / mse_avg_alpha[i]
        # by camera full random
        factors_cv_wrt_best_camera_full[i] = anis_param_tuning_data[i]['GT'][2, np.argmin(anis_param_tuning_data[i]['CV_full_by_camera'][1, :])] / mse_best_alpha[i]
        factors_true_wrt_best_camera_full[i] = mse_true_alpha[i] / mse_best_alpha[i]
        factors_cv_wrt_true_camera_full[i] = anis_param_tuning_data[i]['GT'][2, np.argmin(anis_param_tuning_data[i]['CV_full_by_camera'][1, :])] / mse_true_alpha[i]
        factors_cv_wrt_avg_camera_full[i] = anis_param_tuning_data[i]['GT'][2, np.argmin(anis_param_tuning_data[i]['CV_full_by_camera'][1, :])] / mse_avg_alpha[i]

    best_performing_anis_params = np.zeros((len(anis_param_tuning_data), 5))
    for i in range(best_performing_anis_params.shape[0]):
        best_performing_anis_params[i, 0] = anis_param_tuning_data[i]['GT'][
            0, np.argmin(anis_param_tuning_data[i]['GT'][2, :])]
        best_performing_anis_params[i, 1] = anis_param_tuning_data[i]['CV_single_random'][
            0, np.argmin(anis_param_tuning_data[i]['CV_single_random'][1, :])]
        best_performing_anis_params[i, 2] = anis_param_tuning_data[i]['CV_full_random'][
            0, np.argmin(anis_param_tuning_data[i]['CV_full_random'][1, :])]
        best_performing_anis_params[i, 3] = anis_param_tuning_data[i]['CV_single_by_camera'][
            0, np.argmin(anis_param_tuning_data[i]['CV_single_by_camera'][1, :])]
        best_performing_anis_params[i, 4] = anis_param_tuning_data[i]['CV_full_by_camera'][
            0, np.argmin(anis_param_tuning_data[i]['CV_full_by_camera'][1, :])]


    # save all results
    np.save(saving_dir+'tuning_data.npy', np.array(anis_param_tuning_data))
    np.save(saving_dir+'best_hyperparams.npy', best_performing_anis_params)
    np.save(saving_dir+'factors_cv_wrt_best_random.npy', factors_cv_wrt_best_random)
    np.save(saving_dir + 'factors_true_wrt_best_random.npy', factors_true_wrt_best_random)
    np.save(saving_dir + 'factors_cv_wrt_true_random.npy', factors_cv_wrt_true_random)
    np.save(saving_dir+'factors_cv_wrt_best_random_full.npy', factors_cv_wrt_best_random_full)
    np.save(saving_dir + 'factors_true_wrt_best_random_full.npy', factors_true_wrt_best_random_full)
    np.save(saving_dir + 'factors_cv_wrt_true_random_full.npy', factors_cv_wrt_true_random_full)
    np.save(saving_dir+'factors_cv_wrt_best_camera.npy', factors_cv_wrt_best_camera)
    np.save(saving_dir + 'factors_true_wrt_best_camera.npy', factors_true_wrt_best_camera)
    np.save(saving_dir + 'factors_cv_wrt_true_camera.npy', factors_cv_wrt_true_camera)
    np.save(saving_dir+'factors_cv_wrt_best_camera_full.npy', factors_cv_wrt_best_camera_full)
    np.save(saving_dir + 'factors_true_wrt_best_camera_full.npy', factors_true_wrt_best_camera_full)
    np.save(saving_dir + 'factors_cv_wrt_true_camera_full.npy', factors_cv_wrt_true_camera_full)
    np.save(saving_dir + 'mse_best_alpha.npy', mse_best_alpha)
    np.save(saving_dir + 'mse_true_alpha.npy', mse_true_alpha)
    np.save(saving_dir + 'mse_average_alpha.npy', mse_avg_alpha)
    np.save(saving_dir + 'mse_cv_random.npy', mse_cv_random)
    np.save(saving_dir + 'mse_cv_random_full.npy', mse_cv_random_full)
    np.save(saving_dir + 'mse_cv_camera.npy', mse_cv_camera)
    np.save(saving_dir + 'mse_cv_camera_full.npy', mse_cv_camera_full)
    np.save(saving_dir+'reg_param.npy', reg_param)
    np.save(saving_dir + 'sigma_err.npy', sigma_err)


if __name__ == '__main__':
    # run reg_param tuning routine on training phantoms

    # define directory where phantoms are stored
    phantom_dir = '../../../dataset_generation/sxr_samples_fine_anisotropic_new_bounds'

    # # Noise level 5%
    # sigma_level = 0.05
    # saving_dir_sigma005 = 'hyperparam_tuning/reg_param_tuning_fine_anisotropic_sigma005/'
    # reg_param_tuning_train_phantoms(sigma_level, phantom_dir, saving_dir_sigma005)

    # # Noise level 10%
    # sigma_level = 0.1
    # saving_dir_sigma01 = 'hyperparam_tuning/reg_param_tuning_fine_anisotropic_sigma01/'
    # reg_param_tuning_train_phantoms(sigma_level, phantom_dir, saving_dir_sigma01)

    # reg_param_tuning directory
    reg_param_tuning_dir = '../../../dataset_generation/hyperparam_tuning/reg_param_tuning_fine_anisotropic_newbounds_sigma005/'
    sigma_err = np.load(
        '../../../dataset_generation/hyperparam_tuning/reg_param_tuning_fine_anisotropic_newbounds_sigma005/sigma_err.npy')
    reg_param_mean = np.load(
        '../../../dataset_generation/hyperparam_tuning/reg_param_tuning_fine_anisotropic_newbounds_sigma005/reg_param_mean.npy')
    reg_param_median = np.load(
        '../../../dataset_generation/hyperparam_tuning/reg_param_tuning_fine_anisotropic_newbounds_sigma005/reg_param_median.npy')
    # saving directory
    saving_dir = '../../../dataset_generation/hyperparam_tuning/anis_param_tuning_fine_anisotropic_newbounds_sigma005/'
    if not os.path.isdir(saving_dir):
        os.mkdir(saving_dir)
    anis_param_tuning_train_phantoms(sigma_err=sigma_err, phantom_dir=phantom_dir,
                                     saving_dir=saving_dir, reg_param=reg_param_median, cv_strategy=["random", "by_camera"])
