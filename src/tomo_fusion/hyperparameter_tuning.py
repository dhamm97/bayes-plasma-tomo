import numpy as np
import skimage.transform as skimt
from skimage.metrics import structural_similarity as ssim
import os
import warnings
import copy

from src.tomo_fusion.reg_param_est import RegParamMLE, ProxFuncMoreau
import pyxu.abc as pxa
import pyxu.opt.stop as pyxst
import pyxu.operator as pyxop

import src.tomo_fusion.functionals_definition as fct_def
import src.tomo_fusion.bayesian_computations as bcomp
from src.tomo_fusion.tools import plotting_fcts as plt_tools

dirname = os.path.dirname(__file__)


def run_sapg(f_sapg, g_sapg, lambda_min=1e-6, lambda_max=1e0,
             sapg_max_iter=int(1e4), warm_start=int(1e3),
             seed=0, plot=False):
    # Run Stochastic Approximation Proximal Gradient (SAPG) algorithm
    # fix seed
    rng = np.random.default_rng(seed=seed)
    # homogeneity factors of involved functionals (squared L2 norms, factor is 2)
    homo_factors = 2
    # initialization
    lambda0 = 10 ** ((np.log10(lambda_max)-np.log10(lambda_min)) / 2 + np.log10(lambda_min))
    # set step size ensuring stability for largest possible lambda (most constraining case)
    objective_func_max = f_sapg + lambda_max * g_sapg
    gamma = 0.98 / objective_func_max.diff_lipschitz
    # Initialize SAPG
    delta0 = [5*1e-2]
    stop_crit_SAPG = pyxst.MaxIter(sapg_max_iter)
    sapg = RegParamMLE(f=f_sapg, g=[g_sapg], homo_factors=[homo_factors], verbosity=100)
    sapg.fit(mode=pxa.SolverMode.MANUAL, x0=np.zeros(f_sapg.dim_shape),
             theta0=lambda0, theta_min=lambda_min, theta_max=lambda_max, delta0=delta0,
             warm_start=warm_start, gamma=gamma, log_scale=True,
             stop_crit=stop_crit_SAPG, batch_size=1, rng=rng)
    # Run SAPG
    lambda_list = np.zeros((1, sapg_max_iter))
    it = 0
    print("Runnning SAPG")
    for data in sapg.steps(n=sapg_max_iter):
        if (it+1) % 1000 == 0:
            print("iteration {}".format(it+1))
        lambda_list[:, it] = data["theta"]
        it += 1

    if plot:
        plt_tools.plot_sapg_run(lambda_min, lambda_max, lambda_list)

    return lambda_list


def reg_param_tuning(f, g, with_pos_constraint=False,
                     clipping_mask=None, ground_truth=None,
                     reg_params=np.logspace(-6,0,7),
                     tuning_techniques=["SAPG"],
                     sapg_max_iter=int(1e4), sapg_warm_start=int(1e3),
                     cv_strategy="random",
                     seed=0, plot=False, plot_ssim=False):
    """

    :param f:
    :param g:
    :param with_pos_constraint:
    :param ground_truth:
    :param reg_params:
    :param tuning_techniques: "SAPG", "GT", "CV_single", "CV_full", "CV_by_camera"
    :param sapg_max_iter:
    :param sapg_warm_start:
    :param seed:
    :param plot:
    :return:
    """

    reg_param_tuning_data = {}

    if ground_truth is not None:
        if ground_truth.shape == (240, 80):
            ground_truth = skimt.resize(ground_truth, f.dim_shape[1:], anti_aliasing=False, mode='edge')

    if 'SAPG' in tuning_techniques:
        f_sapg = f
        if with_pos_constraint:
            pos_constraint = pyxop.PositiveOrthant(dim_shape=f.dim_shape)
            f_sapg += ProxFuncMoreau(pos_constraint, mu=1e-3)
        lambdas = run_sapg(f_sapg, g, lambda_min=np.min(reg_params), lambda_max=np.max(reg_params),
                 sapg_max_iter=sapg_max_iter, warm_start=sapg_warm_start,
                 seed=seed, plot=plot)
        reg_param_tuning_data['SAPG'] = lambdas

    if 'GT' in tuning_techniques:
        assert ground_truth is not None, 'Ground truth must be provided for tuning technique `GT`'
        print("Tuning regularization parameter: comparison to ground truth (`GT`)\n")
        # compute stats for each reg_param provided
        tomo_data_MSE, MSE, ssim_val = np.zeros(reg_params.size), np.zeros(reg_params.size), np.zeros(reg_params.size)
        for i, reg_param in enumerate(reg_params):
            im_MAP = bcomp.compute_MAP(f, g, reg_param, with_pos_constraint, clipping_mask, show_progress=False)
            #tomo_data_MSE[i] = f(im_MAP) * (2 * f.sigma_err**2) / f.dim_size
            tomo_data_MSE[i] = np.mean((f.forward_model_linop(im_MAP)-f.noisy_tomo_data)**2)
            MSE[i] = np.mean((ground_truth - im_MAP) ** 2)
            ssim_val[i] = ssim(ground_truth, im_MAP, data_range=im_MAP.max() - im_MAP.min())
        reg_param_tuning_data['GT'] = np.vstack((reg_params, tomo_data_MSE, MSE, ssim_val))

    if "CV_single" in tuning_techniques:
        print("Tuning regularization parameter: 1-fold cross validation (`CV_single`)\n")
        # define cv_single data fidelity functional
        f_cv = fct_def.define_loglikelihood_cv(f, cv_type="CV_single", cv_strategy=cv_strategy)
        # compute stats for each reg_param provided
        tomo_data_MSE, MSE, ssim_val = np.zeros(reg_params.size), np.zeros(reg_params.size), np.zeros(reg_params.size)
        for i, reg_param in enumerate(reg_params):
            im_MAP = bcomp.compute_MAP(f_cv, g, reg_param, with_pos_constraint, clipping_mask, show_progress=False)
            #tomo_data_MSE[i] = (f(im_MAP) - f_cv(im_MAP)) * (2 * f.sigma_err ** 2) / f_cv.cv_test_idx.size
            tomo_data_MSE[i] = np.mean(
                (f.forward_model_linop(im_MAP)[f_cv.cv_test_idx] - f.noisy_tomo_data[f_cv.cv_test_idx])**2
            )
            if ground_truth is not None:
                MSE[i] = np.mean((ground_truth - im_MAP) ** 2)
                ssim_val[i] = ssim(ground_truth, im_MAP, data_range=im_MAP.max() - im_MAP.min())
        reg_param_tuning_data['CV_single'] = np.vstack((reg_params, tomo_data_MSE, MSE, ssim_val))

    if "CV_full" in tuning_techniques:
        print("Tuning regularization parameter: 5-fold cross validation (`CV_full`)\n")
        # define cv_single data fidelity functional
        f_cv = fct_def.define_loglikelihood_cv(f, cv_type="CV_full", cv_strategy=cv_strategy)
        # compute stats for each reg_param provided
        tomo_data_MSE, MSE, ssim_val = np.zeros(reg_params.size), np.zeros(reg_params.size), np.zeros(reg_params.size)
        for cv_round in range(len(f_cv)):
            for i, reg_param in enumerate(reg_params):
                im_MAP = bcomp.compute_MAP(f_cv[cv_round], g, reg_param, with_pos_constraint, clipping_mask, show_progress=False)
                #tomo_data_MSE[i] += (f(im_MAP) - f_cv[cv_round](im_MAP)) * (2 * f.sigma_err ** 2) / f_cv[cv_round].cv_test_idx.size
                tomo_data_MSE[i] += np.mean(
                    (f.forward_model_linop(im_MAP)[f_cv[cv_round].cv_test_idx] - f.noisy_tomo_data[f_cv[cv_round].cv_test_idx]) ** 2
                )
                if ground_truth is not None:
                    MSE[i] += np.mean((ground_truth - im_MAP) ** 2)
                    ssim_val[i] += ssim(ground_truth, im_MAP, data_range=im_MAP.max() - im_MAP.min())
                import matplotlib.pyplot as plt
                #print("CV fold {}, reg_param {}".format(cv_round, i))
                # plt.figure(figsize=(7,3))
                # # plt.plot(f.noisy_tomo_data[~f_cv[cv_round].cv_test_idx], '.r')
                # # plt.plot(f.noisy_tomo_data[f_cv[cv_round].cv_test_idx], '*r')
                # # plt.plot(f.forward_model_linop(im_MAP)[~f_cv[cv_round].cv_test_idx], '.b')
                # # plt.plot(f.forward_model_linop(im_MAP)[f_cv[cv_round].cv_test_idx], '*b')
                # plt.plot(f.noisy_tomo_data, '.r')
                # plt.plot(f.forward_model_linop(im_MAP), '.b')
                # plt.plot(f_cv[cv_round].cv_test_idx, f.noisy_tomo_data[f_cv[cv_round].cv_test_idx], '*r')
                # plt.plot(f_cv[cv_round].cv_test_idx, f.forward_model_linop(im_MAP)[f_cv[cv_round].cv_test_idx], '*b')
                # plt.title("CV fold {}, reg_param {}".format(cv_round, i))
                # plt.show()
        tomo_data_MSE /= len(f_cv)
        MSE /= len(f_cv)
        ssim_val /= len(f_cv)
        reg_param_tuning_data['CV_full'] = np.vstack((reg_params, tomo_data_MSE, MSE, ssim_val))

    if plot:
        plt_tools.plot_hyperparam_tuning_data(reg_param_tuning_data, param="reg_param", plot_ssim=plot_ssim)

    # return data
    return reg_param_tuning_data


def _redefine_anis_param_logprior(g, alpha_new):
    if ('Anis' in g._name) and ('Mfi' not in g._name):
        if g.freezing_arr is not None:
            # redefine frozen diffusion coefficient for new alpha
            g.diffusion_coefficient.alpha = alpha_new
            g.diffusion_coefficient.freeze(g.freezing_arr)
            # assemble matrix-based version of diffusion operator
            g._assemble_matrix_based()
            g.matrix_based_impl = True
        else:
            raise ValueError("Functional g must have a `freezing_arr`")
    else:
        raise ValueError("Functional g must be a non-MFI anisotropic functional")
    return g


def anis_param_tuning(f, g, reg_param, with_pos_constraint=False,
                      clipping_mask=None, ground_truth=None,
                     anis_params=np.logspace(-4,0,5),
                     tuning_techniques='CV_single',
                     cv_strategy=["random"],
                     seed=0, plot=False, plot_ssim=False):
    """

    :param f:
    :param g:
    :param with_pos_constraint:
    :param ground_truth:
    :param reg_params:
    :param tuning_techniques: "SAPG", "GT", "CV_single", "CV_full", "CV_by_camera"
    :param sapg_max_iter:
    :param sapg_warm_start:
    :param seed:
    :param plot:
    :return:
    """

    # save "true" alpha corresponding to prior used to generate phantom
    true_val_alpha = g.diffusion_coefficient.alpha

    anis_param_tuning_data = {}

    if not isinstance(cv_strategy, list):
        cv_strategy = [cv_strategy]

    if ground_truth is not None:
        if ground_truth.shape == (240, 80):
            ground_truth = skimt.resize(ground_truth, f.dim_shape[1:], anti_aliasing=False, mode='edge')

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        if 'GT' in tuning_techniques:
            assert ground_truth is not None, 'Ground truth must be provided for tuning technique `GT`'
            print("Tuning regularization parameter: comparison to ground truth (`GT`)\n")
            # compute stats for each anis_param provided
            tomo_data_MSE, MSE, ssim_val = np.zeros(anis_params.size), np.zeros(anis_params.size), np.zeros(anis_params.size)
            for i, anis_param in enumerate(anis_params):
                g_ = _redefine_anis_param_logprior(g, anis_param)
                im_MAP = bcomp.compute_MAP(f, g_, reg_param, with_pos_constraint, clipping_mask, show_progress=False)
                # tomo_data_MSE[i] = f(im_MAP) * (2 * f.sigma_err**2) / f.dim_size
                tomo_data_MSE[i] = np.mean((f.forward_model_linop(im_MAP) - f.noisy_tomo_data) ** 2)
                MSE[i] = np.mean((ground_truth - im_MAP) ** 2)
                ssim_val[i] = ssim(ground_truth, im_MAP, data_range=im_MAP.max() - im_MAP.min())
            anis_param_tuning_data['GT'] = np.vstack((anis_params, tomo_data_MSE, MSE, ssim_val))

        if "CV_single" in tuning_techniques:
            print("Tuning anisotropy parameter: 1-fold cross validation (`CV_single`)\n")
            for cv_strategy_ in cv_strategy:
                # define cv_single data fidelity functional
                f_cv = fct_def.define_loglikelihood_cv(f, cv_type="CV_single", cv_strategy=cv_strategy_)
                # compute stats for each anis_param provided
                tomo_data_MSE, MSE, ssim_val = np.zeros(anis_params.size), np.zeros(anis_params.size), np.zeros(anis_params.size)
                for i, anis_param in enumerate(anis_params):
                    g_ = _redefine_anis_param_logprior(g, anis_param)
                    im_MAP = bcomp.compute_MAP(f_cv, g_, reg_param, with_pos_constraint, clipping_mask, show_progress=False)
                    # tomo_data_MSE[i] = (f(im_MAP) - f_cv(im_MAP)) * (2 * f.sigma_err ** 2) / f_cv.cv_test_idx.size
                    tomo_data_MSE[i] = np.mean(
                        (f.forward_model_linop(im_MAP)[f_cv.cv_test_idx] - f.noisy_tomo_data[f_cv.cv_test_idx]) ** 2
                    )
                    if ground_truth is not None:
                        MSE[i] = np.mean((ground_truth - im_MAP) ** 2)
                        ssim_val[i] = ssim(ground_truth, im_MAP, data_range=im_MAP.max() - im_MAP.min())
                anis_param_tuning_data['CV_single'+'_'+cv_strategy_] = np.vstack((anis_params, tomo_data_MSE, MSE, ssim_val))

        if "CV_full" in tuning_techniques:
            print("Tuning regularization parameter: 5-fold cross validation (`CV_full`)\n")
            for cv_strategy_ in cv_strategy:
                # define cv_single data fidelity functional
                f_cv = fct_def.define_loglikelihood_cv(f, cv_type="CV_full", cv_strategy=cv_strategy_)
                # compute stats for each anis_param provided
                tomo_data_MSE, MSE, ssim_val = np.zeros(anis_params.size), np.zeros(anis_params.size), np.zeros(anis_params.size)
                for cv_round in range(len(f_cv)):
                    for i, anis_param in enumerate(anis_params):
                        g_ = _redefine_anis_param_logprior(g, anis_param)
                        im_MAP = bcomp.compute_MAP(f_cv[cv_round], g_, reg_param, with_pos_constraint, clipping_mask, show_progress=False)
                        # tomo_data_MSE[i] += (f(im_MAP) - f_cv[cv_round](im_MAP)) * (2 * f.sigma_err ** 2) / f_cv[cv_round].cv_test_idx.size
                        tomo_data_MSE[i] += np.mean(
                            (f.forward_model_linop(im_MAP)[f_cv[cv_round].cv_test_idx] - f.noisy_tomo_data[
                                f_cv[cv_round].cv_test_idx]) ** 2
                        )
                        if ground_truth is not None:
                            MSE[i] += np.mean((ground_truth - im_MAP) ** 2)
                            ssim_val[i] += ssim(ground_truth, im_MAP, data_range=im_MAP.max() - im_MAP.min())
                tomo_data_MSE /= len(f_cv)
                MSE /= len(f_cv)
                ssim_val /= len(f_cv)
                anis_param_tuning_data['CV_full'+'_'+cv_strategy_] = np.vstack((anis_params, tomo_data_MSE, MSE, ssim_val))

        # modify g's alpha back to original value
        g = _redefine_anis_param_logprior(g, true_val_alpha)

    if plot:
        plt_tools.plot_hyperparam_tuning_data(anis_param_tuning_data, param="anis_param", true_param_val=true_val_alpha, plot_ssim=plot_ssim)

    # return data
    return anis_param_tuning_data
