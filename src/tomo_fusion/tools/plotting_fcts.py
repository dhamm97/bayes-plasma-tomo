import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import numpy as np
import os
import skimage.transform as skimt

import src.tomo_fusion.tools.helpers as tomo_helps


dirname = os.path.dirname(__file__)


def plot_sapg_run(lambda_min, lambda_max, lambda_list):
    plt.figure(figsize=(3,3))
    plt.plot(np.log10(lambda_list[0, :]))
    plt.axhline(y=np.log10(lambda_min), color='k', linestyle='--')
    plt.axhline(y=np.log10(lambda_max), color='k', linestyle='--')
    plt.xlabel("Iterations")
    plt.ylabel(r"$\log_{10}(\lambda)$")
    plt.title(r"SAPG: tuning $\lambda$")
    plt.show()


def plot_tomo_data(tomo_data, noisy_tomo_data):
    plt.figure(figsize=(3,3))
    plt.plot(tomo_data.flatten(), '.', label="true")
    plt.plot(noisy_tomo_data.flatten(), 'r.', label="noisy")
    plt.xlabel("LoS")
    plt.ylabel("I")
    plt.title("Tomographic data")
    plt.legend()
    plt.show()


def plot_errbar(xp, m, var, left=1, right=1, col="red", linewidth=2):
    plt.plot([xp-left, xp+right], [m+var, m+var], color=col, linewidth=linewidth)
    plt.plot([xp-left, xp+right], [m-var, m-var], color=col, linewidth=linewidth)
    plt.plot([xp, xp], [m+var, m-var], color=col, linewidth=linewidth)
    #plt.plot(xp, m, '.', color=col, markersize=10)


def plot_errbar_ax(ax, xp, m, var, left=1, right=1, col="red", linewidth=2):
    ax.plot([xp-left, xp+right], [m+var, m+var], color=col, linewidth=linewidth)
    ax.plot([xp-left, xp+right], [m-var, m-var], color=col)
    ax.plot([xp, xp], [m+var, m-var], color=col)
    ax.plot(xp, m, '.', color=col, markersize=10)


def plot_hyperparam_tuning_data(hyperparam_tuning_data, param="reg_param", true_param_val=None, plot_ssim=False):
    for key in list(hyperparam_tuning_data.keys()):
        if key == "GT":
            suptitle_key = "Ground truth"
        elif "CV_single" in key:
            suptitle_key = "Cross Validation (1 fold)"
        elif "CV_full" in key:
            suptitle_key = "Cross Validation (5 fold)"
        else:
            raise ValueError(f"Unknown key {key}")
        data = hyperparam_tuning_data[key]
        params = data[0, :]
        xticks = np.arange(0, int(np.log10(params[-1]) - np.log10(params[0]) + 1))
        params_plot_ticks = np.logspace(int(np.log10(params[0])), int(np.log10(params[-1])),
                                            int(np.log10(params[-1]) - np.log10(params[0]) + 1))
        xtick_labels = [r"$10^{{{}}}$".format(int(np.log10(x))) for x in params_plot_ticks]
        xlabel = r"$\lambda$" if param == "reg_param" else r"$\alpha$"
        nb_subplots = 3 if plot_ssim else 2
        fig_width = 18 if plot_ssim else 12
        fig, ax = plt.subplots(1, nb_subplots, figsize=(fig_width, 3))
        eval_points = np.linspace(xticks[0], xticks[-1], params.size)
        for i in range(nb_subplots):
            ax[i].plot(eval_points, data[i+1, :])
            ax[i].set_xlabel(xlabel)
            ax[i].set_xticks(xticks)
            ax[i].set_xticklabels(xtick_labels)
            ax[i].set_yscale("log")
            if true_param_val is not None:
                coeff = (np.log10(true_param_val)-np.log10(params[0]))/(np.log10(params[-1])-np.log10(params[0]))
                true_param_loc = eval_points[0] + coeff * (eval_points[-1] - eval_points[0])
                ax[i].vlines(true_param_loc, np.min(data[i+1, :]), np.max(data[i+1, :]), 'k', linestyle='--')
        ax[0].set_title("MSE tomo data")
        ax[1].set_title("MSE image")
        if plot_ssim:
            ax[2].set_title("SSIM")
        plt.suptitle(suptitle_key, fontsize=20, y=1.1)
        plt.show()


def plot_profile(image,
                 tcv_plot_clip=False,
                 contour_image=None, levels=15,
                 ax=None, colorbar=False,
                 interpolation=None, vmin=None, vmax=None, cmap="viridis", contour_color="w", aspect=None,
                 peak_stats=None,
                 pad_cbar=0, cbar_tick_params=None):
    if tcv_plot_clip:
        # define TCV patch for plotting
        tcv_shape_coords = np.load(dirname + "/../forward_model/tcv_shape_coords.npy")
        Lr, Lz = 0.511 if peak_stats is None else 0.5, 1.5
        h = Lz / image.shape[0]
        zs = np.linspace(0, Lz, round(Lz / h), endpoint=False) + 0.5 * h
        rs = np.linspace(0, Lr, round(Lr / h), endpoint=False) + 0.5 * h
        tcv_shape_coords[:, 0] = tcv_shape_coords[:, 0] * rs.size - 0.5 * h
        tcv_shape_coords[:, 1] = tcv_shape_coords[:, 1] * zs.size - 0.5 * h
        path = Path(tcv_shape_coords.tolist())
        patch = PathPatch(path, facecolor='none')

    handle = plt if ax is None else ax
    if peak_stats is not None:
        handle.plot(peak_stats["true_loc"][1], peak_stats["true_loc"][0], 'r.', markersize=2*peak_stats["markersize"])
        handle.plot(peak_stats["mean"][1], peak_stats["mean"][0], 'k.', markersize=peak_stats["markersize"])
        lower_bound_hor, upper_bound_hor = (
        peak_stats["mean"][1] - peak_stats["nb_stds"] * peak_stats["std"][1],
        peak_stats["mean"][1] + peak_stats["nb_stds"] * peak_stats["std"][1])
        lower_bound_vert, upper_bound_vert = (
        peak_stats["mean"][0] - peak_stats["nb_stds"] * peak_stats["std"][0],
        peak_stats["mean"][0] + peak_stats["nb_stds"] * peak_stats["std"][0])
        print(peak_stats["mean"])
        print(lower_bound_hor,upper_bound_hor,lower_bound_vert,upper_bound_vert  )
        handle.plot(np.array([lower_bound_hor, lower_bound_hor]), np.array([lower_bound_vert, upper_bound_vert]), 'k', linewidth=peak_stats["linewidth"])#, dashes=(1.5, 1.5))
        handle.plot(np.array([upper_bound_hor, upper_bound_hor]), np.array([lower_bound_vert, upper_bound_vert]), 'k', linewidth=peak_stats["linewidth"])#, dashes=(1.5, 1.5))
        handle.plot(np.array([lower_bound_hor, upper_bound_hor]), np.array([lower_bound_vert, lower_bound_vert]), 'k', linewidth=peak_stats["linewidth"])#, dashes=(1.5, 1.5))
        handle.plot(np.array([lower_bound_hor, upper_bound_hor]), np.array([upper_bound_vert, upper_bound_vert]), 'k', linewidth=peak_stats["linewidth"])#, dashes=(1.5, 1.5))


    # plot clipping to tcv shape
    if ax is None:
        plt.figure(figsize=(2, 3))
        if contour_image is not None:
            c = plt.contour(contour_image, origin="lower", levels=levels, antialiased=True, colors=contour_color,
                        linewidths=0.2)
            lcms_level = np.where(c.levels == 0)[0][0]
            c.collections[lcms_level].set_linewidth(0.75)
        p = plt.imshow(image, interpolation=interpolation, vmin=vmin, vmax=vmax, cmap=cmap, aspect=aspect)
    else:
        # plot on given figure axis
        if contour_image is not None:
            c = ax.contour(contour_image, origin="lower", levels=levels, antialiased=True, colors=contour_color,
                       linewidths=0.2)
            lcms_level = np.where(c.levels == 0)[0][0]
            c.collections[lcms_level].set_linewidth(0.75)
        p = ax.imshow(image, interpolation=interpolation, vmin=vmin, vmax=vmax, cmap=cmap, aspect=aspect)

    if tcv_plot_clip:
        hax = plt.gca() if ax is None else ax
        hax.add_patch(patch)
        p.set_clip_path(patch)
        if contour_image is not None:
           c.set_clip_path(patch)

    handle.axis('off')
    if colorbar:
        if ax is None:
            cbar = plt.colorbar()
        else:
            cbar = plt.colorbar(p, ax=ax, pad=pad_cbar)
        if cbar_tick_params is not None:
            cbar.ax.tick_params(labelsize=cbar_tick_params["labelsize"])
            cbar.ax.set_yticks(cbar_tick_params["yticks"])
            cbar.ax.set_yticklabels(cbar_tick_params["yticklabels"])

    return


def plot_phantom_and_sxr_diag(ground_truth, psi, tomo_data, noisy_tomo_data, levels=12, tcv_plot_clip=False, save_dir=None):
    fig, ax = plt.subplots(1, 3, figsize=(8, 3), width_ratios=[1, 1, 1.5])

    # plot ground truth
    plot_profile(ground_truth, tcv_plot_clip=tcv_plot_clip, contour_image=psi, levels=levels,
                   ax=ax[0], colorbar=True, contour_color="w")
    ax[0].set_title("Phantom")

    # plot ground truth, superposing the sxr LoS
    plot_profile(np.flip(ground_truth, axis=0), levels=levels, tcv_plot_clip=tcv_plot_clip, ax=ax[1])
    sxr_LoS_params = np.load(dirname + "/../forward_model/sxr_LoS_params.npy")
    r = np.linspace(0, ground_truth.shape[0], 10)
    center = np.array(ground_truth.shape, dtype=int) / 2
    Lr, Lz = 0.511, 1.5
    h = Lz / ground_truth.shape[0]
    for i in range(sxr_LoS_params.shape[0]):
        if np.isclose(np.tan(sxr_LoS_params[i, 1]), 0, atol=1e-6):
            ax[i].vlines(center[1] + sxr_LoS_params[i, 0], 0, Lz, "r", linewidth=0.3)
        else:
            p = sxr_LoS_params[i, 0]
            theta = sxr_LoS_params[i, 1]
            y = center[0] + p / h / np.sin(theta) - (r - center[1]) / np.tan(theta)
            ax[1].plot(r, y, "r", linewidth=0.3)
    ax[1].set_xlim([0, ground_truth.shape[1]])
    ax[1].set_ylim([0, round(Lz / h)])
    ax[1].set_title("SXR diagnostic")

    # plot tomographic data
    ax[2].plot(np.arange(1, 101), tomo_data, '.', label="true")
    ax[2].plot(np.arange(1, 101), noisy_tomo_data, 'r.', label="noisy")
    ax[2].set_xlabel("LoS index", fontsize=12)
    ax[2].set_xlim([-2, 103])
    ax[2].set_xticks([1, 50, 100])
    #ax[2].set_ylabel(r"$y$", rotation=0, fontsize=12, labelpad=10)
    ax[2].set_title("Tomographic data")
    ax[2].legend()

    if save_dir is not None:
        # save plot
        plt.savefig(save_dir + "/phantom_and_sxr_diag.eps")

    plt.show()


def plot_uq_data(uq_data, ground_truth, psi, levels=12,
                 plot_ground_truth=True, plot_MAP=False, plot_mean=False, plot_std=False,
                 plot_nb_stds=False, plot_quantiles=False,
                 plot_prad=False,
                 quantiles_idx=2,
                 mask_core=None,
                 cmaps=None,
                 vmin_quantile=0, vmax_std=1, vmax_nb_std=1, vmax_adjust_if_quantile=True,
                 tcv_plot_clip=False, save_dir=None):

    # Reshape ground truth and magnetic equilibrium if necessary
    if ground_truth.shape != uq_data["im_MAP"].shape:
        ground_truth = skimt.resize(ground_truth, uq_data["im_MAP"].shape, anti_aliasing=False, mode='edge')
    if psi.shape != uq_data["im_MAP"].shape:
        psi = skimt.resize(psi, uq_data["im_MAP"].shape, anti_aliasing=False, mode='edge')

    # determine number of subplots
    nb_subplots = np.sum(plot_ground_truth + plot_MAP + plot_mean + plot_std + plot_nb_stds + 2*plot_quantiles + plot_prad)
    if not plot_prad:
        fig_width = 2*nb_subplots
        width_ratios = [1]*nb_subplots
    else:
        fig_width = 2*nb_subplots + 1
        width_ratios = [1]*nb_subplots
        width_ratios[-1] = 1.5

    # define cmaps
    if cmaps is None:
        cmaps = ["viridis"]*nb_subplots
    else:
        assert len(cmaps) == nb_subplots - plot_prad, "`len(cmaps)` incompatible with `nb_subplots`"

    # create figure
    fig, ax = plt.subplots(1, nb_subplots, figsize=(fig_width, 3), width_ratios=width_ratios)
    ax_counter = 0
    if nb_subplots == 1:
        ax = [ax]

    # define max value for plots
    vmax = np.max(uq_data["empirical_quantiles"][-quantiles_idx, :, :]) if (plot_quantiles and vmax_adjust_if_quantile) else 1

    if plot_ground_truth:
        # plot ground truth
        plot_profile(ground_truth, tcv_plot_clip=tcv_plot_clip, contour_image=psi, levels=levels,
                     ax=ax[ax_counter], colorbar=True, contour_color="w", vmax=vmax, cmap=cmaps[ax_counter])
        if not plot_nb_stds and not plot_prad:
            ax[ax_counter].set_title("Phantom")
        else:
            ax[ax_counter].set_title(r"Phantom $x$")
        ax_counter += 1
    if plot_MAP:
        # plot MAP
        plot_profile(uq_data["im_MAP"], tcv_plot_clip=tcv_plot_clip, contour_image=psi, levels=levels,
                     ax=ax[ax_counter], colorbar=True, contour_color="w", vmax=vmax, cmap=cmaps[ax_counter])
        ax[ax_counter].set_title(r"$x_{MAP}$")
        ax_counter += 1
    if plot_mean:
        # plot ULA mean
        plot_profile(uq_data["mean"], tcv_plot_clip=tcv_plot_clip, contour_image=psi, levels=levels,
                     ax=ax[ax_counter], colorbar=True, contour_color="w", vmax=vmax, cmap=cmaps[ax_counter])
        ax[ax_counter].set_title(r"$\mu_{ULA}$")
        ax_counter += 1
    if plot_std:
        # plot ULA standard variation
        plot_profile(np.sqrt(uq_data["var"]), tcv_plot_clip=tcv_plot_clip, contour_image=psi, levels=levels,
                     ax=ax[ax_counter], colorbar=True, contour_color="w", vmin=0, vmax=vmax_std, cmap=cmaps[ax_counter])
        ax[ax_counter].set_title(r"$\sigma_{ULA}$")
        ax_counter += 1
    if plot_nb_stds:
        # plot distance from mean in terms of number of standard deviations
        tbp = np.abs(uq_data["mean"] - ground_truth)
        tbp /= np.sqrt(uq_data["var"])
        plot_profile(tbp, tcv_plot_clip=tcv_plot_clip, contour_image=psi, levels=levels,
                     ax=ax[ax_counter], colorbar=True, contour_color="w", vmax=vmax_nb_std, cmap=cmaps[ax_counter])
        ax[ax_counter].set_title(r'$\vert \mu_{ULA} - x \vert\,/\,\sigma_{ULA}$')
        ax_counter += 1
    if plot_quantiles:
        # plot quantiles
        plot_profile(uq_data["empirical_quantiles"][quantiles_idx-1, :, :], tcv_plot_clip=tcv_plot_clip, contour_image=psi, levels=levels,
                     ax=ax[ax_counter], colorbar=True, contour_color="w", vmin=vmin_quantile, vmax=vmax, cmap=cmaps[ax_counter])
        ax[ax_counter].set_title('{} quantile'.format(uq_data["quantile_marks"][quantiles_idx-1]))
        ax_counter += 1
        plot_profile(uq_data["empirical_quantiles"][-quantiles_idx, :, :], tcv_plot_clip=tcv_plot_clip, contour_image=psi, levels=levels,
                     ax=ax[ax_counter], colorbar=True, contour_color="w", vmax=vmax, cmap=cmaps[ax_counter])
        ax[ax_counter].set_title('{} quantile'.format(uq_data["quantile_marks"][-quantiles_idx]))
        ax_counter += 1
    if plot_prad:
        # plot radiated power
        ax[ax_counter].hist(uq_data["prads_core"], bins=int(5e2), density=True, color="b")
        ax[ax_counter].set_title(r'$P_{rad}^{core}$')
        min_prad = np.min(np.array([np.min(uq_data["prads_core"]), np.min(uq_data["prad_map_core"])]))
        max_prad = np.max(np.array([np.max(uq_data["prads_core"]), np.max(uq_data["prad_map_core"])]))
        xrange_prad = [0.95*min_prad, 1.05*max_prad]
        ax[ax_counter].set_xlim(xrange_prad)
        if mask_core is not None:
            true_prad_core = tomo_helps.compute_radiated_power(ground_truth, mask_core, uq_data["sampling"])
            ax[ax_counter].axvline(x=true_prad_core, ymin=0, ymax=30, color='g', label=r"$P_{rad}^{core}(x)$")
        ax[ax_counter].axvline(x=uq_data["prad_map_core"], ymin=0, ymax=30, color='r', label=r"$P_{rad}^{core}(x_{MAP})$")
        ax[ax_counter].legend(loc="upper right")
        ax[ax_counter].set_yticks([])

    if save_dir is not None:
        # save plot
        plt.savefig(save_dir + "/phantom_and_sxr_diag.eps")

    plt.show()


def plot_ptheta_LoS_tcv(params, markersize=5):
    """
    This function plots, for a given (p, theta) configuration, the LoS configuration
    taking into account the TCV geometry. The shaded areas correspond to lines that fall
    outside the vessel.
    """
    tcv_mask_finesse = round(300)
    pmin = np.min(params[:, 0])
    prange = np.max(params[:, 0]) - np.min(params[:, 0])
    # pmax corresponding to tcv geometry
    tcv_pmax = 0.5646 + 0.25
    PT_intersecting_tcv_mask = np.load(dirname+'/../forward_model/tcv_mask_sinogram.npy')
    plt.figure(figsize=(6,3))
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Helvetica"
    })
    # plot tcv mask
    plt.imshow(PT_intersecting_tcv_mask, origin="upper", aspect="auto", cmap="gray", alpha=0.5)
    # plot considered LoS configuration
    scaling_factor_p = prange / (2*tcv_pmax)  # 2*pmax/(...)
    scaling_factor_theta = np.max(params[:, 1]) / np.pi
    ps = (params[:, 0]+np.abs(np.min(params[:, 0])))/np.max(params[:, 0]+np.abs(np.min(params[:, 0])))*(tcv_mask_finesse-1)*scaling_factor_p
    # shift ps
    #ps += eps / (2*pmax + 2*eps) * (tcv_mask_finesse-1)
    ps += (tcv_pmax + pmin) / (2 * tcv_pmax) * (tcv_mask_finesse-1)
    thetas = params[:, 1]/np.max(params[:, 1])*(tcv_mask_finesse-1)*scaling_factor_theta
    for i in range(params.shape[0]):
        #a=0
        plt.plot(thetas[i], ps[i], "r", marker=".", markersize=markersize)
    plt.xticks([0, tcv_mask_finesse/2, tcv_mask_finesse-1], [r'$0$', r'$\pi/2$', r'$\pi$'], fontsize=20)
    plt.yticks([0, tcv_mask_finesse/2, tcv_mask_finesse-1], [r'$-p_{max}$', r'$0$', r'$p_{max}$'], fontsize=20)
    #plt.yticks([])
    plt.xlabel(r"$\theta$", fontsize=25)
    plt.ylabel(r"$p$", rotation=0, fontsize=25, labelpad=10)
    plt.title("Subsampled sinogram", fontsize=25, color="k")

