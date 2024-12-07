import matplotlib.pyplot as plt
import numpy as np


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
    plt.plot([xp-left, xp+right], [m-var, m-var], color=col)
    plt.plot([xp, xp], [m+var, m-var], color=col)
    plt.plot(xp, m, '.', color=col, markersize=10)


def plot_errbar_ax(ax, xp, m, var, left=1, right=1, col="red", linewidth=2):
    ax.plot([xp-left, xp+right], [m+var, m+var], color=col, linewidth=linewidth)
    ax.plot([xp-left, xp+right], [m-var, m-var], color=col)
    ax.plot([xp, xp], [m+var, m-var], color=col)
    ax.plot(xp, m, '.', color=col, markersize=10)


# def plot_reg_param_tuning_data(reg_param_tuning_data, plot_ssim=False):
#     for key in list(reg_param_tuning_data.keys()):
#         if key == "GT":
#             suptitle_key = "Ground truth"
#         elif key == "CV_single":
#             suptitle_key = "Cross Validation (1 fold)"
#         elif key == "CV_full":
#             suptitle_key = "Cross Validation (5 fold)"
#         else:
#             raise ValueError(f"Unknown key {key}")
#         data = reg_param_tuning_data[key]
#         reg_params = data[0, :]
#         xticks = np.arange(0, int(np.log10(reg_params[-1]) - np.log10(reg_params[0]) + 1))
#         reg_params_plot_ticks = np.logspace(int(np.log10(reg_params[0])), int(np.log10(reg_params[-1])),
#                                             int(np.log10(reg_params[-1]) - np.log10(reg_params[0]) + 1))
#         xtick_labels = [r"$10^{{{}}}$".format(int(np.log10(x))) for x in reg_params_plot_ticks]
#         nb_subplots = 3 if plot_ssim else 2
#         fig_width = 18 if plot_ssim else 12
#         fig, ax = plt.subplots(1, nb_subplots, figsize=(fig_width, 3))
#         eval_points = np.linspace(xticks[0], xticks[-1], reg_params.size)
#         for i in range(nb_subplots):
#             ax[i].plot(eval_points, data[i+1, :])
#             ax[i].set_xlabel(r"$\lambda$")
#             ax[i].set_xticks(xticks)
#             ax[i].set_xticklabels(xtick_labels)
#             ax[i].set_yscale("log")
#         ax[0].set_title("MSE tomo data")
#         ax[1].set_title("MSE image")
#         if plot_ssim:
#             ax[2].set_title("SSIM")
#         plt.suptitle(suptitle_key, fontsize=20, y=1.1)
#         plt.show()


def plot_hyperparam_tuning_data(hyperparam_tuning_data, param="reg_param", true_param_val=None, plot_ssim=False):
    for key in list(hyperparam_tuning_data.keys()):
        if key == "GT":
            suptitle_key = "Ground truth"
        elif key == "CV_single":
            suptitle_key = "Cross Validation (1 fold)"
        elif key == "CV_full":
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



# def plot_statistics_full(data, gt, im_MAP, prad_tcv_true, prad_core_true, psi, s_id, xrange_prad=[1.2, 1.6], seed=0):
#     tomo_data, noisy_tomo_data = compute_tomo_data(gt, seed)
#     # plt.figure(figsize=(3,3))
#     # plt.plot(tomo_data.T, '.', label="true")
#     # plt.plot(noisy_tomo_data.T, 'r.', label="noisy")
#     # plt.xlabel("LoS")
#     # plt.ylabel("I")
#     # plt.title("Tomographic data")
#     # plt.legend()
#     # plt.show()
#     fig, ax = plt.subplots(1, 4, figsize=(10,3), width_ratios=[1, 0.2, 1, 1.25])
#     #plt.subplots_adjust(wspace=0.3)
#     vmin=0
#     vmax=np.max(gt)
#     # im=ax[0].imshow(gt, vmin=vmin, vmax=vmax)
#     # ax[0].contour(psi, origin="lower", levels=10, antialiased=True, colors="r", linewidths=0.2)
#     # plt.colorbar(im, ax=ax[0])
#     # ax[0].axis('off')
#     # ax[0].set_title("Ground truth")
#     # ax[1].plot(tomo_data.T, '.', label="true")
#     # ax[1].plot(noisy_tomo_data.T, 'r.', label="noisy")
#     # ax[1].set_xlabel("LoS",  fontsize=12)
#     # ax[1].set_ylabel("I", rotation=0, fontsize=12, labelpad=10)
#     # ax[1].set_title("Tomographic data")
#     # ax[1].legend()
#     im=ax[0].imshow(gt, vmin=vmin, vmax=vmax)
#     ax[0].contour(psi, origin="lower", levels=10, antialiased=True, colors="r", linewidths=0.2)
#     plt.colorbar(im, ax=ax[0])
#     ax[0].axis('off')
#     ax[0].set_title("Ground truth")
#     ax[1].axis('off')
#
#     import tomo_fusion.forward_model.chord_geometry as geom
#     import copy
#     from matplotlib.path import Path
#     from matplotlib.patches import PathPatch
#     r = np.linspace(0, 40, 10)
#     radcam = geom.RADCAM_system()
#     LoS_params = radcam.sxr_LoS_params
#     center = [20, 60]
#     linewidth=0.5
#     Lr=0.5
#     Lz=1.5
#     h=0.0125
#     zs = np.linspace(0, Lz, round(Lz/h), endpoint=False)+0.5*h
#     rs = np.linspace(0, Lr, round(Lr/h), endpoint=False)+0.5*h
#     extent = copy.deepcopy(radcam.tile_extent_plot)
#     extent[:,0] = extent[:,0] * rs.size - 0.5*h
#     extent[:,1] = extent[:,1] * zs.size - 0.5*h
#     path = Path(extent.tolist())
#     patch=PathPatch(path, facecolor='none')
#     #p = plt.imshow(gt, clip_path=patch, clip_on=True)
#     ax[2].imshow(np.flip(gt, axis=0), origin="lower")
#     #plt.colorbar()
#     #plt.imshow(field_aligned_phantom)
#     #plt.gca().add_patch(patch)
#     #p.set_clip_path(patch)
#     ax[2].axis('off')
#     for i in range(LoS_params.shape[0]):
#         if np.isclose(np.tan(LoS_params[i, 1]), 0, atol=1e-6):
#             #plt.vlines(center[0] + LoS_params[i, 0], 0, Lz, "r")
#             if center[0] + LoS_params[i, 0] < 0 or center[0] + LoS_params[i, 0] > 0.6:
#                 print("Vertical line does not intersect domain at x={}!".format(center[0] + LoS_params[i, 0]))
#         else:
#             p = LoS_params[i, 0]
#             theta = LoS_params[i, 1]
#             y = center[1] + p/h / np.sin(theta) - (r - center[0]) / np.tan(theta)
#             # if np.min(y)>1.6 or np.max(y)<0:
#             #    print("Line does not intersect domain!")
#             ax[2].plot(r, y, "r", linewidth=linewidth)
#     #ax[2].imshow(gt, origin="upper")
#     ax[2].set_xlim([0, 40])
#     ax[2].set_ylim([0, 120])
#     ax[2].set_title("SXR diagnostic")
#
#     ax[3].plot(tomo_data.T, '.', label="true")
#     ax[3].plot(noisy_tomo_data.T, 'r.', label="noisy")
#     ax[3].set_xlabel("LoS",  fontsize=12)
#     ax[3].set_ylabel("I", rotation=0, fontsize=12, labelpad=10)
#     ax[3].set_title("Tomographic data")
#     ax[3].legend()
#     plt.suptitle("Sample {}\n".format(s_id), fontsize=20, y=1.1)
#     plt.show()
#
#
#     fig, ax = plt.subplots(1, 7, figsize=(14,3), width_ratios=[1, 1, 1, 0.5, 2, 1, 1])
#     vmin=0
#     vmax=np.max(gt)
#     im=ax[0].imshow(data["mean"].reshape(arg_shape), vmin=vmin, vmax=vmax)
#     ax[0].contour(psi, origin="lower", levels=10, antialiased=True, colors="r", linewidths=0.2)
#     plt.colorbar(im, ax=ax[0])
#     ax[0].axis('off')
#     ax[0].set_title(r"$\mu_{ULA}$")
#     im=ax[1].imshow(np.sqrt(data["var"]).reshape(arg_shape))#, vmin=vmin, vmax=vmax)
#     ax[1].contour(psi, origin="lower", levels=10, antialiased=True, colors="r", linewidths=0.2)
#     plt.colorbar(im, ax=ax[1])
#     ax[1].axis('off')
#     ax[1].set_title(r"$\sigma_{ULA}$")
#     tbp = np.abs(data["mean"]-gt.flatten())
#     tbp /= np.sqrt(data["var"])
#     im=ax[2].imshow(tbp.reshape(arg_shape))#, vmin=vmin, vmax=vmax)
#     ax[2].contour(psi, origin="lower", levels=10, antialiased=True, colors="r", linewidths=0.2)
#     plt.colorbar(im, ax=ax[2])
#     ax[2].axis('off')
#     ax[2].set_title(r'$\frac{\vert \mu_{ULA} -ground\_truth \vert}{\sigma_{ULA}}$')
#     ax[3].axis('off')
#     im = ax[4].hist(data["prads_core"], bins=int(5e2), density=True, color="b")
#     ax[4].set_title(r'$P_{rad}^{core},\;ULA\;$ empirical')
#     ax[4].set_xlim(xrange_prad)
#     ax[4].axvline(x=prad_core_true, ymin=0, ymax=30, color='g', label=r"true $P_{rad}^{core}$")
#     ax[4].legend(loc="upper left")
#     ax[4].set_yticks([])
#
#     temp_low=np.zeros(N)
#     temp_low[mask_core.flatten()] = data["empirical_quantiles"][0, :]
#     im=ax[5].imshow(temp_low.reshape(arg_shape), vmax=vmax)#, vmin=vmin, vmax=vmax)
#     ax[5].contour(psi, origin="lower", levels=10, antialiased=True, colors="r", linewidths=0.2)
#     plt.colorbar(im, ax=ax[5])
#     ax[5].axis('off')
#     ax[5].set_title("0.05 quantile")
#     temp_high=np.zeros(N)
#     temp_high[mask_core.flatten()] = data["empirical_quantiles"][4, :]
#     im=ax[6].imshow(temp_high.reshape(arg_shape), vmax=vmax)#, vmin=vmin, vmax=vmax)
#     ax[6].contour(psi, origin="lower", levels=10, antialiased=True, colors="r", linewidths=0.2)
#     plt.colorbar(im, ax=ax[6])
#     ax[6].axis('off')
#     ax[6].set_title("0.95 quantile")
#     plt.show()
#
#     #print("Fraction of pixels falling withing 90%-credible interval:  {}".format(frac_pix))
#
#     fig, ax = plt.subplots(1, 7, figsize=(14,3), width_ratios=[1, 1, 1, 0.5, 2, 1, 1])
#     im=ax[0].imshow(im_MAP.reshape(arg_shape), vmin=vmin, vmax=vmax)
#     ax[0].contour(psi, origin="lower", levels=10, antialiased=True, colors="r", linewidths=0.2)
#     plt.colorbar(im, ax=ax[0])
#     ax[0].axis('off')
#     ax[0].set_title(r"$MAP$")
#     im=ax[1].imshow(np.sqrt(data["var_wrtMAP"]).reshape(arg_shape))#, vmin=vmin, vmax=vmax)
#     ax[1].contour(psi, origin="lower", levels=10, antialiased=True, colors="r", linewidths=0.2)
#     plt.colorbar(im, ax=ax[1])
#     ax[1].axis('off')
#     ax[1].set_title(r"$\sigma_{MAP}$")
#     tbp = np.abs(im_MAP-gt.flatten())
#     tbp /= np.sqrt(data["var_wrtMAP"])
#     im=ax[2].imshow(tbp.reshape(arg_shape))#, vmin=vmin, vmax=vmax)
#     ax[2].contour(psi, origin="lower", levels=10, antialiased=True, colors="r", linewidths=0.2)
#     plt.colorbar(im, ax=ax[2])
#     ax[2].axis('off')
#     ax[2].set_title(r'$\frac{\vert MAP -ground\_truth \vert}{\sigma_{MAP}}$')
#     ax[3].axis('off')
#
#     #im = ax[4].hist(data["prads_core"], bins=int(5e2), density=True, color="b")
#     ax[4].set_title(r"$P_{rad}^{MAP}$ estimates")
#     ax[4].set_xlim(xrange_prad)
#     ax[4].axvline(x=prad_core_true, ymin=0, ymax=30, color='g', label=r"true $P_{rad}^{core}$")
#     ax[4].axvline(x=data["prad_map_core"], ymin=0, ymax=30, color='r', label=r"$P_{rad}^{MAP}$")
#     ax[4].legend(loc="upper left")
#     ax[4].set_yticks([])
#     ax[5].axis('off')
#     ax[6].axis('off')
#     # plot_errbar_ax(ax[4], 1, data["prad_map_core"], np.sqrt(data["var_prad_wrtMAP_core"]), left=0.1, right=0.1, col="b", linewidth=2)
#     # plt.plot(1, prad_core_true, 'g.', markersize=10, label=r"true $P_{rad}^{core}$")
#     # ax[4].set_xlim([0,2])
#     # ax[4].set_xticks([1])
#     # ax[4].set_xticklabels([r"$P_{rad}^{core}$"])
#     # ax[4].legend(loc="upper left")
#     # ax[4].set_title(r"$P_{rad}^{MAP}$ estimates")
#     plt.show()

