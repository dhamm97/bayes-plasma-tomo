import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path
import numpy as np
import src.tomo_fusion.forward_model.chord_geometry as geom


def shift_tcv_coords(x_points, y_points):
    xs = x_points - 0.624
    ys = y_points + 0.75
    return xs, ys


def generate_LoS_from_point_couples(x_points, y_points, center=[0.257, 0.75], tcv_coords=True):
    """
    Generates (p,theta)-parametrization for LoS passing through the given points.
    Follows p\in(-\infty,\infty), theta\in(0,pi) convention.
    """
    if tcv_coords:
        x_points, y_points = shift_tcv_coords(x_points, y_points)
    # initializations
    LoS_params = np.zeros((x_points.shape[0], 2))
    startpoints = np.zeros(LoS_params.shape)
    endpoints = np.zeros(LoS_params.shape)
    # cycle over LoS
    for i in range(x_points.shape[0]):
        start_point = np.array([x_points[i, 0], y_points[i, 0]])
        end_point = np.array([x_points[i, 1], y_points[i, 1]])
        # compute (p, theta) parametrization for i-th LoS
        LoS_params[i, :] = LineParametrizationFromTwoPoints(start_point, end_point, center)
        startpoints[i, :] = start_point
        endpoints[i, :] = end_point
    # startpoints, endpoints = cartesian_to_image_coordinates(startpoints, endpoints, Lz, h)
    return LoS_params, startpoints, endpoints


def LineParametrizationFromTwoPoints(P1, P2, center):
    """
    Compute (p,theta)-parametrization for line through P1,P2.
    Follows p\in(-\infty,\infty), theta\in(0,pi) convention
    """
    if np.isclose(P1[0], P2[0], atol=1e-6):
        # vertical line
        p = np.abs((P1[0] - P2[0]) * (P1[1] - center[1]) - (P1[1] - P2[1]) * (P1[0] - center[0])) / np.linalg.norm(
            P2 - P1)
        p *= np.sign(P1[0] - center[0])
        theta = 0
    else:
        # non-vertical line
        steepness = (P2[1] - P1[1]) / (P2[0] - P1[0])
        # intercept = P1[1]-steepness*P1[0]
        # intercept computed w.r.t. parametrization around (0,0)
        intercept = P1[1] - steepness * P1[0]
        # compute center-LoS distance
        p = np.abs((P1[0] - P2[0]) * (P1[1] - center[1]) - (P1[1] - P2[1]) * (P1[0] - center[0])) / np.linalg.norm(P2 - P1)
        # compute angle between horizontal and direction orthogonal to LoS
        theta = np.arctan(steepness)
        # adjust p and theta for consistency with the adopted convention
        p = np.sign((center[0] * steepness + intercept) - center[1]) * p
        # theta=(theta<0)*(np.pi-theta)+(theta>=0)*(theta)
        theta = theta + np.pi / 2
    return p, theta


def LineCellIntersections(p, theta, pixel_origin=np.zeros(2), hx=1.0, hy=1.0):
    # compute boundaries-line intersection points
    tol = 1e-6
    if np.isclose(np.cos(theta), 0, atol=tol):
        # horizontal line
        z_inters = p + pixel_origin[0]
        #z_inters = p / np.sin(theta) - pixel_origin[0] / np.tan(theta)
        if (pixel_origin[1]) < z_inters < (pixel_origin[1] + hy):
            P1 = np.array([pixel_origin[0], z_inters])
            P2 = P1 + np.array([hx, 0])
        else:
            P1, P2 = None, None
    elif np.isclose(np.sin(theta), 0, atol=tol):
        # vertical line
        r_inters = p / np.cos(theta) - pixel_origin[1] * np.tan(theta)
        if (pixel_origin[0]) < r_inters < (pixel_origin[0] + hx):
            P1 = np.array([r_inters, pixel_origin[1]])
            P2 = P1 + np.array([0, hy])
        else:
            P1, P2 = None, None
    else:
        inters = np.array(
            [
                p / np.cos(theta) - pixel_origin[1] * np.tan(theta),
                p / np.sin(theta) - pixel_origin[0] / np.tan(theta),
                p / np.cos(theta) - (pixel_origin[1] + hy) * np.tan(theta),
                p / np.sin(theta) - (pixel_origin[0] + hx) / np.tan(theta),
            ]
        )
        points = []
        if (pixel_origin[0] - tol) <= inters[0] <= (pixel_origin[0] + hx + tol):
            points.append(np.array([inters[0], pixel_origin[1]]))
        if (pixel_origin[1] - tol) <= inters[1] <= (pixel_origin[1] + hy + tol):
            points.append(np.array([pixel_origin[0], inters[1]]))
        if (pixel_origin[0] - tol) <= inters[2] <= (pixel_origin[0] + hx + tol):
            points.append(np.array([inters[2], pixel_origin[1] + hy]))
        if (pixel_origin[1] - tol) <= inters[3] <= (pixel_origin[1] + hy + tol):
            points.append(np.array([pixel_origin[0] + hx, inters[3]]))
        if len(points) > 2:
            # if points through corner, select only one of detected intersections
            norm_diffs = np.linalg.norm(np.diff(points, axis=0), axis=1)
            close_points = np.isclose(norm_diffs, np.zeros(len(norm_diffs)), atol=1e-3)
            index = int(np.where(~close_points)[0][0])
            points = points[index : index + 2]
        # extracts coordinates of points
        if len(points) == 2:
            P1 = points[0]
            P2 = points[1]
        else:
            P1, P2 = None, None

    return P1, P2


def plot_LoS(LoS_params, Lr=0.5, Lz=1.5, center=[0.3, 0.8]):
    r = np.linspace(0, Lr, 10)
    plt.figure(figsize=(2.5, 6))
    for i in range(LoS_params.shape[0]):
        if np.isclose(np.tan(LoS_params[i, 1]), 0, atol=1e-6):
            plt.vlines(center[0] + LoS_params[i, 0], 0, Lz, "r")
            if center[0] + LoS_params[i, 0] < 0 or center[0] + LoS_params[i, 0] > 0.6:
                print("Vertical line does not intersect domain at x={}!".format(center[0] + LoS_params[i, 0]))
        else:
            p = LoS_params[i, 0]
            theta = LoS_params[i, 1]
            y = center[1] + p / np.sin(theta) - (r - center[0]) / np.tan(theta)
            # if np.min(y)>1.6 or np.max(y)<0:
            #    print("Line does not intersect domain!")
            plt.plot(r, y, "r")
    plt.xlim([0, Lr])
    plt.ylim([0, Lz])
    plt.title("Lines of Sight", fontsize=20)


def plot_LoS_ax(LoS_params, ax, Lr=0.5, Lz=1.5, center=[0.3, 0.8]):
    r = np.linspace(0, Lr, 10)
    for i in range(LoS_params.shape[0]):
        if np.isclose(np.tan(LoS_params[i, 1]), 0, atol=1e-6):
            ax.vlines(center[0] + LoS_params[i, 0], 0, Lz, "r")
        else:
            p = LoS_params[i, 0]
            theta = LoS_params[i, 1]
            y = center[1] + p / np.sin(theta) - (r - center[0]) / np.tan(theta)
            ax.plot(r, y, "r", alpha=0.2)
    ax.set_xlim([0, Lr])
    ax.set_ylim([0, Lz])


def LoS_extrema_from_LoS_params(LoS_params, Lr=0.5, Lz=1.5, h=np.array([0.1, 0.1]), convert_to_image_coordinates=False):
    """
    Function receives (p,theta) parametrization wrt centerpoint. It:
    1. converts parameters into (p,theta)-parametrization wrt lower-left corner (cartesian origin)
    2. uses this parametrization to compute intersections between LoS and boundaries of domain
    3. converts result in image coordinates, with origin at upper left corner
    Parameters
    ----------
    LoS_params
    Lr
    Lz

    Returns
    -------

    """
    nLoS = LoS_params.shape[0]
    LoS_params_shifted = np.zeros((nLoS, 2))
    arctan_steepness = np.zeros(nLoS)
    # indices_under90 = np.where(LoS_params[:, 1] <= np.pi / 2)
    mask_under90 = np.where(LoS_params[:, 1] <= np.pi / 2, True, False)
    mask_over90 = ~mask_under90
    mask_negative_ps = np.where(LoS_params[:, 1] > (np.pi / 2 + np.arctan(Lz / Lr)), True, False)
    arctan_steepness[mask_under90] = np.pi / 2 + LoS_params[mask_under90, 1]
    arctan_steepness[mask_over90] = LoS_params[mask_over90, 1] - np.pi / 2
    # abs(dist_no_abs) is distance between the lower-left corner (cartesian origin) and
    # the lines parallel to the LoS and passing through the centerpoint (Lr/2, Lz/2)
    dist_no_abs = np.cos(arctan_steepness) * (Lz / 2) - np.sin(arctan_steepness) * (Lr / 2)
    LoS_params_shifted[:, 0] = np.abs(dist_no_abs) + LoS_params[:, 0]
    # for LoS corresponding to negative p in (p,theta)-parametrization wrt origin, don't take np.abs() of dist_no_abs
    LoS_params_shifted[mask_negative_ps, 0] = dist_no_abs[mask_negative_ps] + LoS_params[mask_negative_ps, 0]
    LoS_params_shifted[:, 1] = LoS_params[:, 1]
    startpoints = []
    endpoints = []
    non_intersecting_indices = []
    for i in range(nLoS):
        P1, P2 = LineCellIntersections(
            LoS_params_shifted[i, 0], LoS_params_shifted[i, 1], pixel_origin=np.zeros(2), hx=Lr, hy=Lz
        )
        if P1 is not None and P2 is not None:
            # requested LoS might not intersect domain. In this case, no LoS is generated
            startpoints.append(P1)
            endpoints.append(P2)
        else:
            #print("{}th Line does not intersect domain!".format(i))
            non_intersecting_indices.append(i)
    startpoints = np.array(startpoints)
    endpoints = np.array(endpoints)
    if convert_to_image_coordinates:
        startpoints, endpoints = cartesian_to_image_coordinates(startpoints, endpoints, Lz=Lz, h=h)
    return startpoints, endpoints, non_intersecting_indices


def cartesian_to_image_coordinates(startpoints, endpoints, Lz=1.5, h=np.array([0.1, 0.1])):
    if h.size == 1:
        h = np.array([h, h])
    startpoints = np.flip(startpoints, axis=1)
    endpoints = np.flip(endpoints, axis=1)
    startpoints[:, 0] = Lz - startpoints[:, 0]
    endpoints[:, 0] = Lz - endpoints[:, 0]
    startpoints[:, 0] /= h[0]
    startpoints[:, 1] /= h[1]
    endpoints[:, 0] /= h[0]
    endpoints[:, 1] /= h[1]
    #startpoints /= h
    #endpoints /= h
    startpoints -= [0.5, 0.5]
    endpoints -= [0.5, 0.5]
    return startpoints, endpoints


def generate_LoS_uniformly_spaced(nP, nTheta, Pmin, Pmax, Lr, Lz, h=np.array([0.1, 0.1]), eps=0.01):
    LoS_params = np.zeros((nP * nTheta, 2))
    ps = np.linspace(Pmin + eps, Pmax - eps, nP)
    thetas = np.linspace(0, np.pi, nTheta, endpoint=False)
    thetas += 0.5 * (thetas[1] - thetas[0])
    Ps, Thetas = np.meshgrid(ps, thetas)
    LoS_params[:, 0] = Ps.reshape(-1)
    LoS_params[:, 1] = Thetas.reshape(-1)
    startpoints, endpoints, non_intersecting_indices = LoS_extrema_from_LoS_params(LoS_params, Lr=Lr, Lz=Lz, h=h, convert_to_image_coordinates=True)
    LoS_params = np.delete(LoS_params, non_intersecting_indices, axis=0)
    return LoS_params, startpoints, endpoints


def generate_LoS(nU, nR, nB, nL, Lr=0.6, Lz=1.6, h=np.array([0.1, 0.1]), eps=0.01):
    """

    Parameters
    ----------
    nU: number of LoS from upper port
    nR: number of LoS from right port
    nB: number of LoS from bottom port
    nL: number of LoS from left port
    Lr: radial length of domain
    Lz: height of domain
    h: spatial discretization parameter
    eps: closest possible distance between LoS and an edge

    Returns
    -------
    LoS_params: (nbLoS, 2) array with (p,theta) parametrization of LoS around image center
    startpoints: (nbLos, 2) array with startpoints of each LoS in pixel coordinates
    endpoints: (nbLos, 2) array with endpoints of each LoS in pixel coordinates
    """
    LoS_params = np.zeros((nU + nR + nB + nL, 2))
    # compute coordinates ( (r,z) with origin at lower left corner ) of LoS-boundaries intersections
    startpoints_upper = np.hstack((np.linspace(0.25 * Lr, 0.75 * Lr, nU).reshape(-1, 1), Lz * np.ones((nU, 1))))
    endpoints_upper = np.hstack((np.linspace(0 + eps, Lr - eps, nU).reshape(-1, 1), np.zeros((nU, 1))))
    startpoints_right = np.hstack((np.zeros((nR, 1)), np.linspace(0 + eps, Lz - eps, nR).reshape(-1, 1)))
    endpoints_right = np.hstack((Lr * np.ones((nR, 1)), np.linspace(Lz / 3, 2 / 3 * Lz, nR).reshape(-1, 1)))
    startpoints_bottom = np.hstack((np.linspace(0 + eps, Lr - eps, nB).reshape(-1, 1), Lz * np.ones((nB, 1))))
    endpoints_bottom = np.hstack((np.linspace(0.25 * Lr, 0.75 * Lr, nB).reshape(-1, 1), np.zeros((nB, 1))))
    startpoints_left = np.hstack((np.zeros((nL, 1)), np.linspace(Lz / 3, 2 / 3 * Lz, nL).reshape(-1, 1)))
    endpoints_left = np.hstack((Lr * np.ones((nL, 1)), np.linspace(0 + eps, Lz - eps, nL).reshape(-1, 1)))
    center = np.array([Lr / 2, Lz / 2])
    for i in range(nU):
        LoS_params[i, :] = LineParametrizationFromTwoPoints(startpoints_upper[i, :], endpoints_upper[i, :], center)
    for j in range(nR):
        LoS_params[j + nU, :] = LineParametrizationFromTwoPoints(startpoints_right[j, :], endpoints_right[j, :], center)
    for k in range(nB):
        LoS_params[k + nU + nR, :] = LineParametrizationFromTwoPoints(
            startpoints_bottom[k, :], endpoints_bottom[k, :], center
        )
    for l_ in range(nL):
        LoS_params[l_ + nU + nR + nB, :] = LineParametrizationFromTwoPoints(
            startpoints_left[l_, :], endpoints_left[l_, :], center
        )
    # transform parametrization to adapt it to implementation of RadonOpTubes
    # LoS_params[:, 1] = np.pi - LoS_params[:, 1]
    # LoS_params[:, 0] *= (-1)
    # transform startpoints and endpoints in format needed for interpolation
    startpoints = np.vstack((startpoints_upper, startpoints_left, startpoints_bottom, startpoints_right))
    endpoints = np.vstack((endpoints_upper, endpoints_left, endpoints_bottom, endpoints_right))
    startpoints, endpoints = cartesian_to_image_coordinates(startpoints, endpoints, Lz, h)
    return LoS_params, startpoints, endpoints


def intersects_domain(params, Lr=0.514, Lz=1.5):
    """
    This function returns a mask whose ones correspond to (p,theta) lines
    that fall inside the TCV vessel
    """
    center = np.array([Lr/2, Lz/2])
    intersection_mask = np.zeros(params.shape[0])
    for i in range(params.shape[0]):
        intersects = 1
        if np.isclose(np.tan(params[i, 1]), 0, atol=1e-6):
            if center[0] + params[i, 0] < 0 or center[0] + params[i, 0] > Lr:
                intersects = 0
        else:
            r = np.array([0, Lr])
            p = params[i, 0]
            theta = params[i, 1]
            y = center[1] + p / np.sin(theta) - (r - center[0]) / np.tan(theta)
            if np.min(y) > Lz or np.max(y) < 0:
                intersects = 0
            # check whether line is in one of the corners
            startpoint, endpoint, _ = LoS_extrema_from_LoS_params(params[i,:].reshape(1,2), Lr, Lz)
            if startpoint.size > 0:
                # lower/upper corners characterization
                # x_coord_lfs_corners = 0.972-0.624 = 0.348
                # x_coord_hfs_corners = 0.67-0.624 = 0.046
                # y_coord_hfs_upper_corners = 1.5 - (0.75 - 0.704) = 1.454
                # y_coord_hfs_lower_corners = (0.75 - 0.704) = 0.046
                # y_coord_lfs_upper_corners = 1.5 - (0.75 - 0.555) = 1.305
                # y_coord_lfs_lower_corners = (0.75 - 0.555) = 0.195
                x_coord_lfs_corners = 0.348
                x_coord_hfs_corners = 0.046
                y_coord_hfs_upper_corners = 1.454
                y_coord_hfs_lower_corners = 0.046
                y_coord_lfs_upper_corners = 1.305
                y_coord_lfs_lower_corners = 0.195
                if startpoint[0][0] > x_coord_lfs_corners and endpoint[0][0] > x_coord_lfs_corners:
                    # check whether line falls outside lfs lower right corner
                    if startpoint[0][1] < y_coord_lfs_lower_corners and endpoint[0][1]< y_coord_lfs_lower_corners:
                        intersects = 0
                    # check whether line falls outside lfs upper right corner
                    if startpoint[0][1] > y_coord_lfs_upper_corners and endpoint[0][1] > y_coord_lfs_upper_corners:
                        intersects = 0
                if startpoint[0][0] < x_coord_hfs_corners and endpoint[0][0] < x_coord_hfs_corners:
                    # check whether line falls outside hfs lower left corner
                    if startpoint[0][1] < y_coord_hfs_lower_corners and endpoint[0][1]< y_coord_hfs_lower_corners:
                        intersects = 0
                    # check whether line falls outside hfs upper left corner
                    if startpoint[0][1] > y_coord_hfs_upper_corners and endpoint[0][1] > y_coord_hfs_upper_corners:
                        intersects = 0
        intersection_mask[i] = intersects
    return intersection_mask


def plot_ptheta_LoS_tcv(params, tcv_mask_finesse=100):
    """
    This function plots, for a given (p, theta) configuration, the LoS configuration
    taking into account the TCV geometry. The shaded areas correspond to lines that fall
    outside the vessel.
    """
    tcv_mask_finesse = round(tcv_mask_finesse)
    pmin = np.min(params[:, 0])
    prange = np.max(params[:, 0]) - np.min(params[:, 0])
    # pmax corresponding to tcv geometry
    tcv_pmax = 0.5646 + 0.25
    T,P = np.meshgrid(np.linspace(0, np.pi, tcv_mask_finesse), np.linspace(-tcv_pmax, tcv_pmax, tcv_mask_finesse))
    params_PT = np.zeros((P.size, 2))
    params_PT[:, 0] = P.flatten()
    params_PT[:, 1] = T.flatten()
    PT_intersecting_tcv_mask = intersects_domain(params_PT)
    PT_intersecting_tcv_mask = PT_intersecting_tcv_mask.reshape(tcv_mask_finesse, tcv_mask_finesse)
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
        plt.plot(thetas[i], ps[i], "r", marker=".", markersize=5)
    plt.xticks([0, tcv_mask_finesse/2, tcv_mask_finesse-1], [r'$0$', r'$\pi/2$', r'$\pi$'], fontsize=20)
    plt.yticks([0, tcv_mask_finesse/2, tcv_mask_finesse-1], [r'$-p_{max}$', r'$0$', r'$p_{max}$'], fontsize=20)
    #plt.yticks([])
    plt.xlabel(r"$\theta$", fontsize=25)
    plt.ylabel(r"$p$", rotation=0, fontsize=25, labelpad=10)
    plt.title("Subsampled sinogram", fontsize=25, color="k")


def plot_LoS_tcv(params, Lr=0.514, Lz=1.5, linewidth=1):
    """
    This function plots, for a given (p, theta) configuration, the LoS configuration
    on a TCV cross-section.
    """
    # read tcv geometry from radcam
    radcam = geom.RADCAM_system()
    # pixel size for (120,40) grid on the cross-section
    h = (0.0125, 0.01285)
    # create patch to reproduce TCV shape
    extent = radcam.tile_extent_plot
    extent[:,1] = extent[:,1] * 120 - 0.5*h[0]
    extent[:,0] = extent[:,0] * 40 - 0.5*h[1]
    path = Path(extent.tolist())
    patch=PathPatch(path, facecolor='none')
    # plot LoS
    plt.figure(figsize=(8,5))
    p = plt.imshow(np.zeros((120,40)), clip_path=patch, clip_on=True, cmap='binary', vmin=0, vmax=1)
    plt.gca().add_patch(patch)
    p.set_clip_path(patch)
    plt.axis('off')
    for i in range(params.shape[0]):
        # for each LoS, compute intersections with vessel and plot it
        startpoint, endpoint, _ =\
            LoS_extrema_from_LoS_params(params[i, :].reshape(1, 2), Lr=0.514, Lz=1.5)
        x_coords = np.array([startpoint[0, 0], endpoint[0, 0]]) / Lr * 40
        y_coords = np.array([startpoint[0, 1], endpoint[0, 1]]) / Lz * 120
        plt.plot(x_coords, y_coords, "r", linewidth=linewidth)
    plt.xlim([0, 40])
    plt.ylim([0, 120])
    plt.axvline(0.3,0.03,0.97,color='k',linewidth=0.5)
    plt.axvline(39.7,0.13,0.875,color='k',linewidth=0.8)
    plt.plot([4,27.5],[0.02,0.02],'k', linewidth=2)
    plt.plot([4,27.5],[120,120],'k', linewidth=1.5)
    plt.title(r"LoS configuration", fontsize=15)
    plt.tight_layout()