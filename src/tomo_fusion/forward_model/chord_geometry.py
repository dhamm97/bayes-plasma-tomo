import numpy as np
import scipy.io as scio
import os

import src.tomo_fusion.forward_model.LoS_handling as LoS


class RADCAM_system:
    def __init__(self):
        file_dir = os.path.dirname(os.path.abspath(__file__))
        bolo_chords = scio.loadmat(file_dir+"/chords/bolo_chord_start_end_points.mat")["B"]
        sxr_chords = scio.loadmat(file_dir+"/chords/sxr_chord_start_end_points.mat")["A"]
        axuv_chords = scio.loadmat(file_dir+"/chords/axuv_chord_start_end_points.mat")["C"]
        self.bolo_xchords = bolo_chords[: round(bolo_chords.shape[0] / 2), :]
        self.bolo_ychords = bolo_chords[round(bolo_chords.shape[0] / 2):, :]
        self.sxr_xchords = sxr_chords[: round(sxr_chords.shape[0] / 2), :]
        self.sxr_ychords = sxr_chords[round(sxr_chords.shape[0] / 2):, :]
        self.axuv_xchords = axuv_chords[: round(axuv_chords.shape[0] / 2), :]
        self.axuv_ychords = axuv_chords[round(axuv_chords.shape[0] / 2):, :]
        # tiles coordinates
        self.x_left_tile = 0.624
        self.x_right_tile = 1.135#1.138(1.138 was what I had so far, which brought small discrepancy with spc matrix... now almost identical!)  # 1.124 approximation used for square pixels
        self.y_lower_tile = -0.75
        self.y_upper_tile = 0.75
        self.tile_extent = np.array(
            [
                [0.972, self.y_lower_tile],
                [self.x_right_tile, -0.555],
                [self.x_right_tile, 0.555],
                [0.972, self.y_upper_tile],
                [0.67, self.y_upper_tile],
                [self.x_left_tile, 0.704],
                [self.x_left_tile, -0.704],
                [0.67, self.y_lower_tile],
                [0.972, self.y_lower_tile],
            ]
        )
        self.tile_extent[:, 0] -= self.x_left_tile
        self.tile_extent[:, 1] -= self.y_lower_tile
        self.tile_extent_plot = self.tile_extent
        # for plotting purposes, we flip to have origin at upper left corner and normalize. We will also need to shift by -0.5*h, with h discretization step
        self.tile_extent_plot[:, 0] /= (self.x_right_tile-self.x_left_tile)
        self.tile_extent_plot[:, 1] = (1.5 - self.tile_extent[:, 1]) / 1.5
        # shift coordinates to have origin at lower left corner, corresponding to (r,z)=(0.624,-0.75)
        self.bolo_xchords -= self.x_left_tile
        self.bolo_ychords -= self.y_lower_tile
        self.sxr_xchords -= self.x_left_tile
        self.sxr_ychords -= self.y_lower_tile
        self.axuv_xchords -= self.x_left_tile
        self.axuv_ychords -= self.y_lower_tile
        # compute LoS_parametrizations
        center = np.array([(self.x_right_tile - self.x_left_tile) / 2, (self.y_upper_tile - self.y_lower_tile) / 2])
        self.bolo_LoS_params, self.bolo_startpoints, self.bolo_endpoints = LoS.generate_LoS_from_point_couples(
            self.bolo_xchords, self.bolo_ychords, center, tcv_coords=False
        )
        self.sxr_LoS_params, self.sxr_startpoints, self.sxr_endpoints = LoS.generate_LoS_from_point_couples(
            self.sxr_xchords, self.sxr_ychords, center, tcv_coords=False
        )
        self.axuv_LoS_params, self.axuv_startpoints, self.axuv_endpoints = LoS.generate_LoS_from_point_couples(
            self.axuv_xchords, self.axuv_ychords, center, tcv_coords=False
        )
