import matplotlib.pyplot as plt
import numpy as np
import cupy as cp
import src.tomo_fusion.helpers as tools
import skimage.transform as skimt
from pyxu.info import ptype as pxt
from pyxu_diffops.operator import AnisCoherenceEnhancingDiffusionOp
import pyxu.util as pycu
import sys
import argparse
import pyxu.abc as pxa
import scipy.sparse as sp
import time
import pyxu.operator as pyxop
import pyxu.info.ptype as pyct
import pyxu.info.deps as pxd
import pyxu.info.warning as pxw
from pyxu.abc.operator import QuadraticFunc, LinOp
import copy


class ExplicitLinOpSparseMatrix(LinOp):
    def __init__(self, dim_shape, mat):
        assert len(mat.shape) == 2, "Matrix `mat` must be a 2-dimensional array"
        super().__init__(dim_shape=dim_shape, codim_shape=mat.shape[0])
        self.mat = mat
        self.num_pixels = np.prod(dim_shape[-2:])

    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        arr = arr.reshape(*arr.shape[:-2], self.num_pixels)
        y = self.mat.dot(arr.T).T
        y = y.reshape(*y.shape[:-1], self.codim_shape)
        return y

    def adjoint(self, arr: pxt.NDArray) -> pxt.NDArray:
        arr = arr.reshape(*arr.shape[:-1], self.codim_shape)
        y = self.mat.T.dot(arr.T).T
        y = y.reshape(*y.shape[:-1], *self.dim_shape[1:])
        return y


def DataFidelityFunctional(dim_shape: pxt.NDArrayShape, tomo_data: pxt.NDArray, sigma_err: pxt.NDArray, geometry_matrix: pxt.SparseArray) -> pxt.OpT:
    # define explicit LinOp from geometry matrix
    forward_model_linop = ExplicitLinOpSparseMatrix(dim_shape=dim_shape, mat=geometry_matrix)
    # define data-fidelity functional
    op = 1 / (2 * sigma ** 2 ) * pyxop.SquaredL2Norm(dim_shape=(tomo_data.size,)).argshift(-tomo_data.ravel()) * forward_model_linop