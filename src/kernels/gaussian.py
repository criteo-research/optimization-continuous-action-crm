import os
import sys
import autograd.numpy as np
# WARNING SCIPY NOT AUTOGRAD
from scipy.spatial.distance import cdist

base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(base_dir)

from src.kernels.base import Kernel
from utils.decorators import accepts

class GaussianKernel(Kernel):
    """Implementation of Gaussian kernel
    """

    @accepts(float, int)
    def __init__(self, bandwith=1.0, verbose=0):
        """
        Args:
            std (float): standard deviation
        """
        super(GaussianKernel, self).__init__(verbose)
        self._std = bandwith

    @property
    def std(self):
        return self._std

    def _evaluate(self, x1, x2):
        """
        Args:
            x1 (array): vector
            x2 (array): vector
        """
        return np.exp(-(x1 - x2)**2 / (2 * self._std))

    def _pairwise(self, X1, X2):
        """
        Args:
            X1 (np.ndarray)
            X2 (np.ndarray)
        """
        pairwise_dists = cdist(X1, X2, 'euclidean')
        return np.exp(-pairwise_dists ** 2 / self._std ** 2)

    def _differentiable_pairwise(self, X1, X2):
        """
        Args:
            X1 (np.ndarray)
            X2 (np.ndarray)
        """
        pairwise_dists = self.differentiable_cdist(X1, X2)
        return np.exp(-pairwise_dists ** 2 / self._std ** 2)

    @staticmethod
    def differentiable_cdist(x, y):
        def row_norms(X):
            return np.einsum('ij,ij->i', X, X)

        XX = row_norms(x)[:, np.newaxis]
        YY = row_norms(y)[np.newaxis, :]
        distances = -2 * np.dot(x, y.T)
        distances += XX
        distances += YY
        distances = np.maximum(distances, 0)
        return np.sqrt(distances)