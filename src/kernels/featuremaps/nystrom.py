import os
import sys
import scipy as sp
import autograd.numpy as np
from autograd import grad, jacobian, elementwise_grad
from sklearn.cluster import KMeans

base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(base_dir)

from src.kernels.featuremaps.base import FeatureMap
from utils.prepare import get_kernel_by_name

class Nystrom(FeatureMap):
    """ Nystrom feature map utilities

    Inherits from the parent class FeatureMap

    """

    def __init__(self, *args):
        """Initializes the class

        Attributes:
            hyperparams (dict): dictionnary parameters
            name (str): name of the distribution
            contextual_K (int): number of action anchor points
            contextual_bandwidth (float): contextual bandwidth for kernel in nystrom

        """
        super(Nystrom, self).__init__(*args)
        self.name = 'nystrom'
        self.contextual_K = self.hyperparams['nb_context_anchors']
        self.contextual_bandwidth = self.hyperparams['contextual_bandwidth']

    def contextual_feature_map_size(self, d):
        """ Gets size of contextual feature_map
        """
        return self.contextual_K

    def set_contextual_anchor_points(self, features):
        """ Set contextual anchor points with K-means for contextual nystrom

        Args:
            features (np.array): observation features

        """
        kmeans = KMeans(n_clusters=self.contextual_K, random_state=self.hyperparams['random_seed']).fit(features)
        self.anchor_points = kmeans.cluster_centers_
        variances = []
        labels = kmeans.labels_
        for label in np.unique(labels):
            mask = labels == label
            variances.append(np.mean((features[mask] - self.anchor_points[labels][mask]) ** 2))
        self.contextual_bandwidth *= np.sqrt(np.mean(variances))
        self.contextual_kernel = get_kernel_by_name('gaussian')(self.contextual_bandwidth)

    def contextual_feature_map(self, features):
        """ Creates contextual feature map

        Args:
            features (np.array): observation features

        """
        gram_matrix = self.contextual_kernel(self.anchor_points, self.anchor_points)
        gram_pred = self.contextual_kernel(self.anchor_points, features)
        return np.dot(np.linalg.inv(sp.linalg.sqrtm(gram_matrix)), gram_pred).T