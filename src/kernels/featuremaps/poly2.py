import os
import sys
import autograd.scipy.stats
import autograd.scipy as sp
import autograd.numpy as np
from autograd import grad, jacobian, elementwise_grad

base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(base_dir)

from src.kernels.featuremaps.base import FeatureMap


class Poly2(FeatureMap):
    """ Poly2 feature map utilities

    Inherits from the parent class FeatureMap

    """

    def __init__(self, *args):
        """Initializes the class

        Attributes:
            hyperparams (dict): dictionnary parameters
            name (str): name of the distribution

        """
        super(Poly2, self).__init__(*args)
        self.name = 'poly2'

    def contextual_feature_map_size(self, d):
        """ Gets size of contextual feature_map
        """
        return d**2+d

    def contextual_feature_map(self, features):
        """ Creates contextual feature map

        Args:
            features (np.array): observation features

        """
        features_squarred = np.einsum('ij,ih->ijh', features, features).reshape(features.shape[0], -1)
        return np.hstack([features_squarred, features])