import os
import sys
import autograd.scipy.stats
import autograd.scipy as sp
import autograd.numpy as np

base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(base_dir)

from src.contextualmodelings.base import ContextualModelling
from utils.prepare import get_feature_map_by_name

class SoftMax(ContextualModelling):
    """ Discrete CRM softmax utilities

    Inherits from the parent class ContextualModelling

    """

    def __init__(self, *args):
        """Initializes the class

        Attributes:
            hyperparams (dict): dictionnary parameters
            name (str): name of the distribution
            feature_map (FeatureMap): see src.kernels.feature_map

        """
        super(SoftMax, self).__init__(*args)
        self.name = 'SoftMax'
        self.feature_map = get_feature_map_by_name(self.hyperparams)

    @staticmethod
    def indicator(u):
        return np.where(u == 0., 1., 0.)

    def get_starting_parameter(self, dataset):
        """ Creates starting parameter

        Args:
            dataset (dataset)

        """
        m, v = self._prepare_starting_parameter(dataset)
        ctxt_size = self.feature_map.contextual_feature_map_size(self.d)
        action_size = self.feature_map.action_feature_map_size(dataset)
        return np.concatenate([self.rng.normal(m, v, size=1),
                               self.rng.normal(scale=self.scale, size=((action_size + 2) * ctxt_size))])

    def get_parameters(self, parameter, features, actions, reinitialize):
        """ Updates the parameters of the distribution

        Args:
            parameter (np.array): parameter of the distribution
            features (np.array): observation features
            actions (np.array): observation actions
            reinitialize (bool): for bucketizer to be applied on new features

        """
        if not self.feature_map.anchor_points_initialized:
            self.feature_map.initialize_anchor_points(features, actions)
            self.bins = self.feature_map.action_anchor_points
            self.inds = np.digitize(actions, self.bins, right=True)
            bucketized_actions = self.bins[self.inds]
            fm_a = np.concatenate([self.indicator(bucketized_actions - b)[:, np.newaxis] for b in self.bins],
                                       axis=1)
            fm_c = self.feature_map.contextual_feature_map(features)
            self.fm_c_shape = fm_c.shape[1]
            self.representation = np.einsum('ij, ik -> ikj', fm_c, fm_a)

        if reinitialize:
            self.inds = np.digitize(actions, self.bins, right=True)
            self.inds[self.inds == np.max(self.inds)] = np.max(self.inds) -1
            bucketized_actions = self.bins[self.inds]
            fm_a = np.concatenate([self.indicator(bucketized_actions - b)[:, np.newaxis] for b in self.bins],
                                       axis=1)
            fm_c = self.feature_map.contextual_feature_map(features)
            self.representation = np.einsum('ij, ik -> ikj', fm_c, fm_a)

        itcp, param = parameter[0], parameter[1:]
        param = param.reshape(self.bins.shape[0], self.fm_c_shape)
        return np.einsum('ikl, kl-> ik', self.representation, param) + itcp, self.inds

