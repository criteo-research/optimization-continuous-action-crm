import os
import sys
import autograd.numpy as np
import scipy as sp
from abc import ABCMeta, abstractmethod

base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(base_dir)
from utils.prepare import get_kernel_by_name
from utils.bucketizer import Bucketizer

class FeatureMap:
    """ General Abstract Feature Map class

    """
    __metaclass__ = ABCMeta

    def __init__(self, hyperparams):
        """Initializes the class

        Attributes:
            hyperparams (dict):  dictionary of hyperparams
            K (int): number of action anchor points
            bucketizer (Bucketizer): bucketize the action space
            anchor_points_initialized (bool): check whether anchor points are initialized with bucketizer

        """
        self.hyperparams = hyperparams
        self.K = self.hyperparams['nb_quantile']
        self.bucketizer = Bucketizer(self.K, self.hyperparams['bucketize_mode'])
        self.anchor_points_initialized = False

    @abstractmethod
    def set_contextual_anchor_points(self, features):
        """ Updates the parameters of the distribution

        Args:
            features (np.array): observation features

        """
        pass

    @abstractmethod
    def contextual_feature_map(self, features):
        """ Updates the parameters of the distribution

        Args:
            features (np.array): observation features

        """
        pass

    def initialize_anchor_points(self, features, actions):
        """ Initialize anchor points

        Args:
            features (np.array): observation features
            actions (np.array): actions taken by logging policy

        """
        self.set_contextual_anchor_points(features)
        self.set_actions_anchor_points(actions)
        self.anchor_points_initialized = True

    def action_feature_map_size(self, dataset):
        """ Get action feature map size from dataset
        """
        return self.bucketizer.action_feature_map_size(dataset, self.hyperparams['bucketize_mode'])

    def set_actions_anchor_points(self, actions):
        """ Set action anchor points

        Args:
            actions (np.array): actions taken by logging policy

        """
        self.action_anchor_points = self.bucketizer.get_anchor_points(actions)[:-1]
        bandwidth = np.diff(self.action_anchor_points[:-1]).min() / 2
        self.action_bandwidth = bandwidth * self.hyperparams['action_bandwidth']
        self.kernel_action = get_kernel_by_name('gaussian')(self.action_bandwidth)

    def action_feature_map(self, actions):
        """ Creates action feature map

        Args:
            actions (np.array): actions taken by logging policy

        """
        actions_anchors = np.expand_dims(self.action_anchor_points, axis=-1)
        if self.hyperparams['action_feature_map'] == 'nystrom':
            gram_matrix = self.kernel_action(actions_anchors, actions_anchors)
            gram_pred = self.kernel_action(actions_anchors, np.expand_dims(actions, axis=-1))
            return np.dot(np.linalg.inv(sp.linalg.sqrtm(gram_matrix)), gram_pred).T
        else:
            return self.kernel_action(np.expand_dims(actions, axis=-1), actions_anchors)

    def joint_feature_map_for_all_action_anchor_points(self, features):
        """ Joint feature map for all action and anchor points

        Args:
            features (np.array): observation features

        """
        action_feature_map = self.action_feature_map(self.action_anchor_points)
        contextual_feature_map = self.contextual_feature_map(features)
        n, k = features.shape[0], self.bucketizer.K
        return np.einsum('ij, kl -> iklj', contextual_feature_map, action_feature_map).reshape((n, k+2, -1))

    def joint_feature_map(self, features, actions):
        """ Joint feature map

        Args:
            features (np.array): observation features
            actions (np.array): actions taken by logging policy

        """
        action_feature_map = self.action_feature_map(actions)
        contextual_feature_map = self.contextual_feature_map(features)
        return np.einsum('ij,ik -> ikj', contextual_feature_map, action_feature_map).reshape((features.shape[0], -1))


