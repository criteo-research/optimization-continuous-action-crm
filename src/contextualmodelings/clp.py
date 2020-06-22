import os
import sys
import autograd.scipy.stats
import autograd.scipy as sp
import autograd.numpy as np
from autograd import grad, jacobian, elementwise_grad

base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(base_dir)
EPS = 1e-8
from src.contextualmodelings.base import ContextualModelling
from utils.prepare import get_feature_map_by_name

class CounterfactualLossPredictor(ContextualModelling):
    """ Counterfactual Loss Predictor contextual modelling

    Inherits from the parent class ContextualModelling

    """

    def __init__(self, *args):
        """Initializes the class

        Attributes:
            hyperparams (dict): dictionnary parameters
            name (str): name of the distribution
            K (int): number of action anchor points
            feature_map (FeatureMap): see src.kernels.feature_map

        """
        super(CounterfactualLossPredictor, self).__init__(*args)
        self.name = 'clp'
        self.K = self.hyperparams['nb_quantile']
        self.feature_map = get_feature_map_by_name(self.hyperparams)

    def get_starting_parameter(self, dataset):
        """ Creates starting parameter

        Args:
            dataset (dataset)

        """
        m, v = self._prepare_starting_parameter(dataset)
        ctxt_size = self.feature_map.contextual_feature_map_size(self.d)
        action_size = self.feature_map.action_feature_map_size(dataset)
        return np.concatenate([self.rng.normal(m, v, size=1),
                               self.rng.normal(scale=self.scale, size=((action_size + 2) * ctxt_size + 1))])

    def get_parameters(self, parameter, features, actions, reinitialize):
        """ Updates the parameters of the distribution

        Args:
            parameter (np.array): parameter of the distribution
            features (np.array): observation features
            actions (np.array): observation actions
            reinitialize (bool): for bucketizer to be applied on new features

        """
        m, v = parameter[:-1], parameter[-1]

        if not self.feature_map.anchor_points_initialized:
            self.feature_map.initialize_anchor_points(features, actions)
            self.action_anchors = self.feature_map.action_anchor_points
            self._set_feature_map_all_actions(features)

        if reinitialize:
            self._set_feature_map_all_actions(features)

        predicted_m = self.prediction(m, features)
        return predicted_m, v

    def _set_feature_map_all_actions(self, features):
        """ Builds feature map for all actions in action set, uses method self.feature_map

        Args:
            features (np.array): observation features

        """
        self._feature_map_all_actions = np.real(self.feature_map.joint_feature_map_for_all_action_anchor_points(features))

    def soft_argmax(self, parameter, feature_map):
        """ Compute soft argmax for action prediction

        Args:
            parameter (np.array): parameter to be minimized
            feature_map (np.array): feature_map on which to make prediction

        """
        intercept = parameter[0]
        exp = np.exp(self.hyperparams['gamma'] * (np.dot(feature_map, parameter[1:])+intercept))
        return np.sum(np.einsum('ij,j->ij', exp / (np.sum(exp, axis=1, keepdims=True)+EPS), self.action_anchors), axis=1)

    def discrete_prediction(self, parameter, feature_map):
        """ Makes discrete prediction

        Args:
            parameter (np.array): parameter to be minimized
            feature_map (np.array): feature_map on which to make prediction

        """
        intercept = parameter[0]
        preds = np.dot(feature_map, parameter[1:])+intercept
        return self.action_anchors[np.argmax(preds, axis=1)]

    def prediction(self, parameter, features):
        """ Makes prediction

        Args:
            parameter (np.array): parameter to be minimized
            features (np.array): observation features

        """
        return self.soft_argmax(parameter, self._feature_map_all_actions)