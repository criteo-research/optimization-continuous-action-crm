import os
import sys
import scipy as sp
import autograd.numpy as np
from autograd import grad, jacobian, hessian

base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(base_dir)

from src.estimator.selfnormalized import SelfNormalizedEstimator
from src.estimator.clp import CounterfactualLossPredictor
from utils.prepare import get_distribution_by_params

class SelfNormalizedCounterfactualLossPredictor(CounterfactualLossPredictor):
    """ Counterfactual Risk Minimization Estimator, see Equation (14)

    """

    def __init__(self, hyperparams, verbose, init_parameter):
        """Initializes the class

        Attributes:
            name (str): name of the estimator

        """
        CounterfactualLossPredictor.__init__(self, hyperparams=hyperparams, verbose=verbose,
                                                          init_parameter=init_parameter)
        self.name = 'SelfNormalizedCounterfactualLossPredictor'

    def _inverse_propensity_score(self, parameter, features, actions, rewards, pi_logging, training=False):
        """ Self-normalized inverse propensity score instead of vanilla IPS

        Args:
            parameter (np.array): parameter to be minimized
            features (np.array): observation features
            actions (np.array): actions taken by logging policy
            rewards (np.array): rewards from taken actions
            pi_logging (np.array): propensity score
            training (bool): remove capping when evaluating for test

        """
        if not self.actions_buckets_built:
            self.set_action_set(actions)
            self.sigma = np.diff(self.action_set[:-1]).min() / 2
            # self.sigma = 1e-8
            self.kernel = self._get_kernel(bandwith=self.sigma)
            self._set_feature_map_all_actions(features)
            self.actions_buckets_built = True

        predicted_actions = self.soft_argmax(parameter, self._feature_map)
        self.impt_smplg_weight = self.kernel(predicted_actions-actions)/pi_logging
        u_i = - rewards * self.impt_smplg_weight
        return np.sum(u_i)/np.sum(self.impt_smplg_weight)

