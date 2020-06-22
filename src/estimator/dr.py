import os
import sys
import scipy as sp
import autograd.numpy as np
from autograd import grad, jacobian, hessian

base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(base_dir)

from src.estimator.base import CRMEstimator
from src.estimator.dm import DirectMethod

class DoublyRobust(CRMEstimator, DirectMethod):
    """ Doubly Robust estimator

    Inherits from CRM base class and direct estimator
    """

    def __init__(self, hyperparams, verbose, contextual_modelling):
        """Initializes the class

        Attributes:
            name (str): name of the estimator
            K (int):  number of actions anchors
            reward_predictor_trained (bool): to initialize the reward predictor

        """
        CRMEstimator.__init__(self, hyperparams=hyperparams, verbose=verbose, contextual_modelling=contextual_modelling)
        DirectMethod.__init__(self, hyperparams=hyperparams, verbose=verbose)
        self.name = 'DoublyRobust'
        self.K = self.hyperparams['nb_quantile']
        self.reward_predictor_trained = False

    def _inverse_propensity_score(self, features, actions, rewards):
        """ IPS term in the objective function

        Args:
            features (np.array): observation features
            actions (np.array): actions taken by logging policy
            rewards (np.array): rewards from taken actions

        """
        reward_predictions = self.reward_predictor(features, actions)
        reward_differences = rewards - reward_predictions
        return np.mean(-reward_differences*self.impt_smplg_weight)

    def direct_method(self, parameter, features, actions):
        """ Direct method term in the objective function

        Args:
            parameter (np.array): parameter to be minimized
            features (np.array): observation features
            actions (np.array): actions taken by logging policy

        """
        repeated_features = np.repeat(features, len(self.action_anchors[:-1]), axis=0)
        repeated_actions = np.tile(self.action_anchors[:-1], features.shape[0])
        self.distribution.update_parameter(parameter, repeated_features, reinitialize=True)
        pi_parameter = self.distribution.pdf(repeated_features, repeated_actions)
        return 1/features.shape[0] * np.sum(pi_parameter * (-self.reward_predictor(repeated_features, repeated_actions)))

    def _std_penalty(self, parameter, features, actions, rewards, pi_logging):
        """ Compute the std penalty of the CRM estimator
        """
        reward_predictions = self.reward_predictor(features, actions)
        reward_differences = rewards - reward_predictions
        return np.std(reward_differences*self.impt_smplg_weight)

    def _objective(self, parameter, features, actions, rewards, pi_logging):
        """ Objective function for the optimization process

        Args:
            parameter (np.array): parameter to be minimized
            features (np.array): observation features
            actions (np.array): actions taken by logging policy
            rewards (np.array): rewards from taken actions
            pi_logging (np.array): propensity scores

        """
        self._set_importance_sampling_weight(parameter, features, actions, pi_logging, reinitialize=True)
        norm_param = np.linalg.norm(parameter[1:-1])**2
        ips_score = self._inverse_propensity_score(features, actions, rewards)
        std_penalty = self._std_penalty(parameter, features, actions, rewards, pi_logging)
        entropy = self.entropy()
        dm_score = self.direct_method(parameter, features, actions)
        return ips_score + dm_score + self.hyperparams['var_lambda'] * std_penalty \
               + 0.5 * self.hyperparams['reg_param'] * norm_param - self.hyperparams['reg_entropy'] * entropy

    def fit(self, data):
        """ Learn parameters and fit the estimator to training data

        Args:
            data (tuple): tuple of np.arrays with features, actions, rewards
        """
        self._fit_reward_regressor(data)

        if self.logger:
            self.logger.info("Counterfactual Risk Minimization in progress...")
        optimized = self._optimize_on(data)
        x, f, d = optimized
        self.fitted_objective = f
        self.parameter = x
        self.information = d

        if self.logger:
            if self.information['warnflag']:
                self.logger.warning("Convergence was not reached!")
                self.logger.warning("Gradient at the minimum was {} \nOptimization stopped because {}".format(
                    self.information['grad'], self.information['task']))
            else:
                self.logger.info("Counterfactual optimization is done!")
                print(optimized)
                print("Objective on the training set: {}".format(self.fitted_objective))

