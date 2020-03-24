import os
import sys
import scipy as sp
import autograd.numpy as np
from autograd import grad, jacobian, hessian

base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(base_dir)

from src.estimator.base import CRMEstimator
from src.estimator.direct import DirectEstimator


class DoublyRobust(CRMEstimator, DirectEstimator):
    """ Doubly Robust estimator in Equation (7)

    Inherits from CRM base class and direct estimator
    """

    def __init__(self, hyperparams, verbose, init_parameter):
        """Initializes the class

        Attributes:
            name (str): name of the estimator
            K (int):  number of actions anchors
            reward_predictor_trained (bool): to initialize the reward predictor
        """
        CRMEstimator.__init__(self, hyperparams=hyperparams, verbose=verbose,
                                                          init_parameter=init_parameter)
        DirectEstimator.__init__(self, hyperparams=hyperparams, verbose=verbose)
        self.name = 'DoublyRobust'
        self.K = self.hyperparams['nb_quantile']
        self.reward_predictor_trained = False

    def _inverse_propensity_score(self, parameter, features, actions, rewards, pi_logging):
        """ IPS term in the objective function in Equation (7)

        Args:
            parameter (np.array): parameter to be minimized
            features (np.array): observation features
            actions (np.array): actions taken by logging policy
            rewards (np.array): rewards from taken actions
            training (bool): remove capping when evaluating for test

        """
        reward_predictions = self.reward_predictor(features, self.bucketize_actions(actions))
        reward_differences = rewards - reward_predictions
        return np.mean(-reward_differences*self.impt_smplg_weight)

    def direct_method(self, parameter, features, actions):
        """ Direct method term in the objective function in Equation (7)

        Args:
            parameter (np.array): parameter to be minimized
            features (np.array): observation features
            actions (np.array): actions taken by logging policy

        """
        repeated_features = np.repeat(features, len(self.action_set[:-1]), axis=0)
        repeated_actions = np.tile(self.action_set[:-1], features.shape[0])
        self.distribution.update_parameter(parameter, repeated_features, reinitialize=True)
        pi_parameter = self.distribution.pdf(repeated_features, repeated_actions)
        return 1/features.shape[0] * np.sum(pi_parameter * (-self.reward_predictor(repeated_features, repeated_actions)))

    def _std_penalty(self, parameter, features, actions, rewards, pi_logging):
        """ Compute the std penalty of the CRM estimator
        """
        reward_predictions = self.reward_predictor(features, self.bucketize_actions(actions))
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
        ips_score = self._inverse_propensity_score(parameter, features, actions, rewards, pi_logging)
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
        self.fitting_risk = f
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
                print("Risk on the training set: {}".format(self.fitting_risk))

    def evaluate(self, dataset, data_train, data_valid, data_test, n_samples):
        """ Performs evaluation on dataset on train, valid and test split

        Args:
            n_samples (int): number of sample folds to perform bootstrap for offline evaluation on or actions to sample
                             for online evaluation

        """
        metrics = {}

        metrics = self.offline_evaluation(metrics, 'train', data_train, self.parameter)
        metrics = self.offline_evaluation(metrics, 'valid', data_valid, self.parameter)
        metrics = self.offline_evaluation(metrics, 'test', data_test, self.parameter, bootstrap=True, n_bootstrap=n_samples)

        if self.logger:
            print("Test IPS with found alpha {} : \n {}".format(self.parameter, metrics['ips_test']))
            print("IPS Confidence T h on set: {}".format(metrics['t_h_test']))
            print("IPS Confidence Std h on set: {}".format(metrics['std_h_test']))
            print("Test SNIPS: \n {}".format(metrics['snips_test']))
            print("SNIPS Confidence Bootstrap h on set: {}".format(metrics['bootstrap_h_snips_test']))
            print('Logging policy baseline: {:2f}'.format(dataset.get_baseline_risk('test')))
            print("Emp. Mean diagnostic on val set: {}".format(metrics['em_diagnostic_test']))
            print("ESS diagnostic on val set: {}".format(metrics['ess_diagnostic_test']))

        if not dataset.evaluation_offline:
            metrics = dataset.evaluation_online(metrics, 'valid', self, n_samples)
            metrics = dataset.evaluation_online(metrics, 'test', self, n_samples)

        return metrics

    def get_samples(self, features, random_seed):
        """ Samples action from the policy learned by the estimator

        Args:
            features (np.array): contextual information to sample action from
            random_seed (int)

        """
        self.distribution.update_parameter(self.parameter, features, reinitialize=True)
        return self.distribution.get_samples(self.parameter, features, random_seed)

    def _optimize_on(self, data):
        """ Plug in any optimizer to perform Empirical Risk Minimization

        Args:
            data (tuple): tuple of np.arrays with features, actions, rewards

        Note:
            Defines a callback function to monitor the training and plot metrics via mlflow
        """
        features, actions, rewards, pi_logging = data

        self.monitoring = {
            'parameters': [],
            'losses': [],

        }
        self.callback_step = 0

        def callback(parameter):
            if self.debug:
                self.monitoring['parameters'].append(parameter)
                objective = self._objective(parameter, features, actions, rewards, pi_logging)
                self.monitoring['losses'].append(objective)
            self.callback_step += 1
            print('Iteration... {}'.format(self.callback_step))

        return self.optimizer.optimize(self._objective, self.parameter, self._gradient_objective,
                                         args=(features, actions, rewards, pi_logging), callback=callback, hess=self._hessian_objective)
