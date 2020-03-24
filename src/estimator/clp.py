import os
import sys
import scipy as sp
import autograd.numpy as np
from autograd import grad, jacobian, hessian

base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(base_dir)

from src.estimator.ips import InversePropensityScore
from src.estimator.direct import DirectEstimator
from utils.prepare import get_distribution_by_params

class CounterfactualLossPredictor(InversePropensityScore, DirectEstimator):
    """ Counterfactual Loss Predictor Estimator, see Equation (14)

    Inherits from IPS and Direct estimators

    """

    def __init__(self, hyperparams, verbose, init_parameter):
        """Initializes the class

        Attributes:
            name (str): name of the estimator
            K (int):  number of actions anchors
            actions_buckets_built (bool): to initialize action anchors
            distribution (distribution): see distribution class

        """
        InversePropensityScore.__init__(self, hyperparams=hyperparams, verbose=verbose,
                                                          init_parameter=init_parameter)
        DirectEstimator.__init__(self, hyperparams=hyperparams, verbose=verbose)
        self.name = 'CounterfactualLossPredictor'
        self.K = self.hyperparams['nb_quantile']
        self.actions_buckets_built = False
        self.hyperparams['learning_distribution'] = 'normal'
        self.distribution = get_distribution_by_params(self.hyperparams)

    def soft_argmax(self, parameter, feature_map):
        """ Compute soft argmax for action prediction

        Args:
            parameter (np.array): parameter to be minimized
            feature_map (np.array): feature_map on which to make prediction

        """
        intercept = parameter[0]
        exp = np.exp(self.hyperparams['gamma'] * (np.dot(feature_map, parameter[:-1])+intercept))
        return np.sum(np.einsum('ij,j->ij', exp / np.sum(exp, axis=1, keepdims=True),
                                             self.action_set[:-1]), axis=1)

    def _set_importance_sampling_weight(self, parameter, features, actions, pi_logging, reinitialize=False):
        """ Gets importance sampling weights for off-policy evaluation

        Args:
            parameter (np.array): parameter to be minimized
            features (np.array): observation features
            actions (np.array): actions taken by logging policy
            pi_logging (np.array): propensity scores
            reinitialize (bool): for stratitification to be applied on new features

        Note:
            (i) Computed once and stored in memory for autograd efficiency
            (ii) Explicit assignment of loc and scale of the distribution necessary here

        """
        self.distribution.loc = self.pred_action(features)
        self.distribution.scale = self.sigma
        pi_parameter = self.distribution.pdf(features, actions)
        self.impt_smplg_weight = pi_parameter/pi_logging

    def feature_map(self, features, action):
        """ Builds feature map

        Args:
            features (np.array): observation features
            action (np.array): single action for which to build feature map \phi(x,a)

        Note:
            Builds feature map with the kernel defined in hyperparameters

        """
        kernel = self._get_kernel()
        return np.concatenate([features * kernel(action - a) for a in self.action_set[:-1]], axis=1)

    def _set_feature_map_all_actions(self, features):
        """ Builds feature map for all actions in action set, uses method self.feature_map

        Args:
            features (np.array): observation features

        """
        feature_map = np.hstack([self.feature_map(features, a) for a in self.action_set[:-1]])
        self._feature_map = np.reshape(feature_map, (feature_map.shape[0], len(self.action_set[:-1]), -1))

    def pred_action(self, features):
        """ Predicts actions from features and the feature map for all possible action possible

        Args:
            features (np.array): observation features

        """
        self._set_feature_map_all_actions(features)
        return self.soft_argmax(self.parameter, self._feature_map)

    def _inverse_propensity_score(self, parameter, features, actions, rewards, pi_logging, training=False):
        """ Inverse propensity score

        Args:
            parameter (np.array): parameter to be minimized
            features (np.array): observation features
            actions (np.array): actions taken by logging policy
            rewards (np.array): rewards from taken actions
            pi_logging (np.array): propensity scores
            training (bool): remove capping when evaluating for test

        """
        if not self.actions_buckets_built:
            self.set_action_set(actions)
            self.sigma = np.diff(self.action_set[:-1]).min() / 2
            self.kernel = self._get_kernel(bandwith=self.sigma)
            self._set_feature_map_all_actions(features)
            self.actions_buckets_built = True

        predicted_actions = self.soft_argmax(parameter, self._feature_map)
        self.impt_smplg_weight = self.kernel(predicted_actions-actions)/pi_logging
        return np.sum(-rewards* self._clip_or_not(self.impt_smplg_weight, is_training=training))

    def _objective(self, parameter, features, actions, rewards, pi_logging):
        """ Objective function for the optimization process

        Args:
            parameter (np.array): parameter to be minimized
            features (np.array): observation features
            actions (np.array): actions taken by logging policy
            rewards (np.array): rewards from taken actions
            pi_logging (np.array): propensity scores

        """
        norm_param = np.linalg.norm(parameter[1:-1])**2
        ips = self._inverse_propensity_score(parameter, features, actions, rewards, pi_logging, training=True)
        self._set_importance_sampling_weight(parameter, features, actions, pi_logging)
        return ips + self.hyperparams['var_lambda'] * self._std_penalty(parameter, features, actions, rewards, pi_logging) \
        + 0.5 * self.hyperparams['reg_param'] * norm_param - self.hyperparams['reg_entropy'] * self.entropy()

    def fit(self, data):
        """ Learn parameters and fit the estimator to training data

        Args:
            data (tuple): tuple of np.arrays with features, actions, rewards
        """
        if self.hyperparams['initialize_clp']:
            self._fit_reward_regressor(data)
            self.parameter[:-1], self.parameter[0] = self.reward_regressor.get_weights()

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

    def get_samples(self, features, random_seed):
        """ Samples action from the policy learned by the estimator

        Args:
            features (np.array): contextual information to sample action from
            random_seed (int)

        """
        preds = self.pred_action(features)
        rng = np.random.RandomState(random_seed)
        noise = rng.normal(loc=0, scale=self.sigma, size=features.shape[0])
        return preds + noise


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


