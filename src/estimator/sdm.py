import os
import sys
import scipy as sp
import autograd.numpy as np

base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(base_dir)

from src.estimator.base import CRMEstimator

from src.estimator.dm import DirectMethod
from utils.prepare import get_distribution_by_params


class StochasticDirect(DirectMethod, CRMEstimator):
    def __init__(self, hyperparams, verbose, contextual_modelling):
        """Initializes the class

        Attributes:
            name (str): name of the estimator
            hyperparams (dict): dictionnary parameters
            distribution (distribution): see distribution class
            sigma (float): noise level of the SDM

        """
        DirectMethod.__init__(self, hyperparams=hyperparams, verbose=verbose)
        CRMEstimator.__init__(self, hyperparams=hyperparams, verbose=verbose, contextual_modelling=contextual_modelling)
        self.name = 'StochasticDirect'
        self.hyperparams['learning_distribution'] = 'normal'
        self.distribution = get_distribution_by_params(self.hyperparams, contextual_modelling)
        self.sigma = 1e-8

    def pred_action(self, features):
        """ Predicts actions from features and the feature map for all possible action possible

        Args:
            features (np.array): observation features

        """
        repeated_features = np.repeat(features, len(self.action_anchors[:-1]), axis=0)
        repeated_actions = np.tile(self.action_anchors[:-1], features.shape[0])
        X = self.feature_map.joint_feature_map(repeated_features, repeated_actions)
        preds = self.reward_regressor.predict(X)
        preds = preds.reshape((features.shape[0], len(self.action_anchors[:-1])))
        return self.action_anchors[:-1][np.argmax(preds, axis=1)]

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

    def evaluate(self, dataset, data_valid, data_test, n_samples):
        """ Performs evaluation on dataset on train, valid and test split

        Args:
            dataset (Dataset)
            data_valid: validation set in dataset
            data_test: test set in dataset
            n_samples (int): number of sample folds to perform bootstrap for offline evaluation on or actions to sample
                             for online evaluation

        """
        metrics = {}
        metrics['fitted_objective'] = 0.
        metrics = self.offline_evaluation(metrics, 'valid', data_valid, self.parameter, bootstrap=True, n_bootstrap=n_samples)
        metrics = self.offline_evaluation(metrics, 'test', data_test, self.parameter, bootstrap=True, n_bootstrap=n_samples)

        if self.logger:
            print("Test IPS with found alpha {} : \n {}".format(self.parameter, metrics['ips_test']))
            print("IPS Confidence T h on set: {}".format(metrics['t_h_test']))
            print("IPS Confidence Std h on set: {}".format(metrics['std_h_test']))
            print("Test SNIPS: \n {}".format(metrics['snips_test']))
            print("SNIPS Confidence Bootstrap h on set: {}".format(metrics['bootstrap_std_snips_test']))
            print('Logging policy baseline: {:2f}'.format(dataset.get_baseline_risk('test')))
            print("Emp. Mean diagnostic on val set: {}".format(metrics['em_diagnostic_test']))
            print("ESS diagnostic on val set: {}".format(metrics['ess_diagnostic_test']))

        if not dataset.evaluation_offline:
            metrics = dataset.evaluation_online(metrics, 'valid', self, n_samples)
            metrics = dataset.evaluation_online(metrics, 'test', self, n_samples)

        return metrics
