import os
import sys
import scipy as sp
import autograd.numpy as np
from autograd import grad, jacobian, hessian
from cyanure import Regression

base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(base_dir)

from src.estimator.base import CRMEstimator


class DirectEstimator:
    """ General Abstract Direct Method Estimator

    """

    def __init__(self, hyperparams, verbose, **kw):
        """Initializes the class

        Attributes:
            hyperparams (dict):  dictionary of hyperparameters
            name (str): name of the estimator
            K (int):  number of actions anchors
            eps (float): value to pad the action set quantiles with, 0 is impossible for lognormal distribution
            reward_predictor_trained (bool): to initialize the reward predictor
            logger (logger): display log messages
        """
        self.hyperparams = hyperparams
        self.name = 'Direct'
        self.K = self.hyperparams['nb_quantile']
        self.eps = 1e-8
        self.reward_predictor_trained = False
        self.logger = verbose

    def _get_kernel(self, bandwith=None):
        """ Get kernel function for the feature map, according to hyperparameter setups, can be gaussian or indicator

        Args:
            bandwith (float): bandwith for the gaussian kernel

        """
        if bandwith == None:
            bandwith = self.hyperparams['kernel_bandwidth']
        if self.hyperparams['feature_map_kernel'] == 'gaussian':
            def gaussian_kernel(u):
                return np.exp(-0.5 * (u / bandwith) ** 2) / (bandwith * np.sqrt(2 * np.pi))
            return gaussian_kernel
        elif self.hyperparams['feature_map_kernel'] == 'indicator':
            def indicator(u):
                return np.where(u == 0., 1., 0.)
            return indicator
        else:
            raise NotImplementedError

    def _build_reward_regressor(self):
        """ Builds reward regressor
        """
        self.reward_regressor = Regression(loss='square', penalty='l2', fit_intercept=True)

    def set_action_set(self, actions):
        """ Builds anchor action point sets for the direct estimator

        Args:
            actions (np.array): actions drawn from the logging policy to build quantiles on
        """
        self.quantiles = np.quantile(actions, np.linspace(0, 1, self.K+1))
        self.action_set = np.pad(self.quantiles, 1, 'constant', constant_values=(self.eps, np.inf))

    def bucketize_actions(self, a):
        """ Gets the closets anchor point of an action

        Args:
            a (np.array): actions for which to find closets anchor points, i.e actions to bucketize
        """
        actions = a.copy()
        for k in range(self.K + 2):
            inf, sup = self.action_set[k], self.action_set[k + 1]
            mask = (actions > inf) & (actions < sup)
            actions[mask] = inf
        return actions

    def get_feature_map(self, features, actions):
        """ Get the feature map of features and actions by finding closets anchor points for each actions

        Args:
            features (np.array): observation features
            actions (np.array): actions taken for the features

        """
        kernel = self._get_kernel()
        return np.concatenate([features * kernel(self.bucketize_actions(actions) - a_i)[:,None] for a_i in self.action_set[:-1]], axis=1)

    def reward_predictor(self, features, actions):
        """ Reward estimator \hat{\eta}(x,a) given features x and actions a

        Args:
            features (np.array): observation features
            actions (np.array): actions taken for the features

        """
        X = self.get_feature_map(features, actions)
        return self.reward_regressor.predict(X)

    def fit(self, data):
        """ Fits the estimator on data, see method self._fit_reward_regressor
        """
        self._fit_reward_regressor(data)

    def _fit_reward_regressor(self, data):
        """ Learn parameters and fit the estimator to training data

        Args:
            data (tuple): tuple of np.arrays with features, actions, rewards
        """
        if self.logger:
            self.logger.info("Reward predictor fitting in progress...")

        features, actions, rewards, _ = data
        self.set_action_set(actions)
        self._build_reward_regressor()
        X = self.get_feature_map(features, self.bucketize_actions(actions))
        self.reward_regressor.fit(X, rewards, lambd=self.hyperparams['reg_param_direct'])

        if self.logger:
            self.logger.info("Reward predictor learning: done!")

    def get_samples(self, features, random_seed):
        """ Samples action from the policy learned by the estimator

        Args:
            features (np.array): contextual information to sample action from
            random_seed (int)

        """
        repeated_features = np.repeat(features, len(self.action_set[:-1]), axis=0)
        repeated_actions = np.tile(self.action_set[:-1], features.shape[0])
        X = self.get_feature_map(repeated_features, repeated_actions)
        preds = self.reward_regressor.predict(X)
        preds = preds.reshape((features.shape[0], len(self.action_set[:-1])))
        return self.action_set[:-1][np.argmax(preds, axis=1)]

    def evaluate(self, dataset, data_train, data_valid, data_test, n_samples):
        """ Performs evaluation on dataset on train, valid and test split

        Args:
            n_samples (int): number of sample folds to perform bootstrap for offline evaluation on or actions to sample
                             for online evaluation

        Note:
            Offline evaluation is not possible with deterministic direct method
        """
        metrics = {}

        if not dataset.evaluation_offline:
            metrics = dataset.evaluation_online(metrics, 'valid', self, n_samples)
            metrics = dataset.evaluation_online(metrics, 'test', self, n_samples)

        return metrics

