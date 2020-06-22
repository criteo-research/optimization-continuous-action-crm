import os
import sys
import scipy as sp
import autograd.numpy as np
from autograd import grad, jacobian, hessian

from cyanure import Regression

base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(base_dir)
from utils.prepare import get_feature_map_by_name, get_contextual_modelling_by_params




class DirectMethod:
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
        self.feature_map = get_feature_map_by_name(self.hyperparams)
        self.reward_regressor = Regression(loss='square', penalty='l2', fit_intercept=True)

    def create_starting_parameter(self, dataset):
        self.parameter = get_contextual_modelling_by_params(self.hyperparams).get_starting_parameter(dataset)

    def reward_predictor(self, features, actions):
        """ Reward estimator \hat{\eta}(x,a) given features x and actions a

        Args:
            features (np.array): observation features
            actions (np.array): actions taken for the features

        """
        X = self.feature_map.joint_feature_map(features, actions)
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
        if not self.feature_map.anchor_points_initialized:
            self.feature_map.initialize_anchor_points(features, actions)
            self.action_anchors = self.feature_map.action_anchor_points
            # self._set_feature_map_all_actions(features)

        # self._build_reward_regressor()
        # X = self.get_feature_map(features, self.bucketize_actions(actions))
        X = self.feature_map.joint_feature_map(features, actions)
        self.reward_regressor.fit(X, rewards, lambd=self.hyperparams['reg_param_direct'])

        if self.logger:
            self.logger.info("Reward predictor learning: done!")

    def get_samples(self, features, random_seed):
        """ Samples action from the policy learned by the estimator

        Args:
            features (np.array): contextual information to sample action from
            random_seed (int)
        """
        repeated_features = np.repeat(features, len(self.action_anchors[:-1]), axis=0)
        repeated_actions = np.tile(self.action_anchors[:-1], features.shape[0])
        X = self.feature_map.joint_feature_map(repeated_features, repeated_actions)
        preds = self.reward_regressor.predict(X)
        preds = preds.reshape((features.shape[0], len(self.action_anchors[:-1])))
        return self.action_anchors[:-1][np.argmax(preds, axis=1)]

    def evaluate(self, dataset, data_valid, data_test, n_samples):
        """ Performs evaluation on dataset on train, valid and test split

        Args:
            dataset (Dataset)
            data_valid: validation set in dataset
            data_test: test set in dataset
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

