import os
import sys
import scipy.stats
import scipy.optimize
import scipy as sp
import autograd.numpy as np
from autograd import grad, jacobian, hessian
from abc import ABCMeta, abstractmethod
EPS = 1e-8

base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(base_dir)

from src.optimizer.optimizer import Optimizer
from utils.prepare import get_distribution_by_params
from src.estimator.dm import DirectMethod

class CRMEstimator:
    """ General Abstract Counterfactual Estimator

    """
    __metaclass__ = ABCMeta
    def __init__(self, hyperparams, verbose, contextual_modelling):
        """Initializes the class

        Attributes:
            hyperparams (dict):  dictionary of hyperparameters
            parameter (np.array): parameter that is to be optimized, intialized with init_parameter
            logger (logger): display log messages
            optimizer (optimizer): see optimizer class
            distribution (distribution): see distribution class
            debug (bool): display debugging messages or not

        """
        self.hyperparams = hyperparams
        self.contextual_modelling = contextual_modelling
        self.logger = verbose
        self.optimizer = Optimizer(hyperparams)
        self.distribution = get_distribution_by_params(hyperparams, self.contextual_modelling)
        self.debug = False

    def create_starting_parameter(self, dataset):
        self.parameter = self.contextual_modelling.get_starting_parameter(dataset)

    def _set_importance_sampling_weight(self, parameter, features, actions, pi_logging, reinitialize=False):
        """ Gets importance sampling weights for off-policy evaluation

        Args:
            parameter (np.array): parameter to be minimized
            features (np.array): observation features
            actions (np.array): actions taken by logging policy
            pi_logging (np.array): propensity scores
            reinitialize (bool): for stratitification to be applied on new features

        Note:
            Computed once and stored in memory for autograd efficiency

        """
        self.distribution.update_parameter(parameter, features, actions, reinitialize)
        pi_parameter = self.distribution.pdf(features, actions)
        self.impt_smplg_weight = pi_parameter/pi_logging
        self.pi_logging = pi_logging

    @abstractmethod
    def risk(self, parameter, features, actions, rewards, pi_logging, training=False):
        """ Empirical Risk

        Args:
            parameter (np.array): parameter to be minimized
            features (np.array): observation features
            actions (np.array): actions taken by logging policy
            rewards (np.array): rewards from taken actions
            pi_logging (np.array): propensity scores
            training (bool): remove capping when evaluating for test

        """
        pass

    @abstractmethod
    def _std_penalty(self, parameter, features, actions, rewards, pi_logging):
        """ Objective function for the optimization process

        Args:
            parameter (np.array): parameter to be minimized
            features (np.array): observation features
            actions (np.array): actions taken by logging policy
            rewards (np.array): rewards from taken actions
            pi_logging (np.array): propensity scores

        """
        pass

    def entropy(self):
        """ Returns entropy of the policy distribution
        """
        return self.distribution.entropy()

    def mean_logbarrier(self, parameter, features):
        """ Mean log barrier function

        Args:
            parameter (np.array): parameter of the policy
            features (np.array): observation features

        """
        if self.hyperparams['barrier_lambda'] > 0:
            return - np.log(self.distribution.mean(parameter, features)).mean()
        else:
            return 0.

    def _objective(self, parameter, features, actions, rewards, pi_logging):
        """ Objective function for the optimization process

        Args:
            parameter (np.array): parameter to be minimized
            features (np.array): observation features
            actions (np.array): actions taken by logging policy
            rewards (np.array): rewards from taken actions
            pi_logging (np.array): propensity scores

        """
        self._set_importance_sampling_weight(parameter, features, actions, pi_logging)
        risk = self.risk(parameter, features, actions, rewards, pi_logging, training=True)
        norm_param = np.linalg.norm(parameter[1:-1])**2
        return risk + self.hyperparams['var_lambda'] * self._std_penalty(parameter, features, actions, rewards, pi_logging) \
               + 0.5 * self.hyperparams['reg_param'] * norm_param - self.hyperparams['reg_entropy'] * self.entropy()

    def _gradient_objective(self, parameter, features, actions, rewards, pi_logging):
        """ Gradient of the objective function
        """
        # import pdb; pdb.set_trace()
        return grad(self._objective)(parameter, features, actions, rewards, pi_logging)

    def _hessian_objective(self, parameter, features, actions, rewards, pi_logging):
        """ Hessian of the objective function
        """
        hessian_objective = hessian(self._objective)(parameter, features, actions, rewards, pi_logging)
        lowest_eigen_value = np.linalg.eigvals(hessian_objective).min()
        eps = 1e-3
        if self.hyperparams['method'] == 'Newton':
            n = hessian_objective.shape[0] if self.hyperparams['contextual_modelling'] != 'unique' else 2
            id = np.eye(n)
            return hessian_objective + (lowest_eigen_value + eps)*id
        return hessian_objective

    def _optimize_on(self, data):
        """ Plug in any optimizer to perform Empirical Risk Minimization

        Args:
            data (tuple): tuple of np.arrays with features, actions, rewards

        Note:
            Defines a callback function to monitor the training
        """
        features, actions, rewards, pi_logging = data

        self.monitoring = {
            'parameters': [],
            'losses': [],
        }

        self.callback_step = 0

        def callback(parameter):

            self.callback_step += 1
            if self.logger:
                print('Iteration... {}'.format(self.callback_step))
            if self.debug:
                self.monitoring['parameters'].append(parameter)
                objective = self._objective(parameter, features, actions, rewards, pi_logging)
                self.monitoring['losses'].append(objective)

        return self.optimizer.optimize(self._objective, self.parameter, self._gradient_objective,
                                       args=(features, actions, rewards, pi_logging), callback=callback,
                                       hess=self._hessian_objective)

    def _update_parameter_clp(self, data):
        """ Update clp parameters when initializing with DM
        """
        assert self.hyperparams['contextual_modelling'] == 'clp'
        direct_estimator = DirectMethod(hyperparams=self.hyperparams, verbose=self.logger)
        direct_estimator.fit(data)
        self.parameter[:-2], self.parameter[0] = direct_estimator.reward_regressor.get_weights()

    def fit(self, data):
        """ Learn parameters and fit the estimator to training data

        Args:
            data (tuple): tuple of np.arrays with features, actions, rewards
        """
        if self.hyperparams['initialize_clp']:
            self._update_parameter_clp(data)

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

    def get_ips_and_snips_metrics(self, parameter, features, actions, rewards, pi_logging):
        """ Common metric to evaluate all estimators

        Args:
            parameter (np.array): parameter to be minimized
            features (np.array): observation features
            actions (np.array): actions taken by logging policy
            rewards (np.array): rewards from taken actions
            pi_logging (np.array): propensity scores

        """
        self._set_importance_sampling_weight(parameter, features, actions, pi_logging, reinitialize=True)
        u_i = - rewards * self.impt_smplg_weight
        return np.mean(u_i), np.sum(u_i)/(np.sum(self.impt_smplg_weight)+EPS)

    def offline_evaluation(self, metrics, mode, data, parameter, confidence=0.95, bootstrap=False, n_bootstrap=1):
        """ Performs offline evaluation

        Args:
            metrics (dic): metrics dictionnary to be filled
            mode (str): train, valid or test split
            data (tuple): tuple of np.arrays with features, actions, rewards
            parameter (np.array): optimized parameter or any baseline parameter
            confidence (float): confidence level for the interval
            bootstrap (bool): choose whether to perform bootstrap or not
            n_bootstrap (int): number of bootstrap folds

        Note:
            Computes ips, snips scores. Also computes t-student test, std, bootstrap std on ips and snips, and importance
            sampling diagnostics

        Returns:
            metrics (dic): contains results information on the data split
        """
        features, actions, rewards, pi_logging = data
        rng_bootstrap = np.random.RandomState(1)
        bootstrap_ips_metric = []
        bootstrap_snips_metric = []
        bootstrap_delta_snips_metric = []
        bootstrap_t_h = []
        bootstrap_std_h = []
        bootstrap_em_diagnostic = []
        bootstrap_ess_diagnostic = []

        for n in range(n_bootstrap):
            idx = rng_bootstrap.choice(np.arange(features.shape[0]), size=features.shape[0], replace=bootstrap)
            ips_metric, snips_metric = self.get_ips_and_snips_metrics(parameter, features[idx], actions[idx],
                                                                      rewards[idx], pi_logging[idx])
            loss_logging = np.mean(-rewards[idx])
            bootstrap_ips_metric.append(ips_metric)
            bootstrap_snips_metric.append(snips_metric)
            bootstrap_delta_snips_metric.append(snips_metric - loss_logging)

            # Student-t distribution test
            n = self.impt_smplg_weight.shape[0]
            se = sp.stats.sem(self.impt_smplg_weight*rewards)
            t_h = se * sp.stats.t.ppf((1 + confidence) / 2., n - 1)
            # Gaussian distribution test
            std_h = np.std(self.impt_smplg_weight)

            bootstrap_t_h.append(t_h)
            bootstrap_std_h.append(std_h)

            # Diagnostics
            empirical_mean_diagnostic = np.mean(self.impt_smplg_weight)
            effective_sample_size_diagnostic = (np.sum(self.impt_smplg_weight)**2/(np.sum(self.impt_smplg_weight**2)+EPS))/n

            bootstrap_em_diagnostic.append(empirical_mean_diagnostic)
            bootstrap_ess_diagnostic.append(effective_sample_size_diagnostic)

        metrics['ips_{}'.format(mode)] = np.mean(bootstrap_ips_metric)
        metrics['snips_{}'.format(mode)] = np.mean(bootstrap_snips_metric)
        metrics['t_h_{}'.format(mode)] = np.mean(bootstrap_t_h)
        metrics['std_h_{}'.format(mode)] = np.mean(bootstrap_std_h)
        metrics['bootstrap_std_ips_{}'.format(mode)] = np.std(bootstrap_ips_metric)
        metrics['bootstrap_h25_ips_{}'.format(mode)] = np.percentile(bootstrap_ips_metric, 2.5)
        metrics['bootstrap_h975_ips_{}'.format(mode)] = np.percentile(bootstrap_ips_metric, 97.5)
        metrics['bootstrap_std_snips_{}'.format(mode)] = np.std(bootstrap_snips_metric)
        metrics['bootstrap_h25_snips_{}'.format(mode)] = np.percentile(bootstrap_snips_metric, 2.5)
        metrics['bootstrap_h975_snips_{}'.format(mode)] = np.percentile(bootstrap_snips_metric, 97.5)
        metrics['em_diagnostic_{}'.format(mode)] = np.mean(bootstrap_em_diagnostic)
        metrics['ess_diagnostic_{}'.format(mode)] = np.mean(bootstrap_ess_diagnostic)

        metrics['snips_delta_{}'.format(mode)] = np.mean(bootstrap_delta_snips_metric)

        metrics['bootstrap_delta_std_snips_{}'.format(mode)] = np.std(bootstrap_delta_snips_metric)
        metrics['bootstrap_delta_h25_snips_{}'.format(mode)] = np.percentile(bootstrap_delta_snips_metric, 2.5)
        metrics['bootstrap_delta_h975_snips_{}'.format(mode)] = np.percentile(bootstrap_delta_snips_metric, 97.5)


        return metrics

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
        metrics['fitted_objective'] = self.fitted_objective
        metrics = self.offline_evaluation(metrics, 'valid', data_valid, self.parameter, bootstrap=True, n_bootstrap=n_samples)
        metrics = self.offline_evaluation(metrics, 'test', data_test, self.parameter, bootstrap=True, n_bootstrap=n_samples)

        if self.logger:
            print("Test IPS with found alpha {} : \n {}".format(self.parameter, metrics['ips_test']))
            print("IPS Confidence T h on set: {}".format(metrics['t_h_test']))
            print("IPS Confidence Std h on set: {}".format(metrics['std_h_test']))
            print("Test SNIPS: \n {}".format(metrics['snips_test']))
            print("SNIPS Confidence Bootstrap h on set: {}".format(metrics['bootstrap_std_snips_test']))
            print('Logging policy baseline risk: {:2f}'.format(dataset.get_baseline_risk('test')))
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
        return self.distribution.get_samples(self.parameter, features, random_seed)