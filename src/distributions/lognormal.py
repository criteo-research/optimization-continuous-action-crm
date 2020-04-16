import os
import sys
import autograd.scipy.stats
import autograd.scipy as sp
import autograd.numpy as np

base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(base_dir)

from src.distributions.base import Distribution
from utils.stratificator import Stratificator

EPS = 1e-7

class LogNormalDistribution(Distribution):
    """ Log Normal Distribution utilities

    Inherits from the parent class Distribution

    """

    def __init__(self, *args):
        """Initializes the class

        Attributes:
            hyperparams (dict): dictionnary parameters
            name (str): name of the distribution

        """
        super(LogNormalDistribution, self).__init__(*args)
        self.name = 'Log Normal'
        if self.hyperparams['contextual_modelling'] == 'strat':
            self.stratificator = Stratificator(self.hyperparams['nb_quantile'])

    def pdf(self, features, x):
        """ Pdf of the lognormal distribution

        Args:
            features (np.array): observation features
            x (np.array): actions taken to evaluate the pdf on

        """
        return np.exp(-(np.log(x) - self.mu) ** 2 / (2 * self.sigma ** 2 + EPS)) / (x * self.sigma * np.sqrt(2 * np.pi) + EPS)

    def entropy(self):
        """ Returns entropy of the distribution
        """
        return np.mean(np.log(self.sigma*np.exp(self.mu+1/2)*np.sqrt(2*np.pi)+EPS))

    def get_samples(self, parameter, features, random_seed=42):
        """ Samples from the distribution

        Args:
            features (np.array): contextual information to sample action from
            random_seed (int)

        """
        size = features.shape[0]
        rng = np.random.RandomState(random_seed)
        return rng.lognormal(self.mu, self.sigma, size)

    def update_parameter(self, parameter, features, reinitialize=False):
        """ Updates the parameters of the log normal distribution

        Args:
            parameter (np.array): parameter of the distribution to be minimized
            features (np.array): observation features
            reinitialize (bool): for stratitification to be applied on new features

        """
        if self.hyperparams['contextual_modelling'] == 'unique':
            m, v = parameter[:-1], parameter[-1]
            self.sigma = np.sqrt(np.log(v / m ** 2 + 1))
            self.mu = np.log(m) - self.sigma ** 2 / 2

        elif self.hyperparams['contextual_modelling'] == 'linear':
            intercept_coeff, mean_coeff, var_coeff = parameter[0], parameter[1:-1], parameter[-1]
            m_linear = np.dot(features, mean_coeff) + intercept_coeff
            self.sigma = np.sqrt(np.log(var_coeff / m_linear ** 2 + 1))
            self.mu = np.log(m_linear) - self.sigma ** 2 / 2

        elif self.hyperparams['contextual_modelling'] == 'kern-poly2':
            f = np.einsum('ij,ih->ijh', features, features).reshape(features.shape[0], -1)
            intercept_coeff, mean_coeff, var_coeff = parameter[0], parameter[1:-1], parameter[-1]
            m_kern = np.dot(f, mean_coeff) + intercept_coeff

            self.sigma = np.sqrt(np.log(var_coeff / m_kern ** 2 + 1))
            self.mu = np.log(m_kern) - self.sigma ** 2 / 2

        elif self.hyperparams['contextual_modelling'] == 'kern-lin-poly2':
            n = features.shape[1]
            intercept_coeff, mean_coeff_lin, mean_coeff_kern, var_coeff = parameter[0], parameter[1:n+1], parameter[n+1:-1], parameter[-1]
            m_linear = np.dot(features, mean_coeff_lin) + intercept_coeff
            f = np.einsum('ij,ih->ijh', features, features).reshape(features.shape[0], -1)
            m_kern = np.dot(f, mean_coeff_kern)
            m = m_kern + m_linear
            self.sigma = np.sqrt(np.log(var_coeff / m ** 2 + 1))
            self.mu = np.log(m) - self.sigma ** 2 / 2

        elif self.hyperparams['contextual_modelling'] == 'strat':
            m, v = parameter[:-1], parameter[-1]
            if not self.stratificator.initialized or reinitialize:
                self.strat_tables = self.stratificator.get_table_strats(features)
            m_strat = m[self.strat_tables]

            self.sigma = np.sqrt(np.log(v / m_strat ** 2 + 1))
            self.mu = np.log(m_strat) - self.sigma ** 2 / 2
