import os
import sys
import autograd.scipy.stats
import autograd.scipy as sp
import autograd.numpy as np

base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(base_dir)

from src.distributions.base import Distribution
from utils.stratificator import Stratificator

class NormalDistribution(Distribution):
    """ Normal Distribution utilities

    Inherits from the parent class Distribution

    """

    def __init__(self, *args):
        """Initializes the class

        Attributes:
            hyperparams (dict): dictionnary parameters
            name (str): name of the distribution

        """
        super(NormalDistribution, self).__init__(*args)
        self.name = 'Normal'
        if self.hyperparams['contextual_modelling'] == 'strat':
            self.stratificator = Stratificator(self.hyperparams['nb_quantile'])

    def pdf(self, features, x):
        """ Pdf of the normal distribution

        Args:
            features (np.array): observation features
            x (np.array): actions taken to evaluate the pdf on

        """
        return 1/(self.scale * np.sqrt(2*np.pi)) * np.exp(-((x - self.loc)/self.scale)**2/2)

    def entropy(self):
        """ Returns entropy of the distribution
        """
        return 0.5 * np.mean(np.log(2*np.pi*np.exp(0)*self.scale**2))

    def get_samples(self, parameter, features, random_seed=42):
        """ Samples from the distribution

        Args:
            features (np.array): contextual information to sample action from
            random_seed (int)

        """
        size = features.shape[0]
        rng = np.random.RandomState(random_seed)
        return rng.normal(self.loc, self.scale, size)

    def update_parameter(self, parameter, features, reinitialize=False):
        """ Updates the parameter of the normal distribution

        Args:
            parameter (np.array): parameter of the distribution to be minimized
            features (np.array): observation features
            reinitialize (bool): for stratitification to be applied on new features

        """
        if self.hyperparams['contextual_modelling'] == 'unique':
            self.loc, self.scale = parameter[:-1], parameter[-1]

        elif self.hyperparams['contextual_modelling'] == 'linear':
            intercept_coeff, mean_coeff, self.scale = parameter[0], parameter[1:-1], parameter[-1]
            self.loc = np.dot(features, mean_coeff) + intercept_coeff

        elif self.hyperparams['contextual_modelling'] == 'kern-poly2':
            f = np.einsum('ij,ih->ijh', features, features).reshape(features.shape[0], -1)
            intercept_coeff, mean_coeff, self.scale = parameter[0], parameter[1:-1], parameter[-1]
            self.loc = np.dot(f, mean_coeff) + intercept_coeff

        elif self.hyperparams['contextual_modelling'] == 'kern-lin-poly2':
            n = features.shape[1]
            intercept_coeff, mean_coeff_lin, mean_coeff_kern, self.scale = parameter[0], parameter[1:n+1], parameter[n+1:-1], parameter[-1]
            m_linear = np.dot(features, mean_coeff_lin) + intercept_coeff
            f = np.einsum('ij,ih->ijh', features, features).reshape(features.shape[0], -1)
            m_kern = np.dot(f, mean_coeff_kern)
            self.loc = m_kern + m_linear

        elif self.hyperparams['contextual_modelling'] == 'strat':
            m, self.scale = parameter[:-1], parameter[-1]
            if not self.stratificator.initialized or reinitialize:
                self.strat_tables = self.stratificator.get_table_strats(features)
            self.loc = m[self.strat_tables]