import os
import sys
import autograd.scipy.stats
import autograd.scipy as sp
import autograd.numpy as np
EPS = 1e-8
base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(base_dir)

from src.distributions.base import Distribution

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

    def pdf(self, features, x):
        """ Pdf of the lognormal distribution

        Args:
            features (np.array): observation features
            x (np.array): actions taken to evaluate the pdf on

        """
        return np.exp(-(np.log(x) - self.mu) ** 2 / (2 * self.sigma ** 2)) / (x * self.sigma * np.sqrt(2 * np.pi))

    def entropy(self):
        """ Returns entropy of the distribution
        """
        return np.mean(np.log(self.sigma*np.exp(self.mu+1/2)*np.sqrt(2*np.pi)))

    def get_samples(self, parameter, features, random_seed=42):
        """ Samples from the distribution

        Args:
            features (np.array): contextual information to sample action from
            random_seed (int)

        """
        self.update_parameter(parameter, features, reinitialize=True)
        size = features.shape[0]
        rng = np.random.RandomState(random_seed)
        return rng.lognormal(self.mu, self.sigma, size)

    def update_parameter(self, parameter, features, actions=None, reinitialize=False):
        """ Updates the parameters of the log normal distribution

        Args:
            parameter (np.array): parameter of the distribution to be minimized
            features (np.array): observation features
            actions (np.array): observation actions
            reinitialize (bool): for stratitification to be applied on new features

        """
        m, v = self.contextual_modelling.get_parameters(parameter, features, actions, reinitialize)
        m += EPS
        self.sigma = np.sqrt(np.log(v / m ** 2 + 1))
        self.mu = np.log(m) - self.sigma ** 2 / 2

