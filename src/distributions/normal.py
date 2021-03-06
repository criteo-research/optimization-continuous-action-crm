import os
import sys
import autograd.scipy.stats
import autograd.scipy as sp
import autograd.numpy as np

base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(base_dir)

from src.distributions.base import Distribution

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
        self.update_parameter(parameter, features, reinitialize=True)
        size = features.shape[0]
        rng = np.random.RandomState(random_seed)
        return rng.normal(self.loc, self.scale, size)

    def update_parameter(self, parameter, features, actions=None, reinitialize=False):
        """ Updates the parameter of the normal distribution

        Args:
            parameter (np.array): parameter of the distribution to be minimized
            features (np.array): observation features
            actions (np.array): observation actions
            reinitialize (bool): for stratitification to be applied on new features

        """
        self.loc, self.scale = self.contextual_modelling.get_parameters(parameter, features, actions, reinitialize)