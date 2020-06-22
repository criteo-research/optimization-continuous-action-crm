import os
import sys
import autograd.scipy.stats
import autograd.scipy as sp
import autograd.numpy as np

base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(base_dir)

from src.contextualmodelings.base import ContextualModelling

class Linear(ContextualModelling):
    """ Linear utilities

    Inherits from the parent class ContextualModelling

    """

    def __init__(self, *args):
        """Initializes the class

        Attributes:
            hyperparams (dict): dictionnary parameters
            name (str): name of the distribution

        """
        super(Linear, self).__init__(*args)
        self.name = 'linear'

    def get_parameters(self, parameter, features, actions, reinitialize):
        """ Updates the parameters of the distribution

        Args:
            parameter (np.array): parameter of the distribution
            features (np.array): observation features
            actions (np.array): observation actions
            reinitialize (bool): for stratitification to be applied on new features

        """
        intercept_coeff, mean_coeff, var = parameter[0], parameter[1:-1], parameter[-1]
        mean = np.dot(features, mean_coeff) + intercept_coeff
        return mean, var

    def get_starting_parameter(self, dataset):
        """ Creates starting parameter

        Args:
            dataset (dataset)

        """
        m, v = self._prepare_starting_parameter(dataset)
        return np.abs(np.concatenate([self.rng.normal(m, v, size=1), self.rng.normal(scale=self.scale,
                                                                                     size=self.d + 1)]))