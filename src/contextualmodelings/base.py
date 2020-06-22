import os
import sys
from abc import ABCMeta, abstractmethod
import autograd.numpy as np

base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(base_dir)

dimension_datasets = {
    'criteo': 3,
    'criteo-small': 3,
    'noisycircles': 2,
    'noisymoons': 2,
    'anisotropic': 2,
    'toy-gmm': 2,
    'gmm': 2,
    'varied': 2,
    'warfarin': 81
}

class ContextualModelling:
    """ General Abstract Contextual Modelling class

    """
    __metaclass__ = ABCMeta

    def __init__(self, hyperparams):
        """Initializes the class

        Attributes:
            hyperparams (dict):  dictionary of hyperparams
            rng (np.random.RandomState): random generator

        """
        self.hyperparams = hyperparams
        self.rng = np.random.RandomState(self.hyperparams['random_seed'])


    @abstractmethod
    def get_parameters(self, parameter, features, actions, reinitialize):
        """ Updates the parameters of the distribution

        Args:
            parameter (np.array): parameter of the distribution
            features (np.array): observation features
            actions (np.array): observation actions
            reinitialize (bool): for stratitification to be applied on new features

        """
        pass

    def _prepare_starting_parameter(self, dataset):
        self.d = dimension_datasets[dataset.name]
        m = np.mean(dataset.actions_train)
        std = np.std(dataset.actions_train)
        self.scale = 1e-1
        v = std * self.scale
        return m, v

    @abstractmethod
    def get_starting_parameter(self, dataset):
        """ Creates starting parameter

        Args:
            dataset (dataset)

        """
        pass