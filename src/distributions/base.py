import os
import sys
from abc import ABCMeta, abstractmethod

base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(base_dir)

class Distribution:
    """ General Abstract Distribution class

    """
    __metaclass__ = ABCMeta

    def __init__(self, hyperparams, contextual_modelling, verbose=None):
        """Initializes the class

        Attributes:
            hyperparams (dict):  dictionary of hyperparams
            logger (logger): display log messages
            contextual_modelling (ContextualModelling): modelling for features

        """
        self.hyperparams = hyperparams
        self.logger = verbose
        self.contextual_modelling = contextual_modelling

    @abstractmethod
    def pdf(self, features, x):
        """ Pdf of the distribution

        Args:
            features (np.array): observation features
            x (np.array): evaluate the pdf on x

        """
        pass

    @abstractmethod
    def update_parameter(self, parameter, features, actions, reinitialize):
        """ Updates the parameters of the distribution

        Args:
            parameter (np.array): parameter of the distribution
            features (np.array): observation features
            actions (np.array): observation actions
            reinitialize (bool): for stratitification to be applied on new features

        """
        pass

    @abstractmethod
    def entropy(self):
        """ Returns entropy of the distribution
        """
        pass