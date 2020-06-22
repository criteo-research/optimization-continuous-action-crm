import os
import sys
import autograd.scipy.stats
import autograd.scipy as sp
import autograd.numpy as np
EPS = 1e-8
base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(base_dir)

from src.distributions.base import Distribution

class SoftMax(Distribution):
    """ Soft Max Distribution utilities

    Inherits from the parent class Distribution

    """

    def __init__(self, *args):
        """Initializes the class

        Attributes:
            hyperparams (dict): dictionnary parameters
            name (str): name of the distribution

        """
        super(SoftMax, self).__init__(*args)
        self.name = 'SoftMax'

    @staticmethod
    def indicator(u):
        return np.where(u == 0., 1., 0.)

    def pdf(self, features, x):
        """ Pdf of the SoftMax distribution

        Args:
            features (np.array): observation features
            x (np.array): actions taken to evaluate the pdf on

        """
        # param = parameter.reshape(self.bins.shape[0], fm_c.shape[1])
        exp = np.exp(self.representation)
        softmax_probas = exp / np.sum(exp, axis=1, keepdims=True)

        return softmax_probas[np.arange(features.shape[0]), self.inds]

    def entropy(self):
        """ Returns entropy of the distribution
        """
        return 0.

    def get_samples(self, parameter, features, random_seed=42):
        """ Samples from the distribution

        Args:
            features (np.array): contextual information to sample action from
            random_seed (int)

        """
        itcp, param = parameter[0], parameter[1:]
        fm_c = self.contextual_modelling.feature_map.contextual_feature_map(features)
        shape_bins = self.contextual_modelling.bins.shape[0]
        representation = np.einsum('ij, ik -> ikj', fm_c, np.ones((features.shape[0], shape_bins)))
        param = param.reshape(shape_bins, fm_c.shape[1])
        representation = np.einsum('ikl, kl-> ik', representation, param) + itcp
        exp = np.exp(representation)
        softmax_probas = exp / np.sum(exp, axis=1, keepdims=True)
        rng = np.random.RandomState(random_seed)
        return np.concatenate([rng.choice(self.contextual_modelling.bins, 1, p=softmax_probas[i]) for i in range(features.shape[0])])



    def update_parameter(self, parameter, features, actions=None, reinitialize=False):
        """ Updates the parameters of the log normal distribution

        Args:
            parameter (np.array): parameter of the distribution to be minimized
            features (np.array): observation features
            actions (np.array): observation actions
            reinitialize (bool): for stratitification to be applied on new features

        """
        self.representation, self.inds = self.contextual_modelling.get_parameters(parameter, features, actions, reinitialize)





