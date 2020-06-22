import os
import sys
import scipy as sp
import autograd.numpy as np

base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(base_dir)

from src.estimator.base import CRMEstimator

class InversePropensityScore(CRMEstimator):
    """ Inverse Propensity Score Estimator
    
    """
    def __init__(self, **kw):
        """Initializes the class

        Attributes:
            name (str): name of the estimator

        """
        super(InversePropensityScore, self).__init__(**kw)
        self.name = 'IPS'

    def risk(self, parameter, features, actions, rewards, pi_logging, training=False):
        """ See docstring in the parent class
        """
        u_i = - rewards * self._clip_or_not(self.impt_smplg_weight, is_training=training)
        return np.mean(u_i)

    def _std_penalty(self, parameter, features, actions, rewards, pi_logging):
        """ See docstring in the parent class
        """
        return np.std(rewards*self.impt_smplg_weight)

    def _update_clipping_parameter(self, impt_smplg_weight):
        """ Updates the clipping parameter for Bottou's heuristic

        Args:
            impt_smplg_weight (np.array): importance sampling weights
        """
        if self.hyperparams['M'] == 'auto':
            sorted_copy = np.sort(impt_smplg_weight.copy())
            self.hyperparams['M'] = sorted_copy[-5]
        elif self.hyperparams['M'] == 'None':
            pass
        else:
            self.hyperparams['M'] = float(self.hyperparams['M'])

    def _clip_or_not(self, value, is_training=False):
        """ According to hyperparameters, performs clipping or not

        Args:
            value (np.array): value to be clipped
            is_training (bool): perform clipping or not during for testing [training?]

        """
        self._update_clipping_parameter(value)
        if self.hyperparams['clip'] == 'classic' and is_training:
            return np.minimum(self.hyperparams['M'], value)
        elif self.hyperparams['clip'] == 'soft' and is_training:
            a = (np.exp(sp.special.lambertw(self.hyperparams['M'])) - self.hyperparams['M']).real
            b = a + self.hyperparams['M']
            F = b * np.log(value + a)
            condition = value <= self.hyperparams['M']
            result=np.where(condition, value, F)
            return result
        else:
            return value
