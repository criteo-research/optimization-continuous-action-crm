import os
import sys
import autograd.numpy as np

base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(base_dir)

from src.estimator.base import CRMEstimator
EPS = 1e-7

class SelfNormalizedEstimator(CRMEstimator):
    """ Self Normalized Estimator, see Equation (6)
    
    """
    def __init__(self, **kw):
        """Initializes the class
        
        Attributes:
            name (str): name of the estimator

        """
        super(SelfNormalizedEstimator, self).__init__(**kw)
        assert self.hyperparams['clip'] == 'None', "Warning: Clipping not supported in Self Normalized Estimator"
        self.name = 'SelfNormalized'

    def risk(self, parameter, features, actions, rewards, pi_logging, training=None):
        """ See docstring in the parent class
        """
        # impt_smplg_weight = self._get_importance_sampling_weight(parameter, features, actions, pi_logging)
        u_i = - rewards * self.impt_smplg_weight
        return np.sum(u_i)/(np.sum(self.impt_smplg_weight)+EPS)

    def _std_penalty(self, parameter, features, actions, rewards, pi_logging):
        """ See docstring in the parent class
        """
        u_i = (-rewards - self.risk(parameter, features, actions, rewards, pi_logging))**2 * self.impt_smplg_weight**2
        u = np.sum(u_i, axis=0)
        v = np.sum(self.impt_smplg_weight, axis=0)**2
        return u/(v+EPS)