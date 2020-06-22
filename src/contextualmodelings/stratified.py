
import os
import sys
import autograd.scipy.stats
import autograd.scipy as sp
import autograd.numpy as np

base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(base_dir)

from src.contextualmodelings.base import ContextualModelling
from utils.stratificator import Stratificator

class Stratified(ContextualModelling):
    """ Piece-wise constant (stratified) contextual modelling utilities

    Inherits from the parent class ContextualModelling

    """

    def __init__(self, *args):
        """Initializes the class

        Attributes:
            hyperparams (dict): dictionnary parameters
            name (str): name of the distribution
            stratificator (Stratificator): for stratifiying feature context space

        """
        super(Stratified, self).__init__(*args)
        self.name = 'stratified'
        self.stratificator = Stratificator(self.hyperparams['nb_quantile'])

    def get_parameters(self, parameter, features, actions, reinitialize):
        """ Updates the parameters of the distribution

        Args:
            parameter (np.array): parameter of the distribution
            features (np.array): observation features
            actions (np.array): observation actions
            reinitialize (bool): for stratitification to be applied on new features

        """
        m, v = parameter[:-1], parameter[-1]
        if not self.stratificator.initialized or reinitialize:
            self.strat_tables = self.stratificator.get_table_strats(features)
        m_strat = m[self.strat_tables]
        return m_strat, v

    def get_starting_parameter(self, dataset):
        """ Creates starting parameter

        Args:
            dataset (dataset)

        """
        m, v = self._prepare_starting_parameter(dataset)
        return np.abs(self.rng.normal(m, v, size=self.hyperparams['nb_quantile'] ** self.d + 1))