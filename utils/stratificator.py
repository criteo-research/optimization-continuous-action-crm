import autograd.numpy as np

class Stratificator:
    """ Stratificator for stratified IPS

    """
    def __init__(self, number_quantiles):
        self.number_quantiles = number_quantiles
        self.initialized = False

    @staticmethod
    def get_strat_by_dimension(feature, quantiles):
        idx = 0
        while not ((quantiles[idx] <= feature) & (feature < quantiles[idx + 1])):
            idx += 1
        return idx

    def set_list_quantiles(self, features):
        self.list_quantiles = list(np.pad(np.array([np.quantile(features, (i + 1) / self.number_quantiles, axis=0) \
                                                    for i in range(self.number_quantiles - 1)]), 1, 'constant', \
                                          constant_values=(-np.inf, np.inf)).T)[1:-1]
        self.initialized = True

    def get_strat(self, feature):
        idx = 0
        for d_ in range(self.dimension):
            idx += self.get_strat_by_dimension(feature[d_], self.list_quantiles[d_]) * self.number_quantiles ** d_
        return idx

    def get_table_strats(self, features):
        self.dimension = features.shape[1]
        if not self.initialized:
            self.set_list_quantiles(features)
        strats_references = np.ones(features.shape[0])
        for idx, feature in enumerate(features):
            strats_references[idx] = self.get_strat(feature)

        return strats_references.astype(int)