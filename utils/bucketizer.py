import autograd.numpy as np

class Bucketizer:
    """ Bucketizer for stratified IPS

    """
    def __init__(self, K, bucketize_mode):
        self.K = K
        self.initialized = False
        self.eps = 1e-8
        self.mode = bucketize_mode
        self.stride = 0.1

    def action_feature_map_size(self, dataset, mode):
        return int(np.max(dataset.actions_train) / self.stride) if mode=='grid' else self.K

    def get_anchor_points(self, actions):
        """ Builds anchor action point sets for the direct estimator

        Args:
            actions (np.array): actions drawn from the logging policy to build quantiles on
        """
        if self.mode == 'quantile':
            self.quantiles = np.quantile(actions, np.linspace(0, 1, self.K+1))
            self.action_set = np.pad(self.quantiles, 1, 'constant', constant_values=(self.eps, np.inf))
        elif self.mode == 'grid':
            self.action_set = np.arange(self.eps, np.max(actions) + self.stride, self.stride)
            self.K = int(np.max(actions)/self.stride)
        self.initialized = True
        return self.action_set

    def bucketize_actions(self, a):
        """ Gets the closets anchor point of an action

        Args:
            a (np.array): actions for which to find closets anchor points, i.e actions to bucketize
        """
        actions = a.copy()
        for k in range(self.K + 2):
            inf, sup = self.action_set[k], self.action_set[k + 1]
            mask = (actions > inf) & (actions < sup)
            actions[mask] = inf
        return actions