import os
import sys
import scipy.stats
import scipy.optimize
import scipy as sp
import autograd.numpy as np

base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(base_dir)
EPS = 1e-7


class Optimizer:
    """ Optimizer class for monitoring the training and experimenting differents methods

    """

    def __init__(self, hyperparams):
        """Initializes the class

        Attributes:
            hyperparams (dict):  dictionary of hyperparams
            rng (np.random.RandomState): random seed state
            kappa_proximal (float): regularization parameter for proximal point subproblems

        """
        self.hyperparams = hyperparams
        self.rng = np.random.RandomState(self.hyperparams['random_seed'])
        self.kappa_proximal = self.hyperparams['kappa']

    def method(self, func, parameter, grad_func, args, callback, hess):
        """ Try different optimization methods

        Args:
            func (python function): function to be minimized
            parameter (np.array): parameter to be learned
            grad_func (python function): gradient of the function to be minimized
            args (tuple): tuple of features, actions, rewards
            callback (python function): called every iterations to perform monitoring
            hess (python function): hessian of the function to be minimized

        Returns:
            (tuple): optimized parameter, value of the function at this point, dictionary of information
        """
        if self.hyperparams['method'] == 'L-BFGS':
            # bounds on all params
            bnds = [(EPS, None) for _ in parameter]
            optimized = sp.optimize.minimize(func, parameter, args, method='L-BFGS-B', jac=grad_func, callback=callback, bounds=bnds)
            d = {'warnflag': not optimized.success,
             'grad': optimized.jac,
             'task': optimized.message,
             'nit': optimized.nit}
            return optimized.x, optimized.fun, d
        elif self.hyperparams['method'] == 'Newton':
            optimized = sp.optimize.minimize(func, parameter, args=args, method='Newton-CG', jac=grad_func, hess=hess, callback=callback)
            d = {'warnflag': not optimized.success,
                 'grad': optimized.jac,
                 'task': optimized.message,
                 'nit': optimized.nit}
            return optimized.x, optimized.fun, d
        else:
            raise NotImplementedError

    def optimize(self, func, parameter, grad_func, args, callback, hess=None):
        """ Checks wether to perform proximal point optimization or not

        """
        print('Optimization starts...')
        if self.hyperparams['proximal']:
            return self._optimize_with_prox(func, parameter, grad_func, args, callback, hess)
        else:
            return self.method(func, parameter, grad_func, args, callback, hess)

    def _optimize_with_prox(self, func, x, grad_func, args, callback, hess):
        """ Proximal point method

        Args:
            func (python function): function to be minimized
            x (np.array): parameter to be learned
            grad_func (python function): gradient of the function to be minimized
            args (tuple): tuple of features, actions, rewards
            callback (python function): called every iterations to perform monitoring
            hess (python function): hessian of the function to be minimized

        """
        iter_ = 0
        x_k = x
        x_k_p_1 = self.rng.normal(size=x.shape[0])

        while iter_ < self.hyperparams['max_iter']:
            optimized = self._subproblem(func, x_k, grad_func, args, callback, hess)
            x, f, d = optimized
            x_k_p_1 = x
            iter_ += 1
            x_k = x_k_p_1

        print("Proximal subproblem optimization ended after {} iterations".format(iter_))
        self.kappa_proximal = 0
        optimized = self._subproblem(func, x_k_p_1, grad_func, args, callback, hess)
        return optimized

    def _subproblem(self, func, x_k, grad_func, args, callback, hess):
        """ Subproblem for the proximal point method

        Args:
            x_k (np.array): k-th parameter of the subproblem to be solved

        Note:
            See method _optimize_with_prox

        """
        features, actions, rewards, pi_logging = args

        def _new_objective(x, features, actions, rewards, pi_logging):
            return func(x, features, actions, rewards, pi_logging) + self.kappa_proximal / 2 * (
                        x - x_k).T @ (x - x_k)

        def _gradient_new_objective(x, features, actions, rewards, pi_logging):
            return grad_func(x, features, actions, rewards, pi_logging) + self.kappa_proximal * (
                        x - x_k)

        def _hessian_new_objective(x, features, actions, rewards, pi_logging):
            return hess(x, features, actions, rewards, pi_logging) + self.kappa_proximal

        return self.method(_new_objective, x_k, _gradient_new_objective, args=(features, actions, rewards, pi_logging),
                             callback=callback, hess=_hessian_new_objective)
