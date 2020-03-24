import autograd.numpy as np
import logging

dimension_datasets = {
    'open': 3,
    'noisycircles': 2,
    'noisymoons': 2,
    'anisotropic': 2,
    'toy-gmm': 2,
}

estimator_dico = {
    'ips': 'InversePropensityScore',
    'selfnormalized': 'SelfNormalizedEstimator',
    'direct': 'DirectEstimator',
    'directstochastic': 'DirectStochastic',
    'doublyrobust': 'DoublyRobust',
    'clp': 'CounterfactualLossPredictor',
    'stochasticdirect': 'StochasticDirect',
    'snclp': 'SelfNormalizedCounterfactualLossPredictor'
}

distribution_dico = {
    'lognormal': 'LogNormalDistribution',
    'normal': 'NormalDistribution'
}

def create_start_parameter(hyperparams, dataset):
    """ Creates starting parameter

    Args:
        hyperparams (dic): dictionary of hyperparameters
        dataset (dataset)

    """
    # Reset random seed generator to ensure same initialization
    rng = np.random.RandomState(hyperparams['random_seed'])
    d = dimension_datasets[dataset.name]
    m = np.mean(dataset.actions_train)
    std = np.std(dataset.actions_train)
    scale = 1e-1
    v = std*scale
    if hyperparams['contextual_modelling'] == 'unique':
        start_parameter = np.abs(np.array([rng.normal(m, v), rng.normal(scale=scale)]))
    elif hyperparams['contextual_modelling'] == 'linear':
        start_parameter = np.abs(np.concatenate([rng.normal(m, v, size=1), rng.normal(scale=scale, size=d + 1)]))
    elif hyperparams['contextual_modelling'] == 'kern-poly2':
        start_parameter = np.abs(np.concatenate([rng.normal(m, v, size=1), rng.normal(scale=scale, size=d ** 2 + 1)]))
    elif hyperparams['contextual_modelling'] == 'kern-lin-poly2':
        start_parameter = np.abs(np.concatenate([rng.normal(m, v, size=1), rng.normal(scale=scale, size=d ** 2 + d + 1)]))
    elif hyperparams['contextual_modelling'] == 'clp':
        start_parameter = np.concatenate([rng.normal(m, v, size=1), rng.normal(scale=scale, size=(hyperparams['nb_quantile']+2)* d)])
    else:
        start_parameter = np.abs(rng.normal(m, v, size=hyperparams['nb_quantile'] ** d + 1))
    return start_parameter

def get_estimator_by_name(name, hyperparams, start_parameter, logger):
    """ Gets estimator  according to hyperparameter choice

    Args:
        hyperparams (dic): dictionary of hyperparameters
        start_parameter (np.array)
        logger (logging): gets information for debugging

    """
    estimator_name = "src.estimator.{}".format(name)
    mod = __import__(estimator_name, fromlist=[estimator_dico[name]])
    return getattr(mod, estimator_dico[name])(hyperparams=hyperparams, verbose=logger, init_parameter=start_parameter)

def get_distribution_by_params(hyperparams):
    """ Gets distribution according to hyperparameter choice

    Args:
        hyperparams (dic): dictionary of hyperparameters

    """
    name = hyperparams['learning_distribution']
    distribution_name = "src.distributions.{}".format(name)
    mod = __import__(distribution_name, fromlist=[distribution_dico[name]])
    return getattr(mod, distribution_dico[name])(hyperparams)

def get_logger():
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter('[%(asctime)s %(levelname)-3s @%(name)s] %(message)s', datefmt='%H:%M:%S'))
    logging.basicConfig(level=logging.DEBUG, handlers=[console])
    return logging.getLogger("Test")
