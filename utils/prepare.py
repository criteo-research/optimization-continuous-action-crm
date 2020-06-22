import autograd.numpy as np
import logging


estimator_dico = {
    'ips': 'InversePropensityScore',
    'snips': 'SelfNormalizedEstimator',
    'dm': 'DirectMethod',
    'dr': 'DoublyRobust',
    'sdm': 'StochasticDirect',
}

contextual_dico = {
    'constant': 'Constant',
    'stratified': 'Stratified',
    'linear': 'Linear',
    'kern-poly2': 'KernPoly2',
    'clp': 'CounterfactualLossPredictor',
    'softmax': 'SoftMax'
}

distribution_dico = {
    'lognormal': 'LogNormalDistribution',
    'normal': 'NormalDistribution',
    'softmax': 'SoftMax'
}

def get_estimator_by_name(name, hyperparams, contextual_modelling, logger):
    """ Gets estimator  according to hyperparameter choice

    Args:
        hyperparams (dic): dictionary of hyperparameters
        start_parameter (np.array)
        logger (logging): gets information for debugging

    """
    estimator_name = "src.estimator.{}".format(name)
    mod = __import__(estimator_name, fromlist=[estimator_dico[name]])
    return getattr(mod, estimator_dico[name])(hyperparams=hyperparams, verbose=logger,
                                              contextual_modelling=contextual_modelling)

def get_distribution_by_params(hyperparams, contextual_modelling):
    """ Gets distribution according to hyperparameter choice

    Args:
        hyperparams (dic): dictionary of hyperparameters

    """
    name = hyperparams['learning_distribution']
    distribution_name = "src.distributions.{}".format(name)
    mod = __import__(distribution_name, fromlist=[distribution_dico[name]])
    return getattr(mod, distribution_dico[name])(hyperparams, contextual_modelling)

def get_contextual_modelling_by_params(hyperparams):
    """ Gets distribution according to hyperparameter choice

    Args:
        hyperparams (dic): dictionary of hyperparameters

    """
    name = hyperparams['contextual_modelling']
    contextual_name = "src.contextualmodelings.{}".format(name)
    mod = __import__(contextual_name, fromlist=[contextual_dico[name]])
    return getattr(mod, contextual_dico[name])(hyperparams)

def get_logger():
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter('[%(asctime)s %(levelname)-3s @%(name)s] %(message)s', datefmt='%H:%M:%S'))
    logging.basicConfig(level=logging.DEBUG, handlers=[console])
    return logging.getLogger("Test")



choices_kernels = {'gaussian': 'GaussianKernel'}


def get_kernel_by_name(name):
    kernel_name = "src.kernels.{}".format(name)
    mod = __import__(kernel_name, fromlist=[choices_kernels[name]])
    try:
        return getattr(mod, choices_kernels[name])
    except KeyError:
        raise KeyError(f"Unkown kernel, please specify a key in {choices_kernels.keys()}")


choices_feature_maps = {'nystrom': 'Nystrom',
                        'poly2': 'Poly2',
                        'linear': 'Linear'}

def get_feature_map_by_name(hyperparams):
    feature_map_name = hyperparams['contextual_feature_map']
    contextual_feature_map_name = "src.kernels.featuremaps.{}".format(feature_map_name)
    mod = __import__(contextual_feature_map_name, fromlist=[choices_feature_maps[feature_map_name]])
    try:
        return getattr(mod, choices_feature_maps[feature_map_name])(hyperparams)
    except KeyError:
        raise KeyError(f"Unkown feature map, please specify a key in {choices_feature_maps.keys()}")

