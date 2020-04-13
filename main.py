import argparse
from utils.save_results import get_task_name, get_metrics_information, get_results_file_name
from utils.prepare import create_start_parameter, get_estimator_by_name, get_logger
from utils.dataset import get_dataset_by_name

# Logging
logger = get_logger()

TOL = 1e-5
MAX_ITER_PROX = 50

def process_experiment(args, random_seed):

    # Data, modelling and hyperparameters setup
    dataset = get_dataset_by_name(args.dataset, random_seed)
    hyperparams = {
        'var_lambda': args.var_lambda,
        'clip': args.clip,
        'M': args.M,
        'proximal': args.proximal,
        'kappa': args.kappa,
        'tol': TOL,
        'max_iter': args.max_iter_prox,
        'random_seed': random_seed,
        'contextual_modelling': args.contextual_modelling,
        'method': args.method,
        'nb_quantile': args.nb_quantile,
        'reg_param': args.reg_param,
        'reg_entropy': args.reg_entropy,
        'barrier_lambda': args.barrier_lambda,
        'learning_distribution': args.learning_distribution,
        'feature_map_kernel': args.feature_map_kernel,
        'kernel_bandwidth': args.kernel_bandwidth,
        'reg_param_direct': args.reg_param_direct,
        'gamma': args.gamma,
        'initialize_clp': args.initialize_clp
    }
    start_parameter = create_start_parameter(hyperparams, dataset)
    data_train, data_valid, data_test = dataset.get_data()

    # Estimator setup
    estimator = get_estimator_by_name(args.estimator, hyperparams, start_parameter, logger)
    estimator.debug = args.debug

    # Perform experiment
    logger.info("Running experiments with {} estimator on the {} dataset".format(args.estimator, args.dataset))
    estimator.fit(data_train)

    # Save results
    task_name = get_task_name(dataset, estimator, hyperparams)
    metrics = estimator.evaluate(dataset, data_train, data_valid, data_test, args.n_samples)
    metrics_information = get_metrics_information(metrics)

    if args.plot:
        estimator.plot_samples(data_test, dataset.name)

    return '{} {}\n'.format(task_name, metrics_information)



def run(args):
    
    has_effect = False

    if args:
        if args.debug:
            for rd in range(args.nb_rd):
                result = process_experiment(args, rd)
        else:
            try:
                fname = get_results_file_name(args)
                with open(fname, 'a') as file:
                    for rd in range(args.nb_rd):
                        result = process_experiment(args, rd)
                        file.write(result)

            except Exception as e:
                logger.exception(e)
                logger.error("Uhoh, the script halted with an error.")
    else:
        if not has_effect:
            logger.error(
                "Script halted without any effect. To run code, use command:\npython3 main.py <args>")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run scripts for the evaluation of methods')
    parser.add_argument('--estimator', nargs="?",  default='ips',  
                        choices=['ips', 'selfnormalized', 'direct', 'doublyrobust', 'directstochastic', 'clp',
                                 'stochasticdirect', 'snclp'],
                        help='estimator name')
    parser.add_argument('--dataset', nargs="?", default='noisymoons',  
                        choices=['criteo', 'noisycircles', 'noisymoons', 'anisotropic', 'criteo-small'],
                        help='dataset')
    parser.add_argument('--method', nargs="?", default='L-BFGS',  choices=['Newton', 'L-BFGS'],
                        help='optimisation method')
    parser.add_argument('--var_lambda', nargs="?", type=float, default=0.01, help='variance penalty')
    parser.add_argument('--clip', nargs="?", default='None',  choices=['None', 'classic', 'soft'],
                        help='choose how to clip')
    parser.add_argument('--M', nargs="?", type=str, default='None', help='clipping term')
    parser.add_argument('--proximal', action='store_true', help='use proximal method')
    parser.add_argument('--kappa', nargs="?", type=float, default=0., help='kappa for proximal point method')
    parser.add_argument('--contextual_modelling', nargs="?", default='linear',  
                        choices=['unique', 'strat', 'linear', 'kern-poly2', 'kern-lin-poly2', 'clp'],
                        help='how to learn parameter alpha')
    parser.add_argument('--nb_rd', nargs="?", type=int, default=5, help='number of random seeds')
    parser.add_argument('--nb_quantile', nargs="?", type=int, default=4, help='number of quantiles for stratification')
    parser.add_argument('--plot', action='store_true', help='plot actions samples')
    parser.add_argument('--n_samples', nargs="?", type=int, default=100, help='evaluation')
    parser.add_argument('--reg_param', nargs="?", type=float, default=0, help='parameter regularization')
    parser.add_argument('--reg_entropy', nargs="?", type=float, default=0.01, help='entropy regularization')
    parser.add_argument('--barrier_lambda', nargs="?", type=float, default=0, help='strength of barrier on the mean')
    parser.add_argument('--learning_distribution', nargs="?", type=str, default='lognormal', choices=['lognormal', 'normal'], help='distribution of the learned policy')
    parser.add_argument('--feature_map_kernel', nargs="?", type=str, default='gaussian', choices=['gaussian', 'indicator'], help='kernel for the feature map of the direct method')
    parser.add_argument('--reg_param_direct', nargs="?", type=float, default=0, help='parameter regularization')
    parser.add_argument('--kernel_bandwidth', nargs="?", type=float, default=1., help='bandwidth parameter for kernel regression')
    parser.add_argument('--gamma', nargs="?", type=float, default=1., help='bandwidth parameter for kernel regression')
    parser.add_argument('--initialize_clp', action='store_true', help='initialize CLP with DM')
    parser.add_argument('--max_iter_prox', nargs="?", type=int, default=MAX_ITER_PROX, help='max iteration for proximal')
    parser.add_argument('--debug', action='store_true', help='print debugging info')

    run(parser.parse_args())

