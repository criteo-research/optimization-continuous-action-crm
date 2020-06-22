import os
from datetime import date

today = date.today()

def get_task_name(dataset, estimator, params):
    task_name = 'dataset:{}|estimator:{}'.format(dataset.name, estimator.name)
    if len(params) > 1:
        task_name += '|' + '|'.join('{}:{}'.format(k, v) for k, v in sorted(params.items()))
    return task_name

def get_metrics_information(metrics):
    metrics_information = ''
    if len(metrics) > 1:
        metrics_information += '|' + '|'.join('{}:{}'.format(k, v) for k, v in sorted(metrics.items()))
    return metrics_information[1:]

def get_results_file_name(args):
    results_dir = os.path.join('results-{}'.format(today.strftime("%d-%m-%Y")),args.dataset , 'proximal' if args.proximal else 'nonproximal', args.estimator,
                               args.clip, args.contextual_modelling)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    return os.path.join(results_dir, 'metrics.txt')
