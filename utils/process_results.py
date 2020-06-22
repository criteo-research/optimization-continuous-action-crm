import re
import os
import numpy as np
import pandas as pd

def get_regex_pattern(params, metrics):
    regex_pattern = 'dataset:(.*)\|estimator:(.*)'
    if len(params) > 1:
        regex_pattern += '\|' + '\|'.join(
            '{}:(.*)'.format(k) for k, _ in sorted(params.items()) if k != 'nami')
        regex_pattern += '' + '\|'.join('{}:(.*)'.format(k) for k, _ in sorted(metrics.items()))
    return regex_pattern

def get_col_df(params, metrics):
    columns = ['dataset', 'estimator']
    columns += list(sorted(params.keys())) + list(sorted(metrics.keys()))
    # columns.remove('logging_policy')
    return columns

def create_df(fname, regex_pattern, columns):
    rgx = re.compile(regex_pattern, flags=re.M)
    with open(fname, 'r') as f:
        lines = rgx.findall(f.read())
        df = pd.DataFrame(lines, columns=columns)

        columns_str = ['M', 'dataset', 'estimator', 'clip', 'learn_parameter', 'contextual_modelling', 'learning_distribution', 'method', 'proximal', 'tau', 'kernel', 'logging_policy', 'feature_map_kernel', 'initialize_causal_dm']
        columns_flt_int = [item for item in columns if item not in columns_str]
        for col in columns_flt_int:
            try:
                df[col] = df[col].astype(np.int32)
            except ValueError:
                df[col] = df[col].astype(np.float32, errors='ignore')
    return df