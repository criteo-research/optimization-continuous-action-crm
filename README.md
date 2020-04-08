# Optimization of Continuous Action Policies with Counterfactual Risk Minimization

This code reproduces the experiments of the paper "Optimization Approaches for counterfactual risk minimization with continuous actions". Please cite it using the following bibtex snippet:

{TBA}

We believe it is also generally useful to experiment and benchmark counterfactual learning of continuous, contextual policies.

## Setup

This code uses the [Cyanure optimization toolkit](http://thoth.inrialpes.fr/people/mairal/cyanure/welcome.html). It is recommended to first install the MKL library through `conda` before proceeding to installing other requirements. 
```
conda install mkl
``` 
And then
```
pip install -r requirements.txt
```

## Datasets

This code includes synthetic datasets as well as a real-life, large-scale dataset donated by [Criteo AI Lab](https://ailab.criteo.com/). The latter is free to use for research purposes and will be downloaded automatically the first time it is used in the code.

## Experiments

To run an experiment, run according to the following examples:

IPS estimator on the open datasetm with proximal point method with kappa value, soft clipping and clipping M value
```
python main.py --estimator ips --dataset open --proximal --kappa 0.001 --clip soft --M 100
```

Selfnormalized estimator on the Noisymoons dataset with contextual modelling linear and normal distribution for the learned PDF
```
python main.py --estimator selfnormalized --dataset noisymoons --contextual_modelling linear --learning_distribution normal
```

CLP estimator on the anisotropic dataset with variance penalty, gamma value and penalties on the norm of the IPS parameter 
```
python main.py --estimator clp --dataset anisotropic --contextual_modelling clp --gamma 100 --reg_param 10
```

See the file main.py for detailed commands.

## To check that you have the same setup as us, run the following paired example and verify results:

```
python main.py --estimator ips --dataset open --var_lambda 0.0 --reg_entropy 0.0001 --clip soft --M 10 --contextual_modelling strat --nb_rd 1 
```
You should see the result in txt file:
```
bootstrap_h_ips_test:0.3215048166326543|bootstrap_h_ips_valid:0.20751913655716103|bootstrap_h_snips_test:0.44773391191823303|bootstrap_h_snips_valid:0.28910924541024274|em_diagnostic_test:0.862643695851652|em_diagnostic_valid:0.8466896469605204|ess_diagnostic_test:6.778598640879997e-05|ess_diagnostic_valid:0.00015364638187059036|ips_test:-8.778060541478553|ips_valid:-8.812978737208667|snips_test:-10.183163677795509|snips_valid:-10.410955647868903|std_h_test:139.61806948994678|std_h_valid:72.6275369109096|t_h_test:1.5662200830195232|t_h_valid:1.2662114104278828
```

```
python main.py --estimator ips --dataset open --var_lambda 0.0 --reg_entropy 0.0001 --clip soft --M 10 --contextual_modelling strat --nb_rd 1 --proximal --kappa 0.001 --max_iter 10
```
You should see the result in txt file:
```
bootstrap_h_ips_test:0.03591404493588903|bootstrap_h_ips_valid:0.040243139615426804|bootstrap_h_snips_test:0.03568534968078637|bootstrap_h_snips_valid:0.04017202700905228|em_diagnostic_test:0.9997871218339688|em_diagnostic_valid:1.0008526683202326|ess_diagnostic_test:0.1915967745959949|ess_diagnostic_valid:0.1915552329296553|ips_test:-11.398008015075185|ips_valid:-11.501523225252736|snips_test:-11.400434975838014|snips_valid:-11.49172520182447|std_h_test:2.053654853338936|std_h_valid:2.056119342900212|t_h_test:0.07897104954668842|t_h_valid:0.08045501033559577
```

