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

To run an experiment, run according to the following examples. You can also look at the file `main.py` for detailed commands.

### Synthetic Datasets
Selfnormalized estimator on the Noisymoons dataset with contextual modelling linear and normal distribution for the learned PDF
```
python main.py --estimator selfnormalized --dataset noisymoons --contextual_modelling linear --learning_distribution normal
```

CLP estimator on the anisotropic dataset with variance penalty, gamma value and penalties on the norm of the IPS parameter
```
python main.py --estimator clp --dataset anisotropic --contextual_modelling clp --gamma 100 --reg_param 10
```


### Criteo Continuous Offline Dataset

IPS estimator on the open datasetm with proximal point method with kappa value, soft clipping and clipping M value
```
python main.py --estimator ips --dataset open --proximal --kappa 0.001 --clip soft --M 100
```


### Sanity Checks

#### Synthetic dataset

To check that you have the same setup as us, run the following paired example and verify results on a synthetic dataset:

```
python main.py --estimator selfnormalized --dataset noisycircles --var_lambda 0.1 --reg_entropy 0.0001 --contextual_modelling kern-poly2 --nb_rd 1
```
You should see the result in txt file:
```
bootstrap_h_ips_test:0.5591555398423407|bootstrap_h_ips_valid:0.12503530913739397|bootstrap_h_snips_test:0.078604862769276|bootstrap_h_snips_valid:0.12692125236206286|em_diagnostic_test:2.256203528104498|em_diagnostic_valid:0.3610341103989956|ess_diagnostic_test:0.0004889707635067942|ess_diagnostic_valid:0.0001644024987240737|ips_test:-1.2598683463459917|ips_valid:-0.1314177178172699|snips_test:-0.5577064238301641|snips_valid:-0.43333562660798264|std_h_test:99.59228336700215|std_h_valid:27.56747565157582|t_h_test:1.155835068543718|t_h_valid:0.30464179059068014|test_policy_std:4.2741911926731926e-05|test_reward:0.6005687662484943|test_test_context_std:4.394506103180005e-07|valid_policy_std:4.286021927941692e-05|valid_reward:0.6000213021197933|valid_test_context_std:3.7024934048650937e-07
```

```
python main.py --estimator selfnormalized --dataset noisycircles --var_lambda 0.1 --reg_entropy 0.0001 --contextual_modelling kern-poly2 --nb_rd 1 --proximal --kappa 0.1
```
You should see the result in txt file:
```
bootstrap_h_ips_test:0.7880889894924185|bootstrap_h_ips_valid:0.6401819625960339|bootstrap_h_snips_test:0.09485290624006576|bootstrap_h_snips_valid:0.08442520732469028|em_diagnostic_test:1.3670278866066194|em_diagnostic_valid:2.5076555401253433|ess_diagnostic_test:0.00024129358726949956|ess_diagnostic_valid:0.0003895512213533832|ips_test:-1.1083350877649485|ips_valid:-1.2498919714121797|snips_test:-0.7999308903772422|snips_valid:-0.5051460411109807|std_h_test:83.3809676223928|std_h_valid:122.49863522959008|t_h_test:0.9478270128489855|t_h_valid:1.3562256165938131|test_policy_std:4.3379780649088184e-05|test_reward:0.6156644902498727|test_test_context_std:4.3046225811738055e-07|valid_policy_std:4.346545737493252e-05|valid_reward:0.6179033871580074|valid_test_context_std:3.7837349603543975e-07
```

#### Open dataset

To check that you have the same setup as us, run the following paired example and verify results on the open dataset:

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

