# Optimization of Continuous Action Policies with Counterfactual Risk Minimization

This code reproduces the experiments of the paper "Optimization Approaches for counter-factual risk minimization with continuous actions". Please cite it using the following Bibtex snippet:

{TBA}

We believe it is also generally useful to experiment and benchmark off-policy (counter-factual) learning of continuous, contextual policies.

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

{Url of Dataset Documentation}

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

To check that you have the same setup as us, run the following examples and verify results on synthetic dataset:

```
$ python main.py --estimator selfnormalized --dataset noisycircles --var_lambda 0.1 --reg_entropy 0.0001 --contextual_modelling kern-poly2 --nb_rd 1 --proximal --kappa 0.1 |grep 'test reward'
test reward 0.615664 policy std 0.000043 context std 0.000000
```

```
$ python main.py --estimator selfnormalized --dataset noisycircles --var_lambda 0.1 --reg_entropy 0.0001 --contextual_modelling kern-poly2 --nb_rd 1 |grep 'test reward'
test reward 0.615664 policy std 0.000043 context std 0.000000
```

#### Criteo dataset

To check that you have the same setup as us, run the following paired example and verify results on the open dataset:

```
$ python main.py --estimator ips --dataset open --var_lambda 0.0 --reg_entropy 0.0001 --clip soft --M 10 --contextual_modelling strat --nb_rd 1 
```

Note: this will take some time. 

You should see now in the result file:
```
$ cat results/L-BFGS/nonproximal/ips/soft/strat/metrics.txt |sed 's/|/\n/g' |grep 'snips_test'
snips_test:-10.183163677795509
```

