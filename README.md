# Counterfactual Learning of Continuous Stochastic Policies

This code reproduces the experiments of the paper [Optimization Approaches for counter-factual risk minimization with continuous actions](https://arxiv.org/abs/2004.11722). Please cite it using the following Bibtex snippet:

```
@misc{zenati2020optimization,
    title={Optimization Approaches for Counterfactual Risk Minimization with Continuous Actions},
    author={Houssam Zenati and Alberto Bietti and Matthieu Martin and Eustache Diemert and Julien Mairal},
    year={2020},
    eprint={2004.11722},
    archivePrefix={arXiv},
    primaryClass={stat.ML}
}
```

We believe it is also generally useful to experiment and benchmark off-policy (counter-factual) learning of continuous, contextual policies.

## Setup

This code uses the [Cyanure optimization toolkit](http://thoth.inrialpes.fr/people/mairal/cyanure/welcome.html). It is recommended to first install the MKL library through `conda` before proceeding to installing other requirements. 
```
$ conda install mkl numpy
```
And then
```
$ pip install -r requirements.txt
```

## Datasets

This code includes synthetic datasets as well as a real-life, large-scale dataset donated by [Criteo AI Lab](https://ailab.criteo.com/). The latter is free to use for research purposes and will be downloaded automatically the first time it is used in the code.

Details on the dataset can be found in the paper. You can also download it directly from [here](https://criteostorage.blob.core.windows.net/criteo-research-datasets/criteo-continuous-offline-dataset.csv.gz) (2.3GB zipped CSV). 

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
$ python main.py --estimator snips --dataset noisycircles --var_lambda 1 --reg_entropy 0.0001 --contextual_modelling kern-poly2 --nb_rd 1 --proximal --kappa 0.1 --max_iter 10|grep 'test reward'
test reward 0.616125 policy std 0.000043 context std 0.000000
```

```
$ python main.py --estimator snips --dataset noisycircles --var_lambda 1 --reg_entropy 0.0001 --contextual_modelling kern-poly2 --nb_rd 1|grep 'test reward'
test reward 0.614141 policy std 0.000059 context std 0.000001
```

#### Criteo dataset

To check that you have the same setup as us, run the following paired example and verify results on the open dataset:

```
$ python main.py --estimator ips --dataset criteo-small --var_lambda 0.001 --reg_entropy 0.00 --clip soft --M 10 --contextual_modelling strat --nb_rd 1
```

Note: this will take some time.

You should see now in the result file:
```
$ cat results/L-BFGS/nonproximal/ips/soft/strat/metrics.txt |sed 's/|/\n/g' |grep 'snips_test'
snips_test:-11.279917272089614
```

