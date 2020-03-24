Add the open-data npy file in the data folder

To install the needed libraries, run

```
pip install -r requirements.txt
```

To run an experiment, run according to the following examples:

IPS estimator on the open datasetm with proximal point method with kappa value, soft clipping and clipping M value
```
python main.py --estimator ips --dataset open --proximal --kappa 0.001 --clipping soft --M 100
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
