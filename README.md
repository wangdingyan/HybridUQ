# A Hybrid Framework For Uncertainty Quantification in Deep Learning-Based QSAR Regression Modeling

---
## Requirements

* python == 3.7.0
* numpy
* pandas
* scipy
* scikit-learn
* pytorch
* tqdm
* tap (pip install typed-argument-parser)
* rdkit

Building of D-MPNN models are based on chemprop package (https://github.com/chemprop/chemprop).

### Datasets
All datasets are stored in `~/data/dataset`\
Splitting indexes for three different splitting strategies are stored in `~/data/ivit`, `~/data/ivot` and `~/data/ovot`

### Usage

To produce 5-fold cross-validation results on 24 datasets using the D-MPNN model, please run:
```
$ python script/produce_results.py SPLITTING_TYPE
```
For example:
```
$ python script/produce_results.py ivit
```
Here `SPLITTING_TYPE` should be one of `ivit`,`ivot` and `ovot`

All results will be stored in `~/results`

