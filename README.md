# Line Graph Vietoris-Rips Persistence Diagram
This repository holds a pytorch version of the code for the paper: Line Graph Vietoris-Rips Persistence Diagram for Topological Graph Representation Learning.

## Prerequisites

### Code
python 3.8 \
torch 1.9.0 \
torchph 0.1.1 \
numpy 1.20.1 \
pandas 1.1.4 \
matplotlib 3.4.1 \
scikit-learn 0.24.2 \
networkx 2.0 \
tqdm 4.53.0

### Data
Before running the code, the data should be downloaded from the following Dropbox link.
```
https://www.dropbox.com/s/dajg409tgs607gf/data.zip?dl=1
```

## Train Models
### Graph Classification

To run 10-fold cross-validation for graph classification tasks, run the run_all_10fold_{model you want}.py with the following command:
```
python main_scripts/run_all_10fold_{model you want}.py 
```
or you may use run_10fold_gc.sh with the following command:
```
bash sh/run_10fold_gc.sh
```

### Graph Regression

To run QM9 experiment for graph regression tasks, run the main_qm9_experiment.py with the following command:
```
python main_scripts/main_qm9_experiment.py --config=configs/qm9_config_{model you want}.json
```
or you may use run_qm9.sh with the following command:
```
bash sh/run_qm9.sh
```

## Acknowledgements

The code is based on:

* Provably Powerful Graph Networks
  (https://github.com/hadarser/ProvablyPowerfulGraphNetworks_torch)
* How Powerful are Graph Neural Networks?
  (https://github.com/weihua916/powerful-gnns)

We sincerely thank them for their contributions.
