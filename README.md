# Line Graph Vietoris-Rips Persistence Diagram

This repository is the official implementation of the paper: Line Graph Vietoris-Rips Persistence Diagram for Topological Graph Representation Learning. 

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
easydict 1.1 \
tqdm 4.53.0

### Data
Before running the code, the data should be downloaded from the following Dropbox link.
```
https://www.dropbox.com/s/dajg409tgs607gf/data.zip?dl=1
```

## Training

To train the model(s) in the paper, run the following commands:

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

## Results

Our model achieves the following performance on :

### Graph Classification Results 
Since there is no separate test set for graph classification datasets, we follow the standard 10-fold cross-validation and report the following performance (meaning a large value indicates a better model):
$$\max_{i={1,\dots, t}} \frac{1}{10} \cdot \sum_{k=1}^{10} \text{(k-fold validation accuracy at i-th epoch)}$$


#### GIN Type
| **Model / Dataset** | **MUTAG**   | **PTC**     | **PROTEINS**  | **NCI1**   | **NCI109** | **IMDB-B** | **IMDB-M** |
|---------------------|-------------|-------------|---------------|------------|------------|-----------|------------|
| GIN                 | 78.89       | 59.71       | 68.11         | 70.19      | 69.34      | 73.8      | 43.8       |
| GFL                 | 86.11       | 60.0        | 73.06         | 71.14      | 70.53      | 68.7      | 44.67      |
| GFL+                | 79.44       | 59.11       | 69.55         | 71.16      | 70.12      | 71.9      | 44.33      |
| ***GIN-LGVR(1)***   | **86.67**   | 60.29       | **73.42**     | 69.22      | 69.39      | 72.5      | 45.47      |
| ***GIN-LGVR+(1)***  | 85.0        | **61.76**   | 69.55         | **71.61**  | **70.68**  | **74.0**  | **45.8**   |

#### PPGN Type
| **Model / Dataset**  | **MUTAG**   | **PTC**    | **PROTEINS** | **NCI1**    | **NCI109**  | **IMDB-B**  | **IMDB-M** |
|----------------------|-------------|------------|--------------|-------------|-------------|-------------|------------|
| PPGN                 | 88.88       | 64.7       | 76.39        | 81.21       | 81.77       | 72.2        | 44.73      |
| ***PPGN-LGVR+(1)***  | **91.11**   | **66.47**  | **76.76**    | **83.04**   | **81.88**   | **73.5**    | **51.0**   |

#### Auxiliary Results for GIN Type (with other performance metric)
To compare models in a different perspective, we further measure performances of GIN type models based on another commonly used metric (meaning a large value indicates a better model): $$\frac{1}{10} \cdot \sum_{k=1}^{10} (\max_{i={1,\dots, t}} \text{(k-fold validation accuracy at i-th epoch)})$$

| **Model / Dataset** | **MUTAG**  | **PTC**     | **PROTEINS** | **NCI1**   | **NCI109** | **IMDB-B** | **IMDB-M** |
|---------------------|------------|-------------|--------------|------------|------------|------------|------------|
| GIN                 | 88.33      | 70.29       | 74.5         | 72.31      | 71.97      | 77.3       | 50.0       |
| GFL                 | 92.22      | 71.47       | 76.84        | 73.23      | 72.88      | 75.2       | 49.6       |
| GFL+                | 91.11      | 69.41       | 73.96        | 73.28      | 72.62      | 77.4       | 51.0       |
| ***GIN-LGVR(1)***   | **92.78**  | 70.29       | **77.3**     | 72.34      | 71.77      | 76.7       | 50.07      |
| ***GIN-LGVR+(1)***  | 91.11      | **72.64**   | 74.68        | **73.94**  | **73.45**  | **77.8**   | **52.0**   |



### Graph Regression Results (QM9 with 12 tasks)
Since the QM9 dataset has a separate test set, we report the mean absolute error for 12 tasks as performance on the test set (meaning a smaller value indicates a better model).

#### GIN Type
| **Target / Model** | GIN     | GFL     | GFL+         | ***GIN-LGVR(1)***   | ***GIN-LGVR+(1)***  |
|--------------------|---------|---------|--------------|---------------------|---------------------|
| mu                 | 0.729   | 0.917   | **0.655**    | 1.058               | 0.661               |
| alpha              | 3.435   | 4.818   | 2.985        | 3.542               | **2.76**            |
| epsilon_homo       | 0.00628 | 0.01595 | **0.00581**  | 0.01528             | 0.00683             |
| epsilon_lumo       | 0.00957 | 0.02356 | 0.00987      | 0.02952             | **0.00911**         |
| Delta_epsilon      | 0.01023 | 0.01737 | 0.01003      | 0.03253             | **0.00931**         |
| R^2                | 124.05  | 175.23  | 121.33       | 174.96              | **113.1**           |
| ZPVE               | 0.00719 | 0.02354 | 0.00393      | 0.00534             | **0.0026**          |
| U_0                | 17.477  | 18.938  | 18.121       | 17.458              | **15.705**          |
| U                  | 17.477  | 18.881  | 18.12        | 17.807              | **15.706**          |
| H                  | 17.476  | 18.923  | 18.118       | 17.626              | **15.705**          |
| G                  | 17.477  | 18.889  | 18.121       | 17.72               | **15.708**          |
| C_v                | 1.361   | 3.515   | 1.258        | 1.993               | **1.163**           |

#### PPGN Type
| **Target / Model** | PPGN    | ***PPGN-LGVR+(1)*** |
|--------------------|---------|---------------------|
| mu                 | 0.231   | **0.09**            |
| alpha              | 0.382   | **0.19**            |
| epsilon_homo       | 0.00276 | **0.00178**         |
| epsilon_lumo       | 0.00287 | **0.0019**          |
| Delta_epsilon      | 0.00406 | **0.00253**         |
| R^2                | 16.07   | **3.47**            |
| ZPVE               | 0.00064 | **0.00032**         |
| U_0                | 0.234   | **0.216**           |
| U                  | 0.234   | **0.215**           |
| H                  | 0.229   | **0.217**           |
| G                  | 0.238   | **0.216**           |
| C_v                | 0.184   | **0.081**           |



## Acknowledgements

The code is based on:

* Provably Powerful Graph Networks
  (https://github.com/hadarser/ProvablyPowerfulGraphNetworks_torch)
* How Powerful are Graph Neural Networks?
  (https://github.com/weihua916/powerful-gnns)

We sincerely thank them for their contributions.
