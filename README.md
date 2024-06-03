# Stocks-Prediction-Learning

To test out the different datasets, modify the ```DATA_PATH``` variable in the project.ipynb notebook.

```hyperparams.yaml``` contain the all the parameters used for this project. To use variable sequence lengths, set 
```
use_time_horizon: True
horizon: 1
```

Otherwise to use static sequence lengths, set

```
use_time_horizon: False
horizon: X
```
where X is an integer greater than 0