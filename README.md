# DeepST-Rebuild

## Introduction
This repo is a tensorflow rebuilding from the work of

Junbo Zhang, Yu Zheng, Dekang Qi. Deep Spatio-Temporal Residual Networks for Citywide Crowd Flows Prediction. In AAAI 2017.
https://github.com/lucktroy/DeepST

The purpose of this modified script is for interview of JD and practice of
building deep neural network.

All rights belong to the researchers and institutes listed in above paper.

Details of this model please refer to above paper and repo.

## Modified subroutine
There are two subroutine modified from original work, `exptBikeNYC.py` and
`/deepst/models/STResNet1.py`, and one added `deepst/utils/random_mini_batches.py`.

In the original work, `STResNet.py` and `exptBikeNYC.py` are developed based on
Keras 1.2. Here I rebuild them using python3 and tensorflow 1.8. `random_mini_batches.py` is added
for model traning purpose.

I didn't change other subroutines and modules, since they are focusing on the
input and preprocessing of data.

## Hyperparameters and settings
All the hyperparameters are same as original work in the repo above.

Main packages including tensorflow, numpy, pandas.

## Run
In command line:
```
export DATAPATH=[path_to_your_data]
python exptBikeNYC.py
```

## Result
First I need to mention is long running time of the program for lacking
of computational power. For 500 epoches, it took 10.5 hours to finish. `RMSE` is
used for evaluation of the model, and result as below:

Train Accuracy: 0.011191

Test Accuracy: 0.042504

Cost of every 5 epoches is recored in `output.txt`.

It looks like there is a potential overfitting of the model. One possible reason
is the limited number of data points. Training set has 3480 data points and Test
has 240. Though shuffle and random mini batch are used for training, the dataset
is still not very large. The other potential reason is the hyperparameters setting
of the model. Like number of residual units, learning rate, etc. I will dig more
about the model and update.
