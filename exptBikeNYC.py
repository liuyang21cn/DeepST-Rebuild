# -*- coding: utf-8 -*-
"""
This script is modified from the orginal work of

Junbo Zhang, Yu Zheng, Dekang Qi. Deep Spatio-Temporal Residual Networks for Citywide Crowd Flows Prediction. In AAAI 2017.

The purpose of this modified script is for interview of JD.

All rights belong to the researchers and institutes listed in above paper.

"""
from __future__ import print_function
import os
import pickle
import numpy as np
import math
import tensorflow as tf

from deepst.models.STResNet1 import stresnet
from deepst.config import Config
import deepst.metrics as metrics
from deepst.datasets import BikeNYC
from deepst.utils.random_mini_batches import random_mini_batches
np.random.seed(1337)  # for reproducibility

# parameters
# data path, you may set your own data path with the global envirmental
# variable DATAPATH
DATAPATH = Config().DATAPATH
nb_epoch = 500  # number of epoch at training stage
nb_epoch_cont = 100  # number of epoch at training (cont) stage
batch_size = 32  # batch size
T = 24  # number of time intervals in one day

lr = 0.0002  # learning rate
len_closeness = 3  # length of closeness dependent sequence
len_period = 4  # length of peroid dependent sequence
len_trend = 4  # length of trend dependent sequence
nb_residual_unit = 4   # number of residual units

nb_flow = 2  # there are two types of flows: new-flow and end-flow
# divide data into two subsets: Train & Test, of which the test set is the
# last 10 days
days_test = 10
len_test = T * days_test
map_height, map_width = 16, 8  # grid size
# For NYC Bike data, there are 81 available grid-based areas, each of
# which includes at least ONE bike station. Therefore, we modify the final
# RMSE by multiplying the following factor (i.e., factor).
nb_area = 81
m_factor = math.sqrt(1. * map_height * map_width / nb_area)
# print('factor: ', m_factor)
path_result = 'RET'
path_model = 'MODEL'

if os.path.isdir(path_result) is False:
    os.mkdir(path_result)
if os.path.isdir(path_model) is False:
    os.mkdir(path_model)


def build_model(external_dim):

    # create placeholder for X and Y
    XC = tf.placeholder(tf.float64, [None, len_closeness * nb_flow, map_height, map_width],
                        name="XC") if len_closeness > 0 else None
    XP = tf.placeholder(tf.float64, [None, len_period * nb_flow, map_height, map_width],
                        name="XP") if len_period > 0 else None
    XT = tf.placeholder(tf.float64, [None, len_trend * nb_flow, map_height, map_width],
                        name="XT") if len_trend > 0 else None
    Y  = tf.placeholder(tf.float64, [None, nb_flow, map_height, map_width], name="Y")

    # model output of predictions
    model_output, X_ext = stresnet(XC, XP, XT,
                            external_dim = external_dim,
                            nb_flow = 2,
                            map_height = map_height,
                            map_width = map_width,
                            nb_residual_unit = nb_residual_unit)
    # cost function
    cost = tf.losses.mean_squared_error(labels = Y, predictions = model_output)
    # optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)
    # RMSE for metrics
    accuracy = tf.sqrt(tf.losses.mean_squared_error(Y, model_output))

    return cost, optimizer, accuracy, model_output, XC, XP, XT, X_ext, Y


def main():
    # load data
    print("loading data...")
    X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_test = BikeNYC.load_data(
        T=T, nb_flow=nb_flow, len_closeness=len_closeness, len_period=len_period, len_trend=len_trend, len_test=len_test,
        preprocess_name='preprocessing.pkl', meta_data=True)

    # print("\n days (test): ", [v[:8] for v in timestamp_test[0::T]])
    print('='*10 + ' Build model and start traning ' + '='*10)
    cost, optimizer, accuracy, model_output, XC, XP, XT, X_ext, Y = build_model(external_dim)
    # Initialize all the variables
    init = tf.global_variables_initializer()
    m = X_train[0].shape[0]
    seed = 0
    print("number of data points %i" % m)

    # Start the session to compute the tensorflow graph
    print_cost = True

    # timer
    import time
    start = time.time()

    # variables for early_stopping
    early_stopping_num = 0
    epoch_cost_prev = float('inf')

    # model saver
    saver = tf.train.Saver()

    # start training
    with tf.Session() as sess:
        # Initialize variables
        sess.run(init)
        for epoch in range(nb_epoch):
            epoch_cost = 0.
            # shuffle dataset and use random_mini_batches for training
            num_minibatches = int(m / batch_size)
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, m, batch_size, seed)
            for minibatch in minibatches:
                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                _ , minibatch_cost = sess.run([optimizer, cost],
                                                feed_dict={XC: minibatch_X[0],
                                                           XP: minibatch_X[1],
                                                           XT: minibatch_X[2],
                                                           X_ext: minibatch_X[3],
                                                           Y: minibatch_Y})
                # update cost at current epoch
                epoch_cost += minibatch_cost / num_minibatches

            # early_stopping
            if epoch_cost > epoch_cost_prev:
                early_stopping_num += 1
            else:
                early_stopping_num = 0
            if early_stopping_num > 5:
                print("Training early stops at epcho %i" % epoch)
                print("Current training error is %f" % epoch_cost)
                break # break from iterations

            # update previous cost
            epoch_cost_prev = epoch_cost
            # Print the cost every 5 epoch
            if print_cost == True and epoch % 5 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))

        # save model
        saver.save(sess, 'saved_model/model', global_step = epoch)

        # examine RMSE for both train and test
        print("="*10 + " Check model accuracy " + "="*10)
        print ("Train Accuracy: %f" % accuracy.eval({XC: X_train[0],
                                                 XP: X_train[1],
                                                 XT: X_train[2],
                                                 X_ext: X_train[3],
                                                 Y: Y_train}))
        print ("Test Accuracy: %f" % accuracy.eval({XC: X_test[0],
                                                XP: X_test[1],
                                                XT: X_test[2],
                                                X_ext: X_test[3],
                                                Y: Y_test}))
    print(" ")
    end = time.time()
    print("Running time of training and evaluation in seconds %f" % (end-start))


if __name__ == '__main__':
    main()
