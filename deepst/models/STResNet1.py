'''
ST-ResNet: Deep Spatio-temporal Residual Networks

This script is modified from the orginal work of

Junbo Zhang, Yu Zheng, Dekang Qi. Deep Spatio-Temporal Residual Networks for Citywide Crowd Flows Prediction. In AAAI 2017.

The purpose of this modified script is for interview of JD.

All rights belong to the researchers and institutes listed in above paper.
'''

from __future__ import print_function
import tensorflow as tf
import numpy as np

def _shortcut(input, residual):
    # print("_shortcut", str(input), str(residual))
    return tf.add(input, residual)

def _bn_relu_conv(input, nb_filter, nb_row, nb_col, strides=(1, 1), bn=False):
    # print("_bn_relu_conv", str(input))
    if bn:
        # float64 is not supported in batch_normalization
        input = tf.cast(input, tf.float32)
        input = tf.layers.batch_normalization(input)
    input = tf.cast(input, tf.float64)
    activation = tf.nn.relu(input)
    output = tf.layers.conv2d(
            inputs = activation,
            filters = nb_filter,
            kernel_size = [nb_row, nb_col],
            strides = strides,
            padding = 'same')
            # data_format = 'channels_first')
    return output

def _residual_unit(input, nb_filter, strides=(1, 1)):
    # print("_residual_unit", str(input))
    residual = _bn_relu_conv(input, nb_filter, 3, 3)
    residual = _bn_relu_conv(residual, nb_filter, 3, 3)
    return _shortcut(input, residual)


def ResUnits(input, residual_unit, nb_filter, repetations=1):
    # print("ResUnits", str(input))
    # print("residual_unit")
    for i in range(repetations):
        strides = (1, 1)
        input = residual_unit(input, nb_filter = nb_filter,
                                  strides = strides)
    return input


def stresnet(XC, XP, XT, external_dim=8, nb_flow=2, map_height=16, map_width=8, nb_residual_unit=3):
    '''
    XC - tensor of Temporal Closeness
    XP - tensor of Period
    XT - tensor of Trend
    '''

    outputs = []
    for i, X in enumerate([XC, XP, XT]):
        if X is not None:
            # print(X)
            # Since for current dataset the channel is at index 1, change
            # channel to last index
            input = tf.transpose(X, [0, 2, 3, 1])
            # Conv1
            conv1 = tf.layers.conv2d(input,
                                    filters = 64,
                                    kernel_size = [3, 3],
                                    padding = 'same')
                                    # data_format = 'channels_first')
            # print("conv1", str(conv1))
            # [nb_residual_unit] Residual Units
            residual_output = ResUnits(conv1, _residual_unit, nb_filter=64,
                              repetations=nb_residual_unit)
            # print("residual_output", str(residual_output))
            # Conv2
            activation = tf.nn.relu(residual_output)
            # print("activation", str(activation))
            conv2 = tf.layers.conv2d(
                    inputs = activation,
                    filters = nb_flow,
                    kernel_size = [3, 3],
                    padding = 'same')
                    # data_format = 'channels_first')
            print("conv2", conv2)
            outputs.append(conv2)

    # parameter-matrix-based fusion
    if len(outputs) == 1:
        main_output = outputs[0]
    else:
        new_outputs = []
        for output in outputs:
            w = tf.Variable(np.random.random(output.get_shape().as_list()[1:]))
            new_output = tf.multiply(w, output)
            new_outputs.append(new_output)
        main_output = tf.add_n(new_outputs)
        print("main_output", main_output)

    # # fusing with external component
    if external_dim != None and external_dim > 0:
        # create tensor of external input
        external_input = tf.placeholder(tf.float64, [None, external_dim], name="X_ext")
        embedding = tf.layers.dense(external_input, units=10)
        embedding = tf.nn.relu(embedding)
        # print("embedding", embedding)
        h1 = tf.layers.dense(embedding, units=nb_flow * map_height * map_width)
        # print("h1", h1)
        activation = tf.nn.relu(h1)
        # print("activation", activation)
        external_output = tf.reshape(activation, shape=[tf.shape(activation)[0], nb_flow, map_height, map_width])
        # same here to put channel to last index
        external_output = tf.transpose(external_output, [0, 2, 3, 1])
        print("external_output", external_output)
        main_output = tf.add(main_output, external_output)
    else:
        external_input = None
        print('external_dim:', external_dim)

    main_output = tf.nn.tanh(main_output)
    # print("main_output", main_output)
    # change channel back to index 1
    main_output = tf.transpose(main_output, [0, 3, 1, 2])
    print("main_output", main_output)

    # return Y and external tensor
    return main_output, external_input

if __name__ == '__main__':
    model = stresnet(external_dim=28, nb_residual_unit=12)
    #plot(model, to_file='ST-ResNet.png', show_shapes=True)
    model.summary()
