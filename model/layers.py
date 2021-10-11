# Copyright (c) Gorilla Lab, SCUT.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Custom layers.

Modified based on: https://github.com/daniilidis-group/spherical-cnn.
"""


import tensorflow as tf
import numpy as np

import spherical_utils
import tfnp_compatibility as tfnp


def sphconv(inputs, filters, use_bias=True, n_filter_params=0, weight_decay=0,
            *args, **kwargs):
    shapei = tfnp.shape(inputs)
    spectral_input = True if len(shapei) == 5 else False
    nchan = shapei[-1]
    n = shapei[2]
    if spectral_input:
        n *= 2

    with tf.variable_scope(None, default_name='sphconv'):
        # we desire that variance to stay constant after every layer
        # factor n // 2 is because we sum n // 2 coefficients
        # factor nchan is because we sum nchan channels
        # factor 2pi is the 'gain' of integrating over SO(3)
        # the factor 2 takes into account non-zero mean
        # (see He et al 2015 "Delving deep into rectifiers")
        std = 2./(2 * np.pi * np.sqrt((n // 2) * (nchan)))
        regularizer = tf.contrib.layers.l2_regularizer(
            weight_decay) if weight_decay > 0 else None

        if n_filter_params == 0:
            weights = tf.get_variable('W',
                                      trainable=True,
                                      initializer=tf.truncated_normal([nchan, n // 2, filters],
                                                                      stddev=std),
                                      regularizer=regularizer)
            ker = weights[:, np.newaxis, :, np.newaxis, :]
        else:
            nw_in = n_filter_params
            if nw_in > n // 2:
                nw_in = n // 2

            weights = tf.get_variable('W',
                                      trainable=True,
                                      initializer=tf.truncated_normal([nchan,
                                                                       nw_in,
                                                                       filters],
                                                                      stddev=std),
                                      regularizer=regularizer)
            xw_in = np.linspace(0, 1, nw_in)
            xw_out = np.linspace(0, 1, n // 2)
            id_out = np.searchsorted(xw_in, xw_out)
            subws = []
            for i, x in zip(id_out, xw_out):
                # linear interpolation
                # HACK! we assume the first indices match so i-1 when i==0 cancels out
                subws.append(weights[:, i-1, :] +
                             (weights[:, i, :] - weights[:, i-1, :]) *
                             (x-xw_in[i-1]) /
                             ((xw_in[i]-xw_in[i-1])))
            ker = tf.stack(subws, axis=1)[:, np.newaxis, :, np.newaxis, :]

        if use_bias:
            bias = tf.get_variable('b',
                                   trainable=True,
                                   initializer=tf.zeros([1, 1, 1, filters], dtype=tf.float32))
        else:
            bias = tf.zeros([1, 1, 1, filters], dtype=tf.float32)

        conv = spherical_utils.sph_conv_batch(inputs, ker, *args, **kwargs)
        conv = conv + bias

        for k, v in {'W': weights, 'b': bias, 'activations': conv}.items():
            tf.summary.histogram(k, v)
        # avg
        tf.summary.scalar('norm_activation', tf.reduce_mean(
            tf.norm(conv, axis=(1, 2)) / n))

    return conv


def block(params, fun, is_training=None, *args, **kwargs):
    """ Block consisting of weight layer + batch norm + nonlinearity"""
    params.batch_norm = False
    use_bias = not params.batch_norm
    curr = fun(*args, **kwargs, use_bias=use_bias)
    if params.batch_norm:
        curr = tf.layers.batch_normalization(curr,
                                             # doesn't make sense to learn scale when using ReLU
                                             fused=False,
                                             scale=False,
                                             training=is_training,
                                             renorm=params.batch_renorm)
        for v in tf.get_variable_scope().trainable_variables():
            if 'batch_normalization' in v.name:
                tf.summary.histogram(v.name, v)

    return nonlin(params)(curr)


def nonlin(params):
    return getattr(tf.nn, params.nonlin, globals().get(params.nonlin))


def identity(inputs):
    return inputs


def prelu(inputs):
    """ From: https://stackoverflow.com/a/40264459 """
    alphas = tf.Variable(0.1 * tf.ones(inputs.get_shape()[-1]),
                         trainable=True,
                         dtype=tf.float32)
    pos = tf.nn.relu(inputs)
    neg = alphas * (inputs - abs(inputs)) * 0.5

    return pos + neg


def area_weights(x, invert=False):
    """ Apply weight each cell according to its area; useful for averaging/avg pooling. """
    n = tfnp.shape(x)[1]
    phi, theta = spherical_utils.sph_sample(n)
    phi += np.diff(phi)[0]/2
    # this is proportional to the cell area, not exactly the area
    # this is the same as using |cos\phi_1 - cos\phi_2|
    if invert:
        x /= np.sin(phi)[np.newaxis, np.newaxis, :, np.newaxis]
    else:
        x *= np.sin(phi)[np.newaxis, np.newaxis, :, np.newaxis]

    return x


def fully_connected(inputs,
                    num_outputs,
                    scope,
                    use_xavier=True,
                    stddev=1e-3,
                    weight_decay=0.00001,
                    activation_fn=tf.nn.relu,
                    bn=False,
                    bn_decay=None,
                    use_bias=True,
                    is_training=None):
    """ Fully connected layer with non-linear operation.

    Args:
            inputs: 2-D tensor BxN
            num_outputs: int

    Returns:
            Variable tensor of size B x num_outputs.
    """

    with tf.variable_scope(scope) as sc:
        if use_xavier:
            initializer = tf.contrib.layers.xavier_initializer()
        else:
            initializer = tf.truncated_normal_initializer(stddev=stddev)

        if weight_decay > 0:
            outputs = tf.layers.dense(inputs, num_outputs,
                                      use_bias=use_bias, kernel_initializer=initializer,
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                          weight_decay),
                                      bias_regularizer=tf.contrib.layers.l2_regularizer(
                                          weight_decay),
                                      reuse=None)
        else:
            outputs = tf.layers.dense(inputs, num_outputs,
                                      use_bias=use_bias, kernel_initializer=initializer,
                                      kernel_regularizer=None,
                                      bias_regularizer=None,
                                      reuse=None)
        if bn:
            # outputs = tf.layers.batch_normalization(outputs, momentum=bn_decay, training=is_training, renorm=False)
            outputs = tf.layers.batch_normalization(
                outputs, training=is_training, renorm=False)

        if activation_fn is not None:
            outputs = activation_fn(outputs)

        return outputs


def conv1d(inputs,
           num_output_channels,
           kernel_size,
           scope=None,
           stride=1,
           padding='SAME',
           use_xavier=True,
           stddev=1e-3,
           weight_decay=0.00001,
           activation_fn=tf.nn.relu,
           bn=False,
           bn_decay=None,
           use_bias=True,
           is_training=None,
           reuse=None):
    """ 1D convolution with non-linear operation.

    Args:
    inputs: 3-D tensor variable BxHxWxC
    num_output_channels: int
    kernel_size: int
    scope: string
    stride: a list of 2 ints
    padding: 'SAME' or 'VALID'
    use_xavier: bool, use xavier_initializer if true
    stddev: float, stddev for truncated_normal init
    weight_decay: float
    activation_fn: function
    bn: bool, whether to use batch norm
    bn_decay: float or float tensor variable in [0,1]
    is_training: bool Tensor variable

    Returns:
    Variable tensor
    """
    with tf.variable_scope(scope, reuse=reuse):
        if use_xavier:
            initializer = tf.contrib.layers.xavier_initializer()
        else:
            initializer = tf.truncated_normal_initializer(stddev=stddev)

        if weight_decay > 0:
            outputs = tf.layers.conv1d(inputs, num_output_channels, kernel_size, stride, padding,
                                       kernel_initializer=initializer,
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                           weight_decay),
                                       bias_regularizer=tf.contrib.layers.l2_regularizer(
                                           weight_decay),
                                       use_bias=use_bias, reuse=None)
        else:
            outputs = tf.layers.conv1d(inputs, num_output_channels, kernel_size, stride, padding,
                                       kernel_initializer=initializer,
                                       kernel_regularizer=None,
                                       bias_regularizer=None,
                                       use_bias=use_bias, reuse=None)

        if bn:
            outputs = tf.layers.batch_normalization(
                outputs, momentum=bn_decay, training=is_training, renorm=False, fused=True)

        if activation_fn is not None:
            outputs = activation_fn(outputs)

    return outputs


def quaternion2mat(q):
    # quaternion2mat(q) = transforms3d.euler.qua2mat(q).T
    b = tf.shape(q)[0]

    w = tf.slice(q, [0, 0], [-1, 1])
    x = tf.slice(q, [0, 1], [-1, 1])
    y = tf.slice(q, [0, 2], [-1, 1])
    z = tf.slice(q, [0, 3], [-1, 1])

    r1 = tf.reshape(
        tf.concat([1-2*(y**2)-2*(z**2), 2*x*y+2*w*z, 2*x*z-2*w*y], axis=1), [b, 1, 3])
    r2 = tf.reshape(
        tf.concat([2*x*y-2*w*z, 1-2*(x**2)-2*(z**2), 2*y*z+2*w*x], axis=1), [b, 1, 3])
    r3 = tf.reshape(
        tf.concat([2*x*z+2*w*y, 2*y*z-2*w*x, 1-2*(x**2)-2*(y**2)], axis=1), [b, 1, 3])

    r = tf.concat([r1, r2, r3], axis=1)
    return r


def point_transformation(pc, rotation, translation, scale):
    b = tf.shape(pc)[0]

    r = quaternion2mat(rotation)
    pc = (pc - tf.tile(tf.reshape(translation, [b, 1, 3]), [1, 1024, 1]))/tf.tile(
        tf.reshape((tf.linalg.norm(scale, axis=1)+1e-10), [b, 1, 1]), [1, 1024, 3])
    pc = tf.matmul(pc, r)
    return pc