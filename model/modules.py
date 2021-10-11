# Copyright (c) Gorilla Lab, SCUT.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Main Modules in DualPoseNet.

* The code of encoder is modified based on: https://github.com/daniilidis-group/spherical-cnn.

Author: Jiehong Lin
"""
import numpy as np
import tensorflow as tf

import spherical_utils
from layers import sphconv, conv1d, fully_connected, area_weights, block


def dup(x):
    """ Return two references for input; useful when creating NNs and storing references to layers """
    return [x, x]


class encoder(object):
    def __init__(self, opts, is_training, name="encoder"):
        self.opts = opts
        self.is_training = is_training
        self.name = name
        self.reuse = False
        self.bn = False

        self.method = self.opts.transform_method
        self.real = self.opts.real_inputs

        if self.method == 'naive':
            args = self.opts
            fun = lambda *args, **kwargs: spherical_utils.sph_harm_all(
                *args, **kwargs, real=self.real)

        with tf.name_scope('harmonics_or_legendre'):
            res = self.opts.input_res
            self.harmonics = [fun(res // (2**i), as_tfvar=True)
                              for i in range(sum(self.opts.pool_layers) + 1)]

    def __call__(self, dis_inputs, rgb_inputs):

        net = {}
        high = 0
        low = 1
        args = self.opts
        convfun = sphconv
        l_or_h = self.harmonics
        method = self.method
        dis_curr = dis_inputs
        rgb_curr = rgb_inputs

        with tf.variable_scope(self.name, reuse=self.reuse):

            # Main Streams with Spherical Fusion
            for i, (nf, pool) in enumerate(zip(args.nfilters, args.pool_layers)):
                if i in [3, 5, 7]:
                    with tf.variable_scope('fusion_conv{}'.format(i), reuse=tf.AUTO_REUSE):
                        fusion_curr = tf.concat([dis_curr, rgb_curr], axis=-1)
                        net['fusion_conv{}'.format(i)], fusion_curr = dup(block(args, convfun, self.is_training, fusion_curr, nf,
                                                                                n_filter_params=args.n_filter_params, weight_decay=args.weight_decay, 
                                                                                harmonics_or_legendre=l_or_h[high], method=method))
                        dis_curr = tf.concat([dis_curr, fusion_curr], axis=-1)
                        rgb_curr = tf.concat([rgb_curr, fusion_curr], axis=-1)
                else:
                    if not pool:
                        with tf.variable_scope('dis_conv{}'.format(i), reuse=tf.AUTO_REUSE):
                            net['dis_conv{}'.format(i)], dis_curr = dup(block(args, convfun, self.is_training, dis_curr, nf,
                                                                              n_filter_params=args.n_filter_params, weight_decay=args.weight_decay, 
                                                                              harmonics_or_legendre=l_or_h[high], method=method))
                        with tf.variable_scope('rgb_conv{}'.format(i), reuse=tf.AUTO_REUSE):
                            net['rgb_conv{}'.format(i)], rgb_curr = dup(block(args, convfun, self.is_training, rgb_curr, nf,
                                                                              n_filter_params=args.n_filter_params, weight_decay=args.weight_decay, 
                                                                              harmonics_or_legendre=l_or_h[high], method=method))
                    else:
                        with tf.variable_scope('dis_conv{}'.format(i), reuse=tf.AUTO_REUSE):
                            net['dis_conv{}'.format(i)], dis_curr = dup(block(args, convfun, self.is_training, dis_curr, nf, n_filter_params=args.n_filter_params,
                                                                              weight_decay=args.weight_decay, harmonics_or_legendre=l_or_h[
                                                                                  high], method=method,
                                                                              spectral_pool=0, harmonics_or_legendre_low=l_or_h[low]))
                        with tf.variable_scope('rgb_conv{}'.format(i), reuse=tf.AUTO_REUSE):
                            net['rgb_conv{}'.format(i)], rgb_curr = dup(block(args, convfun, self.is_training, rgb_curr, nf, n_filter_params=args.n_filter_params,
                                                                              weight_decay=args.weight_decay, harmonics_or_legendre=l_or_h[
                                                                                  high], method=method,
                                                                              spectral_pool=0, harmonics_or_legendre_low=l_or_h[low]))
                        # pooling
                        dis_curr = area_weights(tf.layers.average_pooling2d(
                            area_weights(dis_curr), 2*pool, 2*pool, 'same'), invert=True)
                        rgb_curr = area_weights(tf.layers.average_pooling2d(
                            area_weights(rgb_curr), 2*pool, 2*pool, 'same'), invert=True)

                        high += 1
                        low += 1

            # Aggregation of Multi-scale Spherical Features
            feat1 = net['fusion_conv3']
            feat1 = tf.layers.flatten(inputs=feat1)
            feat1 = fully_connected(feat1, 1024, scope='flatten_conv3', weight_decay=self.opts.weight_decay,
                                    activation_fn=tf.nn.leaky_relu, is_training=self.is_training)

            feat2 = net['fusion_conv5']
            feat2 = tf.layers.flatten(inputs=feat2)
            feat2 = fully_connected(feat2, 1024, scope='flatten_conv5', weight_decay=self.opts.weight_decay,
                                    activation_fn=tf.nn.leaky_relu, is_training=self.is_training)

            feat3 = net['fusion_conv7']
            feat3 = tf.layers.flatten(inputs=feat3)
            feat3 = fully_connected(feat3, 1024, scope='flatten_conv7', weight_decay=self.opts.weight_decay,
                                    activation_fn=tf.nn.leaky_relu, is_training=self.is_training)

            if self.opts.fusion == 'Avg':
                curr = tf.concat([tf.expand_dims(feat1, 2), tf.expand_dims(
                    feat2, 2), tf.expand_dims(feat3, 2)], axis=2)
                curr = tf.reduce_mean(curr, axis=2)
            elif self.opts.fusion == 'Max':
                curr = tf.concat([tf.expand_dims(feat1, 2), tf.expand_dims(
                    feat2, 2), tf.expand_dims(feat3, 2)], axis=2)
                curr = tf.reduce_max(curr, axis=2)
            elif self.opts.fusion == 'Cat':
                curr = tf.concat([feat1, feat2, feat3], axis=-1)
            else:
                assert False

            curr = fully_connected(curr, 1024, scope='fc1', weight_decay=self.opts.weight_decay,
                                   activation_fn=tf.nn.leaky_relu, is_training=self.is_training)
            curr = fully_connected(curr, 1024, scope='fc2', weight_decay=self.opts.weight_decay,
                                   activation_fn=tf.nn.leaky_relu, is_training=self.is_training)

        self.reuse = True
        self.variables = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, self.name)

        return curr


class explicit_decoder(object):
    def __init__(self, opts, is_training, name="explicit_decoder"):
        self.opts = opts
        self.is_training = is_training
        self.name = name
        self.reuse = False
        self.bn = False

    def __call__(self, feat):

        with tf.variable_scope(self.name, reuse=self.reuse):

            rot_feat = fully_connected(feat, 1024, scope='rot_fc1', weight_decay=self.opts.weight_decay,
                                       activation_fn=tf.nn.leaky_relu, is_training=self.is_training)
            rot_feat = fully_connected(rot_feat, 512, scope='rot_fc2', weight_decay=self.opts.weight_decay,
                                       activation_fn=tf.nn.leaky_relu, is_training=self.is_training)
            rot_feat = tf.layers.dense(
                inputs=rot_feat, activation=None, units=4, name="rot_fc3")
            rotation = rot_feat / \
                (tf.linalg.norm(rot_feat, axis=1, keepdims=True)+1e-10)

            trans_feat = fully_connected(feat, 1024, scope='trans_fc1', weight_decay=self.opts.weight_decay,
                                         activation_fn=tf.nn.leaky_relu, is_training=self.is_training)
            trans_feat = fully_connected(trans_feat, 512, scope='trans_fc2', weight_decay=self.opts.weight_decay,
                                         activation_fn=tf.nn.leaky_relu, is_training=self.is_training)
            translation = tf.layers.dense(
                inputs=trans_feat, activation=None, units=3, name="trans_fc3")

            scale_feat = fully_connected(feat, 1024, scope='scale_fc1', weight_decay=self.opts.weight_decay,
                                         activation_fn=tf.nn.leaky_relu, bn=self.opts.batch_norm, is_training=self.is_training)
            scale_feat = fully_connected(scale_feat, 512, scope='scale_fc2', weight_decay=self.opts.weight_decay,
                                         activation_fn=tf.nn.leaky_relu, bn=self.opts.batch_norm, is_training=self.is_training)
            scale = tf.layers.dense(
                inputs=scale_feat, activation=None, units=3, name="scale_fc3")


        self.reuse = True
        self.variables = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, self.name)

        return translation, rotation, scale


class implicit_decoder(object):
    def __init__(self, opts, is_training, output_npoint=1024, name="implcit_decoder"):
        self.opts = opts
        self.npoint = output_npoint
        self.is_training = is_training
        self.name = name
        self.reuse = False

    def __call__(self, pts, feat):
        with tf.variable_scope(self.name, reuse=self.reuse):
            b = pts.get_shape()[0].value
            n = pts.get_shape()[1].value
            c = feat.get_shape()[1].value

            feat = tf.concat(
                [pts, tf.tile(tf.reshape(feat, [b, 1, c]), [1, n, 1])], axis=-1)
            feat = conv1d(feat, 1024, 1, scope='conv1', weight_decay=self.opts.weight_decay,
                          activation_fn=tf.nn.leaky_relu, is_training=self.is_training)
            feat = conv1d(feat, 512, 1, scope='conv2', weight_decay=self.opts.weight_decay,
                          activation_fn=tf.nn.leaky_relu, is_training=self.is_training)
            feat = conv1d(feat, 256, 1, scope='conv3', weight_decay=self.opts.weight_decay,
                          activation_fn=tf.nn.leaky_relu, is_training=self.is_training)
            Q = conv1d(feat, 3, 1, scope='conv4', weight_decay=self.opts.weight_decay,
                       activation_fn=None, bn=False, is_training=self.is_training)

        self.reuse = True
        self.variables = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, self.name)

        return Q
