# Copyright (c) Gorilla Lab, SCUT.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import os
import yaml

def parse(args=None):
    parser = argparse.ArgumentParser(description='Configurations of DualPoseNet.',
                                     fromfile_prefix_chars='@')
    parser.add_argument('--phase', default='train',
                        help="train/test/test_refine_encoder/test_refine_feature")
    parser.add_argument('--dataset', default='REAL275',
                        help="REAL275/CAMERA25")

    # dataset
    parser.add_argument('--n_classes', '-nc', type=int, default=6,
                        help='Number of classes in dataset')
    parser.add_argument('--input_res', '-res', type=int, default=64,
                        help='Resolution for spherical inputs; may subsample if larger')

    # model
    parser.add_argument('--nfilters', default=[16, 16, 32, 32, 64, 64, 128, 128],
                        type=lambda x: [int(_) for _ in x.split(',')],
                        help='Number of filters per layer')
    parser.add_argument('--pool_layers', default=[0, 0, 1, 0, 1, 0, 1, 0],
                        type=lambda x: [int(_) for _ in x.split(',')],
                        help='Pooling layer indicator')
    parser.add_argument('--n_filter_params', '-nfp', type=int, default=8,
                        help='Number of filter params (if 0, use max, else do spectral linear interpolation for localized filters.)')
    parser.add_argument('--nonlin', '-nl', type=str, default='prelu',
                        help='Nonlinearity to be used')
    parser.add_argument('--fusion', '-fs', type=str, default='Max',
                        help='fusion method in multi-scale aggregation[Max, Avg, Cat]')
    parser.add_argument('--transform_method', '-tm', choices=['naive', 'sep'], default='naive',
                        help='SH transform method: NAIVE or SEParation of variables')
    parser.add_argument('--real_inputs', '-ri', action='store_true', default=True,
                        help='Leverage symmetry when inputs are real.')

    # train
    parser.add_argument('--learning_rate', '-lr', type=yaml.load, default=0.0001,
                        help='Initial learning rate')
    parser.add_argument('--lr_decay_steps', type=int, default=40000)
    parser.add_argument('--lr_decay_rate', type=float, default=0.5)
    parser.add_argument('--training_epoch', '-ne', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', '-bs', type=int, default=64,
                        help='training batch size')
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--implicit_loss_weight', type=float, default=10.0,
                        help='Balanced parameter for the loss term of the inplicit pose decoder')

    # test
    parser.add_argument('--test_epoch', type=int,
                        default=30, help='Epoch for test')

    # use given args instead of cmd line, if they exist
    if isinstance(args, list):
        # if list, parse as cmd lines arguments
        args_out = parser.parse_args(args)
    elif args is not None:
        # if dict, set values directly
        args_out = parser.parse_args('')
        for k, v in args.items():
            setattr(args_out, k, v)
    else:
        args_out = parser.parse_args()
    return args_out
