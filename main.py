# Copyright (c) Gorilla Lab, SCUT.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Training/testing routines of DualPoseNet for category-level pose estimation on CAMERA25 or REAL275.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import logging
import pprint
pp = pprint.PrettyPrinter()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'model'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

import tensorflow as tf
import configs
from dualposenet import DualPoseNet
from evaluation_utils import evaluate


def run():
    FLAGS = configs.parse()
    assert FLAGS.dataset=='REAL275' or FLAGS.dataset=='CAMERA25', 'Error dataset of {}, which should be chosen from [REAL275, CAMERA25]'.format(FLAGS.dataset)
    assert FLAGS.phase in ['train', 'test', 'test_refine_encoder', 'test_refine_feature'], 'Error dataset of {}, which should be chosen from [train, test, test_refine_encoder, test_refine_feature]'.format(FLAGS.phase)

    FLAGS.log_dir = os.path.join('log', FLAGS.dataset)
    if not os.path.exists('log'):
        os.makedirs('log')
    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    if FLAGS.phase !='train':
        FLAGS.test_log_dir = os.path.join(FLAGS.log_dir, FLAGS.phase + '_epoch' + str(FLAGS.test_epoch))
        if not os.path.exists(FLAGS.test_log_dir):
            os.makedirs(FLAGS.test_log_dir)

    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True

    with tf.Session(config=run_config) as sess:
    	model = DualPoseNet(FLAGS,sess)

    	if FLAGS.phase == 'train':
    		model.train()
    	else:
            if FLAGS.phase == 'test':
                model.test()
            elif FLAGS.phase == 'test_refine_encoder':
                model.test_refine_encoder()
            elif FLAGS.phase == 'test_refine_feature':
                model.test_refine_feature()

            print('\n*********** Evaluate the results on {} ***********'.format(FLAGS.dataset))
            evaluate(FLAGS.test_log_dir)


def main(unused_argv):
    run()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    tf.app.run()



