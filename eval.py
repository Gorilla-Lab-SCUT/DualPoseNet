# Copyright (c) Gorilla Lab, SCUT.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Evaluation of results of DualPoseNet on CAMERA25 and REAL275.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from evaluation_utils import evaluate


CAMERA25_path = os.path.join('results', 'CAMERA25')
REAL275_path = os.path.join('results', 'REAL275')


print('\n*********** Evaluate the results of DualPoseNet on CAMERA25 ***********')
evaluate(CAMERA25_path)

print('\n*********** Evaluate the results of DualPoseNet on REAL275  ***********')
evaluate(REAL275_path)
