# Copyright (c) Gorilla Lab, SCUT.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Dataloders of REAL275 and CAMERA25.

Author: Jiehong Lin
"""

import os
import numpy as np
import queue
import threading
import glob
import _pickle as cPickle

class Fetcher(threading.Thread):
    def __init__(self, opts):
        super(Fetcher, self).__init__()
        self.queue = queue.Queue(50)
        self.stopped = False
        self.opts = opts


        data_paths = glob.glob('data/training_instance/CAMERA25_*.pkl')
        if self.opts.dataset == 'REAL275':
            data_paths.append('data/training_instance/REAL275.pkl')
        print(data_paths)

        self.observed_pc = []
        self.input_dis = []
        self.input_rgb = []
        self.rotation = []
        self.translation = []
        self.scale = []

        for data_path in data_paths:
            print(data_path)
            with open(data_path, 'rb') as f:
                data = cPickle.load(f)
            self.observed_pc.append(data['observed_pc'])
            self.input_dis.append(data['input_dis'])
            self.input_rgb.append(data['input_rgb'])
            self.rotation.append(data['rotation'])
            self.translation.append(data['translation'])
            self.scale.append(data['scale'])

        self.observed_pc = np.concatenate(self.observed_pc, axis=0)
        self.input_dis = np.concatenate(self.input_dis, axis=0)
        self.input_rgb = np.concatenate(self.input_rgb, axis=0)
        self.rotation = np.concatenate(self.rotation, axis=0)
        self.translation = np.concatenate(self.translation, axis=0)
        self.scale = np.concatenate(self.scale, axis=0)

        self.batch_size = self.opts.batch_size
        self.sample_cnt = self.input_dis.shape[0]
        self.num_batches = self.sample_cnt//self.batch_size
        print ("NUM_INSTANCE is %s"%(self.sample_cnt))
        print ("NUM_BATCH is %s"%(self.num_batches))

    def run(self):
        while not self.stopped:
            idx = np.arange(self.sample_cnt)
            np.random.shuffle(idx)
            self.observed_pc = self.observed_pc[idx, ...]
            self.input_dis = self.input_dis[idx, ...]
            self.input_rgb = self.input_rgb[idx, ...]
            self.rotation = self.rotation[idx, ...]
            self.translation = self.translation[idx, ...]
            self.scale = self.scale[idx, ...]

            for batch_idx in range(self.num_batches):
                if self.stopped:
                    return None
                start_idx = batch_idx * self.batch_size
                end_idx = (batch_idx + 1) * self.batch_size

                batch_input_dis = self.input_dis[start_idx:end_idx, :, :, :].copy()
                batch_input_rgb = self.input_rgb[start_idx:end_idx, :, :, :].copy()
                batch_observed_pc = self.observed_pc[start_idx:end_idx, :, :].copy()
                batch_rotation = self.rotation[start_idx:end_idx, :].copy()
                batch_translation = self.translation[start_idx:end_idx, :].copy()
                batch_scale = self.scale[start_idx:end_idx, :].copy()
                self.queue.put((batch_input_dis, batch_input_rgb, batch_observed_pc, batch_rotation, batch_translation, batch_scale))
        return None

    def fetch(self):
        if self.stopped:
            return None
        return self.queue.get()

    def shutdown(self):
        self.stopped = True
        print ("Shutdown .....")
        while not self.queue.empty():
            self.queue.get()
        print ("Remove all queue data")

