# Copyright (c) Gorilla Lab, SCUT.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Dataloders of REAL275 and CAMERA25.

Author: Jiehong Lin
"""

import numpy as np
import queue
import threading
import glob


class Fetcher(threading.Thread):
    def __init__(self, opts):
        super(Fetcher, self).__init__()
        self.queue = queue.Queue(50)
        self.stopped = False
        self.opts = opts

        data_path = glob.glob('data/training_instances/CAMERA25_*.npz')
        if self.opts.dataset == 'REAL275':
            data_path += 'data/training_instances/REAL275.npz'
        data = [np.load(p) for p in range(data_path)]
        K = len(data)

        self.observed_pc = np.concatenate([data[k]['pts'] for k in range(K)], aixs=0)
        self.input_dis = np.concatenate([data[k]['smap'][:, :, :, 0][:, :, :, np.newaxis] for k in range(K)], aixs=0)
        self.input_rgb = np.concatenate([data[k]['smap'][:, :, :, 1:] for k in range(K)], aixs=0)
        self.rotation = np.concatenate([data[k]['rotation'] for k in range(K)], aixs=0)
        self.translation = np.concatenate([data[k]['translation'] for k in range(K)], aixs=0)
        self.scale = np.concatenate([data[k]['size'] for k in range(K)], aixs=0)

        # print(input_data)
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

