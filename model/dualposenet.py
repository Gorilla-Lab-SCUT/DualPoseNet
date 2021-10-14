# Copyright (c) Gorilla Lab, SCUT.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Dual Pose Network with Refined Learning of Pose Consistency.

Author: Jiehong Lin
"""

import os
import tensorflow as tf
import numpy as np
import math
import cmath
import glob
import _pickle as cPickle
from tqdm import tqdm
import cv2
from transforms3d.euler import quat2mat

from layers import point_transformation
from modules import encoder, explicit_decoder, implicit_decoder
from dataloder import Fetcher
from pc_utils import load_depth, backproject, pc2sphericalmap


class DualPoseNet(object):
    def __init__(self, opts, sess):
        self.sess = sess
        self.opts = opts

        if self.opts.dataset == 'REAL275':
            self.intrinsics = np.array(
                [[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]])
        elif self.opts.dataset == 'CAMERA25':
            self.intrinsics = np.array(
                [[577.5, 0, 319.5], [0, 577.5, 239.5], [0, 0, 1]])

    def allocate_placeholders(self):
        self.is_training = tf.placeholder_with_default(
            True, shape=[], name='is_training')
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        self.input_dis = tf.placeholder(tf.float32, shape=[
                                        self.opts.batch_size, self.opts.input_res, self.opts.input_res, 1])
        self.input_rgb = tf.placeholder(tf.float32, shape=[
                                        self.opts.batch_size, self.opts.input_res, self.opts.input_res, 3])
        self.observed_pc = tf.placeholder(
            tf.float32, shape=[self.opts.batch_size, 1024, 3])

        self.gt_rotation = tf.placeholder(
            tf.float32, shape=[self.opts.batch_size, 4])
        self.gt_translation = tf.placeholder(
            tf.float32, shape=[self.opts.batch_size, 3])
        self.gt_scale = tf.placeholder(
            tf.float32, shape=[self.opts.batch_size, 3])

    def build_model(self):
        # model
        self.encoder = encoder(self.opts, self.is_training, name='encoder')
        self.explicit_decoder = explicit_decoder(
            self.opts, self.is_training, name='explicit_decoder')
        self.implicit_decoder = implicit_decoder(
            self.opts, self.is_training, name="implicit_decoder")

        # graphs
        self.pose_feat = self.encoder(self.input_dis, self.input_rgb)
        self.pred_translation, self.pred_rotation, self.pred_scale = self.explicit_decoder(
            self.pose_feat)
        self.pred_canonical_points = self.implicit_decoder(
            self.observed_pc, self.pose_feat)
        self.gt_canonical_points = point_transformation(
            self.observed_pc, self.gt_rotation, self.gt_translation, self.gt_scale)

        # loss
        self.translation_loss = tf.losses.huber_loss(
            self.pred_translation, self.gt_translation)
        self.rotation_loss = tf.losses.huber_loss(
            self.pred_rotation, self.gt_rotation)
        self.scale_loss = tf.losses.huber_loss(self.pred_scale, self.gt_scale)
        self.implicit_loss = tf.losses.huber_loss(
            self.pred_canonical_points, self.gt_canonical_points)
        self.loss = self.rotation_loss + self.translation_loss + \
            self.scale_loss + self.opts.implicit_loss_weight*self.implicit_loss

    def setup_optimizer(self):
        self.learning_rate = tf.train.exponential_decay(self.opts.learning_rate, self.global_step,
                                                        self.opts.lr_decay_steps, self.opts.lr_decay_rate, staircase=True)
        self.learning_rate = tf.maximum(self.learning_rate, 0.000001)

        all_update_ops = [op for op in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if op.name.startswith(
            "encoder") or op.name.startswith("explicit_decoder") or op.name.startswith("implicit_decoder")]
        all_tvars = [var for var in tf.trainable_variables() if var.name.startswith(
            "encoder") or var.name.startswith("explicit_decoder") or var.name.startswith("implicit_decoder")]
        with tf.control_dependencies(all_update_ops):
            self.all_optimizers = tf.train.AdamOptimizer(self.learning_rate).minimize(
                self.loss, var_list=all_tvars, colocate_gradients_with_ops=True, global_step=self.global_step)

    def train(self):
        print('\n*********** Training of DualPoseNet ***********')

        # model & graph
        print('building model ...')
        self.allocate_placeholders()
        self.build_model()
        self.setup_optimizer()
        print('model built !')

        # dataset
        print('loading data ...')
        fetchworker = Fetcher(self.opts)
        fetchworker.start()
        print('data loaded !')

        print('starting training ...')
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=None)
        step = self.sess.run(self.global_step)

        for epoch in range(1, self.opts.training_epoch+1):
            sum_loss = 0.0
            sum_r_loss = 0.0
            sum_t_loss = 0.0
            sum_s_loss = 0.0
            sum_i_loss = 0.0
            count = 0

            for batch_idx in range(fetchworker.num_batches):

                batch_input_dis, batch_input_rgb, batch_observed_pc, batch_rotation,  batch_translation, batch_scale = fetchworker.fetch()
                curr_bs = batch_input_dis.shape[0]

                feed_dict = {self.input_dis: batch_input_dis,
                             self.input_rgb: batch_input_rgb,
                             self.observed_pc: batch_observed_pc,
                             self.gt_rotation: batch_rotation,
                             self.gt_translation: batch_translation,
                             self.gt_scale: batch_scale,
                             self.is_training: True}

                _, loss, r_loss, t_loss, s_loss, i_loss = self.sess.run(
                    [self.all_optimizers, self.loss, self.rotation_loss, self.translation_loss, self.scale_loss, self.implicit_loss], feed_dict=feed_dict)

                sum_loss += (loss*curr_bs)
                sum_r_loss += (r_loss*curr_bs)
                sum_t_loss += (t_loss*curr_bs)
                sum_s_loss += (s_loss*curr_bs)
                sum_i_loss += (i_loss*curr_bs)
                count += curr_bs

                print('[{}][{}/{}] loss: {:.5f}({:.5f}) r_loss: {:.5f}({:.5f}) t_loss: {:.5f}({:.5f}) s_loss: {:.5f}({:.5f}) i_loss: {:.5f}({:.5f})'.format(
                    epoch, batch_idx, fetchworker.num_batches, sum_loss /
                    float(count), loss, sum_r_loss /
                    float(count), r_loss, sum_t_loss/float(count),
                    t_loss, sum_s_loss/float(count), s_loss, sum_i_loss/float(count), i_loss))
                step += 1

            if epoch % 10 == 0:
                self.saver.save(self.sess, os.path.join(
                    self.opts.log_dir, 'model'), epoch)

        fetchworker.shutdown()
        print('training finished !')

    def test(self):

        print('\n*********** Testing on {} ***********'.format(self.opts.dataset))

        # inputs
        self.input_dis = tf.placeholder(
            tf.float32, shape=[1, self.opts.input_res, self.opts.input_res, 1])
        self.input_rgb = tf.placeholder(
            tf.float32, shape=[1, self.opts.input_res, self.opts.input_res, 3])
        self.observed_pc = tf.placeholder(tf.float32, shape=[1, 1024, 3])
        self.is_training = tf.placeholder_with_default(
            False, shape=[], name='is_training')

        # modules
        self.encoder = encoder(self.opts, self.is_training, name='encoder')
        self.explicit_decoder = explicit_decoder(
            self.opts, self.is_training, name='explicit_decoder')
        self.implicit_decoder = implicit_decoder(
            self.opts, self.is_training, name="implicit_decoder")

        # graphs
        self.pose_feat = self.encoder(self.input_dis, self.input_rgb)
        self.pred_translation, self.pred_rotation, self.pred_scale = self.explicit_decoder(
            self.pose_feat)
        self.pred_canonical_points = self.implicit_decoder(
            self.observed_pc, self.pose_feat)

        # checkpoints
        print("loading from checkpoint ...")
        saver = tf.train.Saver()
        checkpoint_path = os.path.join(
            self.opts.log_dir, 'model-'+str(self.opts.test_epoch))
        saver.restore(self.sess, checkpoint_path)

        # test
        result_pkl_list = glob.glob(
            'data/segmentation_results/{}/results_*.pkl'.format(self.opts.dataset))
        result_pkl_list = sorted(result_pkl_list)
        n_image = len(result_pkl_list)
        print('no. of test images: {}\n'.format(n_image))

        for i, path in tqdm(enumerate(result_pkl_list)):
            with open(path, 'rb') as f:
                data = cPickle.load(f)
            image_path = data['image_path']
            num_instance = len(data['pred_class_ids'])

            image_path = os.path.join(
                '/data/linjiehong/posed3Ddet/NOCS_CVPR2019-master/', image_path)

            image = cv2.imread(image_path + '_color.png')[:, :, :3]
            image = image[:, :, ::-1]
            depth = load_depth(image_path)
            pred_mask = data['pred_masks']

            pred_RTs = np.zeros((num_instance, 4, 4))
            pred_scales = np.zeros((num_instance, 3))
            pred_RTs[:, 3, 3] = 1

            if num_instance != 0:
                for j in range(num_instance):
                    inst_mask = 255 * pred_mask[:, :, j].astype('uint8')
                    pts_ori, idx = backproject(
                        depth, self.intrinsics, inst_mask)
                    pts_ori = pts_ori/1000.0
                    rgb_ori = image[idx[0], idx[1], :]
                    rgb_ori = (
                        rgb_ori - np.array([123.7, 116.8, 103.9])[np.newaxis, :])/255.0

                    FLAG = pts_ori.shape[0] > 32
                    pts = pts_ori.copy()
                    rgb = rgb_ori.copy()

                    for k in range(3):
                        if FLAG:
                            centroid = np.mean(pts, axis=0)
                            pts = pts - centroid[np.newaxis, :]

                            input_dis, input_rgb = pc2sphericalmap(
                                pts, rgb, resolution=self.opts.input_res)
                            if pts.shape[0] > 1024:
                                input_pts = pts[np.random.choice(
                                    pts.shape[0], 1024, replace=False), :]
                            else:
                                input_pts = pts[np.random.choice(
                                    pts.shape[0], 1024), :]
                            feed_dict = {self.input_dis: input_dis, self.input_rgb: input_rgb,
                                         self.observed_pc: input_pts[np.newaxis, :, :], self.is_training: False}
                            pred_rotation, pred_translation, pred_size = self.sess.run(
                                [self.pred_rotation, self.pred_translation, self.pred_scale], feed_dict=feed_dict)

                            pred_rotation = quat2mat(pred_rotation[0])
                            pred_translation = pred_translation[0] + centroid
                            pred_scale = np.linalg.norm(pred_size[0])
                            pred_size = pred_size[0]/pred_scale

                            pred_canonical_pts = (
                                (pts_ori - pred_translation[np.newaxis, :])/pred_scale) @ np.transpose(pred_rotation)
                            dis = np.linalg.norm(pred_canonical_pts, axis=1)

                            FLAG = np.sum(dis < 0.6) > 32
                            pts = pts_ori[dis < 0.6].copy()
                            rgb = rgb_ori[dis < 0.6].copy()
                        else:
                            break

                    if pts_ori.shape[0] > 32:
                        pred_RTs[j, :3, :3] = np.diag(
                            np.ones((3))*pred_scale) @ pred_rotation.transpose()
                        pred_RTs[j, :3, 3] = pred_translation
                        pred_scales[j] = pred_size

                        z_180_RT = np.zeros((4, 4), dtype=np.float32)
                        z_180_RT[:3, :3] = np.diag([-1, -1, 1])
                        z_180_RT[3, 3] = 1
                        pred_RTs[j, :, :] = z_180_RT @ pred_RTs[j, :, :]
                    else:
                        pred_RTs[j] = np.eye(4)
                        pred_scales[j] = np.ones((3))

            data.pop('pred_masks')
            data['pred_RTs'] = pred_RTs
            data['pred_scales'] = pred_scales

            with open(os.path.join(self.opts.test_log_dir, path.split('/')[-1]), 'wb') as f:
                cPickle.dump(data, f)

    def test_refine_encoder(self):

        print(
            '\n*********** Testing & Refining on {} ***********'.format(self.opts.dataset))

        # inputs
        self.input_dis = tf.placeholder(
            tf.float32, shape=[1, self.opts.input_res, self.opts.input_res, 1])
        self.input_rgb = tf.placeholder(
            tf.float32, shape=[1, self.opts.input_res, self.opts.input_res, 3])
        self.observed_pc = tf.placeholder(tf.float32, shape=[1, 1024, 3])
        self.is_training = tf.placeholder_with_default(
            False, shape=[], name='is_training')
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        # modules
        self.encoder = encoder(self.opts, self.is_training, name='encoder')
        self.explicit_decoder = explicit_decoder(
            self.opts, self.is_training, name='explicit_decoder')
        self.implicit_decoder = implicit_decoder(
            self.opts, self.is_training, name="implicit_decoder")

        # graphs
        self.pose_feat = self.encoder(self.input_dis, self.input_rgb)
        self.pred_translation, self.pred_rotation, self.pred_scale = self.explicit_decoder(
            self.pose_feat)
        self.ex_canonical_points = point_transformation(
            self.observed_pc, self.pred_rotation, self.pred_translation, self.pred_scale)
        self.im_canonical_points = self.implicit_decoder(
            self.observed_pc, self.pose_feat)

        # self-adaptive loss
        self.loss = tf.losses.huber_loss(
            self.ex_canonical_points, self.im_canonical_points)

        # optimizer
        self.learning_rate = tf.train.exponential_decay(
            self.opts.refine_learning_rate, self.global_step, 100000000, 0, staircase=True)
        all_update_ops = [op for op in tf.get_collection(
            tf.GraphKeys.UPDATE_OPS) if op.name.startswith("encoder")]
        all_tvars = [var for var in tf.trainable_variables()
                     if var.name.startswith("encoder")]
        with tf.control_dependencies(all_update_ops):
            self.all_optimizers = tf.train.AdamOptimizer(self.learning_rate).minimize(
                self.loss, var_list=all_tvars, colocate_gradients_with_ops=True, global_step=self.global_step)

        # checkpoints
        print("loading from checkpoint ...")
        self.sess.run(tf.global_variables_initializer())
        checkpoint_path = os.path.join(
            self.opts.log_dir, 'model-'+str(self.opts.test_epoch))
        var_to_restore = [
            val for val in tf.global_variables() if ('/Adam' not in val.name) and ('encoder' in val.name or 'explicit_decoder' in val.name or 'implicit_decoder' in val.name)]
        saver = tf.train.Saver(var_to_restore)
        saver.restore(self.sess, checkpoint_path)
        step = self.sess.run(self.global_step)

        # test
        result_pkl_list = glob.glob(
            'data/segmentation_results/{}/results_*.pkl'.format(self.opts.dataset))
        result_pkl_list = sorted(result_pkl_list)
        n_image = len(result_pkl_list)
        print('no. of test images: {}\n'.format(n_image))

        for i, path in tqdm(enumerate(result_pkl_list)):
            with open(path, 'rb') as f:
                data = cPickle.load(f)
            image_path = data['image_path']
            num_instance = len(data['pred_class_ids'])

            image_path = os.path.join(
                '/data/linjiehong/posed3Ddet/NOCS_CVPR2019-master/', image_path)

            image = cv2.imread(image_path + '_color.png')[:, :, :3]
            image = image[:, :, ::-1]
            depth = load_depth(image_path)
            pred_mask = data['pred_masks']

            pred_RTs = np.zeros((num_instance, 4, 4))
            pred_scales = np.zeros((num_instance, 3))
            pred_RTs[:, 3, 3] = 1

            if num_instance != 0:
                for j in range(num_instance):
                    saver.restore(self.sess, checkpoint_path)  # re-load model

                    inst_mask = 255 * pred_mask[:, :, j].astype('uint8')
                    pts_ori, idx = backproject(
                        depth, self.intrinsics, inst_mask)
                    pts_ori = pts_ori/1000.0
                    rgb_ori = image[idx[0], idx[1], :]
                    rgb_ori = (
                        rgb_ori - np.array([123.7, 116.8, 103.9])[np.newaxis, :])/255.0

                    # test
                    dis = np.zeros((pts_ori.shape[0]))
                    for k in range(3):
                        pts = pts_ori[dis < 0.6].copy()
                        rgb = rgb_ori[dis < 0.6].copy()

                        if pts.shape[0] > 32:
                            centroid = np.mean(pts, axis=0)
                            pts = pts - centroid[np.newaxis, :]

                            input_dis, input_rgb = pc2sphericalmap(
                                pts, rgb, resolution=self.opts.input_res)
                            if pts.shape[0] > 1024:
                                input_pts = pts[np.random.choice(
                                    pts.shape[0], 1024, replace=False), :]
                            else:
                                input_pts = pts[np.random.choice(
                                    pts.shape[0], 1024), :]
                            feed_dict = {self.input_dis: input_dis, self.input_rgb: input_rgb,
                                         self.observed_pc: input_pts[np.newaxis, :, :], self.is_training: False}
                            iter_rotation, iter_translation, iter_size, iter_loss = self.sess.run(
                                [self.pred_rotation, self.pred_translation, self.pred_scale, self.loss], feed_dict=feed_dict)

                            pred_rotation = quat2mat(iter_rotation[0])
                            pred_translation = iter_translation[0] + centroid
                            pred_scale = np.linalg.norm(iter_size[0])
                            pred_size = iter_size[0]/pred_scale
                            pred_loss = iter_loss

                            pred_canonical_pts = (
                                (pts_ori - pred_translation[np.newaxis, :])/pred_scale) @ np.transpose(pred_rotation)
                            dis = np.linalg.norm(pred_canonical_pts, axis=1)

                        else:
                            break

                    # refinement
                    if pts.shape[0] > 32:
                        for k in range(self.opts.refine_iteration):
                            _, iter_rotation, iter_translation, iter_size, iter_loss = self.sess.run(
                                [self.all_optimizers, self.pred_rotation, self.pred_translation, self.pred_scale, self.loss], feed_dict=feed_dict)
                            if iter_loss < pred_loss:
                                pred_rotation = quat2mat(iter_rotation[0])
                                pred_translation = iter_translation[0] + centroid
                                pred_scale = np.linalg.norm(iter_size[0])
                                pred_size = iter_size[0]/pred_scale
                                pred_loss = iter_loss
                            if pred_loss <= self.opts.refine_threshold:
                                break

                    if pts_ori.shape[0] > 32:
                        pred_RTs[j, :3, :3] = np.diag(
                            np.ones((3))*pred_scale) @ pred_rotation.transpose()
                        pred_RTs[j, :3, 3] = pred_translation
                        pred_scales[j] = pred_size

                        z_180_RT = np.zeros((4, 4), dtype=np.float32)
                        z_180_RT[:3, :3] = np.diag([-1, -1, 1])
                        z_180_RT[3, 3] = 1
                        pred_RTs[j, :, :] = z_180_RT @ pred_RTs[j, :, :]
                    else:
                        pred_RTs[j] = np.eye(4)
                        pred_scales[j] = np.ones((3))

            data.pop('pred_masks')
            data['pred_RTs'] = pred_RTs
            data['pred_scales'] = pred_scales

            with open(os.path.join(self.opts.test_log_dir, path.split('/')[-1]), 'wb') as f:
                cPickle.dump(data, f)

    def test_refine_feature(self):
        print(
            '\n*********** Testing & Refining on {} ***********'.format(self.opts.dataset))

        # inputs
        self.input_dis = tf.placeholder(
            tf.float32, shape=[1, self.opts.input_res, self.opts.input_res, 1])
        self.input_rgb = tf.placeholder(
            tf.float32, shape=[1, self.opts.input_res, self.opts.input_res, 3])
        self.observed_pc = tf.placeholder(tf.float32, shape=[1, 1024, 3])
        self.new_pose_feat = tf.Variable(tf.zeros([1,1024]), trainable=True, name='new_pose_feat')
        self.is_training = tf.placeholder_with_default(
            False, shape=[], name='is_training')
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        # modules
        self.encoder = encoder(self.opts, self.is_training, name='encoder')
        self.explicit_decoder = explicit_decoder(
            self.opts, self.is_training, name='explicit_decoder')
        self.implicit_decoder = implicit_decoder(
            self.opts, self.is_training, name="implicit_decoder")

        # graphs
        self.pose_feat = self.encoder(self.input_dis, self.input_rgb)
        self.pred_translation0, self.pred_rotation0, self.pred_scale0 = self.explicit_decoder(
            self.pose_feat)
        self.ex_canonical_points0 = point_transformation(
            self.observed_pc, self.pred_rotation0, self.pred_translation0, self.pred_scale0)
        self.im_canonical_points0 = self.implicit_decoder(
            self.observed_pc, self.pose_feat)

        # refine_graphs
        self.pred_translation, self.pred_rotation, self.pred_scale = self.explicit_decoder(
            self.new_pose_feat)
        self.ex_canonical_points = point_transformation(
            self.observed_pc, self.pred_rotation, self.pred_translation, self.pred_scale)
        self.im_canonical_points = self.implicit_decoder(
            self.observed_pc, self.new_pose_feat)

        # self.adaptive loss
        self.loss = tf.losses.huber_loss(
            self.ex_canonical_points, self.im_canonical_points)

        # optimizer
        self.learning_rate = tf.train.exponential_decay(
            self.opts.refine_learning_rate, self.global_step, 100000000, 0, staircase=True)
        self.all_optimizers = tf.train.AdamOptimizer(self.opts.refine_f_learning_rate).minimize(
            self.loss, var_list=[self.new_pose_feat], colocate_gradients_with_ops=True, global_step=self.global_step)

        # checkpoints
        print("loading from checkpoint ...")
        self.sess.run(tf.global_variables_initializer())
        checkpoint_path = os.path.join(
            self.opts.log_dir, 'model-'+str(self.opts.test_epoch))
        var_to_restore = [
            val for val in tf.global_variables() if ('/Adam' not in val.name) and ('encoder' in val.name or 'explicit_decoder' in val.name or 'implicit_decoder' in val.name)]
        saver = tf.train.Saver(var_to_restore)
        saver.restore(self.sess, checkpoint_path)
        step = self.sess.run(self.global_step)

        # test
        result_pkl_list = glob.glob(
            'data/segmentation_results/{}/results_*.pkl'.format(self.opts.dataset))
        result_pkl_list = sorted(result_pkl_list)
        n_image = len(result_pkl_list)
        print('no. of test images: {}\n'.format(n_image))

        for i, path in tqdm(enumerate(result_pkl_list)):
            with open(path, 'rb') as f:
                data = cPickle.load(f)
            image_path = data['image_path']
            num_instance = len(data['pred_class_ids'])

            image_path = os.path.join(
                '/data/linjiehong/posed3Ddet/NOCS_CVPR2019-master/', image_path)

            image = cv2.imread(image_path + '_color.png')[:, :, :3]
            image = image[:, :, ::-1]
            depth = load_depth(image_path)
            pred_mask = data['pred_masks']

            pred_RTs = np.zeros((num_instance, 4, 4))
            pred_scales = np.zeros((num_instance, 3))
            pred_RTs[:, 3, 3] = 1

            if num_instance != 0:
                for j in range(num_instance):
                    inst_mask = 255 * pred_mask[:, :, j].astype('uint8')
                    pts_ori, idx = backproject(
                        depth, self.intrinsics, inst_mask)
                    pts_ori = pts_ori/1000.0
                    rgb_ori = image[idx[0], idx[1], :]
                    rgb_ori = (
                        rgb_ori - np.array([123.7, 116.8, 103.9])[np.newaxis, :])/255.0

                    # test
                    dis = np.zeros((pts_ori.shape[0]))
                    for k in range(3):
                        pts = pts_ori[dis < 0.6].copy()
                        rgb = rgb_ori[dis < 0.6].copy()

                        if pts.shape[0] > 32:
                            centroid = np.mean(pts, axis=0)
                            pts = pts - centroid[np.newaxis, :]

                            input_dis, input_rgb = pc2sphericalmap(
                                pts, rgb, resolution=self.opts.input_res)
                            if pts.shape[0] > 1024:
                                input_pts = pts[np.random.choice(
                                    pts.shape[0], 1024, replace=False), :]
                            else:
                                input_pts = pts[np.random.choice(
                                    pts.shape[0], 1024), :]
                            feed_dict = {self.input_dis: input_dis, self.input_rgb: input_rgb,
                                         self.observed_pc: input_pts[np.newaxis, :, :], self.is_training: False}
                            iter_rotation, iter_translation, iter_size = self.sess.run(
                                [self.pred_rotation0, self.pred_translation0, self.pred_scale0], feed_dict=feed_dict)

                            pred_rotation = quat2mat(iter_rotation[0])
                            pred_translation = iter_translation[0] + centroid
                            pred_scale = np.linalg.norm(iter_size[0])
                            pred_size = iter_size[0]/pred_scale

                            pred_canonical_pts = (
                                (pts_ori - pred_translation[np.newaxis, :])/pred_scale) @ np.transpose(pred_rotation)
                            dis = np.linalg.norm(pred_canonical_pts, axis=1)

                        else:
                            break

                    # refinement
                    if pts.shape[0] > 32:
                        pose_feat = self.sess.run([self.pose_feat], feed_dict=feed_dict)
                        update = tf.assign(self.new_pose_feat, pose_feat[0])
                        self.sess.run(update)
                        pred_loss = np.inf

                        for k in range(self.opts.refine_f_iteration):
                            _, iter_rotation, iter_translation, iter_size, iter_loss = self.sess.run(
                                [self.all_optimizers, self.pred_rotation, self.pred_translation, self.pred_scale, self.loss], feed_dict=feed_dict)
                            if iter_loss < pred_loss:
                                pred_rotation = quat2mat(iter_rotation[0])
                                pred_translation = iter_translation[0] + centroid
                                pred_scale = np.linalg.norm(iter_size[0])
                                pred_size = iter_size[0]/pred_scale
                                pred_loss = iter_loss
                            if pred_loss <= self.opts.refine_f_threshold:
                                break

                    if pts_ori.shape[0] > 32:
                        pred_RTs[j, :3, :3] = np.diag(
                            np.ones((3))*pred_scale) @ pred_rotation.transpose()
                        pred_RTs[j, :3, 3] = pred_translation
                        pred_scales[j] = pred_size

                        z_180_RT = np.zeros((4, 4), dtype=np.float32)
                        z_180_RT[:3, :3] = np.diag([-1, -1, 1])
                        z_180_RT[3, 3] = 1
                        pred_RTs[j, :, :] = z_180_RT @ pred_RTs[j, :, :]
                    else:
                        pred_RTs[j] = np.eye(4)
                        pred_scales[j] = np.ones((3))

            data.pop('pred_masks')
            data['pred_RTs'] = pred_RTs
            data['pred_scales'] = pred_scales

            with open(os.path.join(self.opts.test_log_dir, path.split('/')[-1]), 'wb') as f:
                cPickle.dump(data, f)
