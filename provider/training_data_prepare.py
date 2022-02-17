"""
    Modified from https://github.com/mentian/object-deformnet
"""

import os
import sys
import glob
import cv2
import math
import numpy as np
import _pickle as cPickle
from tqdm import tqdm
from autolab_core import RigidTransform

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, '..', 'utils'))
from align_utils import align_nocs_to_depth

sym_id = [0,1,3]
DATA_DIR = 'data'
OBJ_MODEL_DIR = os.path.join(DATA_DIR, 'obj_models')


def create_img_list(data_dir):
    """ Create train/val/test data list for CAMERA and Real. """
    # CAMERA dataset
    for subset in ['train', 'val']:
        img_list = []
        img_dir = os.path.join(data_dir, 'camera', subset)
        folder_list = [name for name in os.listdir(img_dir) if os.path.isdir(os.path.join(img_dir, name))]
        for i in range(10*len(folder_list)):
            folder_id = int(i) // 10
            img_id = int(i) % 10
            img_path = os.path.join(subset, '{:05d}'.format(folder_id), '{:04d}'.format(img_id))
            img_list.append(img_path)
        with open(os.path.join(data_dir, 'camera', subset+'_list_all.txt'), 'w') as f:
            for img_path in img_list:
                f.write("%s\n" % img_path)
    # Real dataset
    for subset in ['train', 'test']:
        img_list = []
        img_dir = os.path.join(data_dir, 'real', subset)
        folder_list = [name for name in sorted(os.listdir(img_dir)) if os.path.isdir(os.path.join(img_dir, name))]
        for folder in folder_list:
            img_paths = glob.glob(os.path.join(img_dir, folder, '*_color.png'))
            img_paths = sorted(img_paths)
            for img_full_path in img_paths:
                img_name = os.path.basename(img_full_path)
                img_ind = img_name.split('_')[0]
                img_path = os.path.join(subset, folder, img_ind)
                img_list.append(img_path)
        with open(os.path.join(data_dir, 'real', subset+'_list_all.txt'), 'w') as f:
            for img_path in img_list:
                f.write("%s\n" % img_path)
    print('Write all data paths to file done!')

def load_depth(img_path):
    """ Load depth image from img_path. """
    depth_path = img_path + '_depth.png'
    depth = cv2.imread(depth_path, -1)
    if len(depth.shape) == 3:
        # This is encoded depth image, let's convert
        # NOTE: RGB is actually BGR in opencv
        depth16 = depth[:, :, 1]*256 + depth[:, :, 2]
        depth16 = np.where(depth16==32001, 0, depth16)
        depth16 = depth16.astype(np.uint16)
    elif len(depth.shape) == 2 and depth.dtype == 'uint16':
        depth16 = depth
    else:
        assert False, '[ Error ]: Unsupported depth type.'
    return depth16


def process_data(img_path, depth, subset=None):
    """ Load instance masks for the objects in the image. """

    mask_path = img_path + '_mask.png'
    mask = cv2.imread(mask_path)[:, :, 2]
    mask = np.array(mask, dtype=np.int32)
    all_inst_ids = sorted(list(np.unique(mask)))
    assert all_inst_ids[-1] == 255
    del all_inst_ids[-1]    # remove background
    num_all_inst = len(all_inst_ids)
    h, w = mask.shape

    coord_path = img_path + '_coord.png'
    coord_map = cv2.imread(coord_path)[:, :, :3]
    coord_map = coord_map[:, :, (2, 1, 0)]
    # flip z axis of coord map
    coord_map = np.array(coord_map, dtype=np.float32) / 255
    coord_map[:, :, 2] = 1 - coord_map[:, :, 2]

    class_ids = []
    instance_ids = []
    model_list = []
    masks = np.zeros([h, w, num_all_inst], dtype=np.uint8)
    coords = np.zeros((h, w, num_all_inst, 3), dtype=np.float32)
    bboxes = np.zeros((num_all_inst, 4), dtype=np.int32)
    scales = np.zeros([num_all_inst, 3], dtype=np.float32)

    meta_path = img_path + '_meta.txt'
    with open(meta_path, 'r') as f:
        i = 0
        for line in f:
            line_info = line.strip().split(' ')
            inst_id = int(line_info[0])
            cls_id = int(line_info[1])
            # background objects and non-existing objects
            if cls_id == 0 or (inst_id not in all_inst_ids):
                continue

            if len(line_info) == 3:
                model_id = line_info[2]    # Real scanned objs
                if model_id[-3:] == 'npz':
                    npz_path = os.path.join(OBJ_MODEL_DIR, 'real_val', model_id)
                    with np.load(npz_path) as npz_file:
                        scale = npz_file['scale']
                else:
                    bbox_file = os.path.join(OBJ_MODEL_DIR, 'real_'+subset, model_id+'.txt')
                    scale = np.loadtxt(bbox_file)

                scales[i, :] = scale/ (np.linalg.norm(scale)+1e-10)
            else:
                model_id = line_info[3]    # CAMERA objs
                bbox_file = os.path.join(OBJ_MODEL_DIR, subset, line_info[2], line_info[3], 'bbox.txt')
                bbox = np.loadtxt(bbox_file)
                scales[i, :] = bbox[0, :] - bbox[1, :]

            # remove one mug instance in CAMERA train due to improper model
            if model_id == 'b9be7cfe653740eb7633a2dd89cec754':
                continue
            # process foreground objects
            inst_mask = np.equal(mask, inst_id)
            # bounding box
            horizontal_indicies = np.where(np.any(inst_mask, axis=0))[0]
            vertical_indicies = np.where(np.any(inst_mask, axis=1))[0]
            assert horizontal_indicies.shape[0], print(img_path)
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
            # object occupies full image, rendering error, happens in CAMERA dataset
            if np.any(np.logical_or((x2-x1) > 600, (y2-y1) > 440)):
                return None, None, None, None, None, None, None
            # not enough valid depth observation
            final_mask = np.logical_and(inst_mask, depth > 0)
            if np.sum(final_mask) < 64:
                continue
            class_ids.append(cls_id)
            instance_ids.append(inst_id)
            model_list.append(model_id)
            masks[:, :, i] = inst_mask
            coords[:, :, i, :] = np.multiply(coord_map, np.expand_dims(inst_mask, axis=-1))
            bboxes[i] = np.array([y1, x1, y2, x2])

            i += 1
    # no valid foreground objects
    if i == 0:
        return None, None, None, None, None, None, None

    masks = masks[:, :, :i]
    coords = np.clip(coords[:, :, :i, :], 0, 1)
    bboxes = bboxes[:i, :]
    scales = scales[:i, :]

    return masks, coords, class_ids, instance_ids, model_list, bboxes, scales


def annotate_camera_train(data_dir):
    """ Generate gt labels for CAMERA train data. """
    camera_train = open(os.path.join(data_dir, 'camera', 'train_list_all.txt')).read().splitlines()
    intrinsics = np.array([[577.5, 0, 319.5], [0, 577.5, 239.5], [0, 0, 1]])

    all_input_dis = []
    all_input_rgb = []
    all_observed_pc = []
    all_translation = []
    all_rotation = []
    all_scale = []

    part_counter = 1
    instance_counter = 0

    for i, img_path in enumerate(tqdm(camera_train)):
        img_full_path = os.path.join(data_dir, 'camera', img_path)
        all_exist = os.path.exists(img_full_path + '_color.png') and \
                    os.path.exists(img_full_path + '_coord.png') and \
                    os.path.exists(img_full_path + '_depth.png') and \
                    os.path.exists(img_full_path + '_mask.png') and \
                    os.path.exists(img_full_path + '_meta.txt')
        if not all_exist:
            continue

        image = cv2.imread(img_full_path + '_color.png')[:, :, :3]
        image = image[:, :, ::-1]
        depth = load_depth(img_full_path)
        masks, coords, class_ids, instance_ids, model_list, bboxes, sizes = process_data(img_full_path, depth, subset='train')
        if instance_ids is None:
            continue
        # Umeyama alignment of GT NOCS map with depth image
        pts, input_dis, input_rgb, scales, rotations, translations, error_messages, _ = \
            align_nocs_to_depth(masks, coords, depth, image, intrinsics, instance_ids, img_path)
        if error_messages:
            continue
        # write results
        quaternions = np.zeros((rotations.shape[0], 4))
        for k in range(pts.shape[0]):
            label = class_ids[k] - 1
            r = rotations[k]
            if label in sym_id:
                theta_x = r[0, 0] + r[2, 2]
                theta_y = r[0, 2] - r[2, 0]
                r_norm = math.sqrt(theta_x**2 + theta_y**2)
                s_map = np.array([[theta_x/r_norm, 0.0, -theta_y/r_norm],
                                    [0.0,            1.0,  0.0           ],
                                    [theta_y/r_norm, 0.0,  theta_x/r_norm]])
                r = s_map@r
            quaternions[k] = RigidTransform(rotation=r).quaternion

        all_input_dis.append(input_dis)
        all_input_rgb.append(input_rgb)
        all_observed_pc.append(pts)
        all_translation.append(translations)
        all_rotation.append(quaternions)
        all_scale.append(scales[:, np.newaxis] * sizes)

        instance_counter += pts.shape[0]
        if instance_counter >=80000 or i==len(camera_train)-1:
            dataset = {}
            dataset['input_dis'] = np.concatenate(all_input_dis, axis=0).astype(np.float32)
            dataset['input_rgb'] = np.concatenate(all_input_rgb, axis=0).astype(np.float32)
            dataset['observed_pc'] = np.concatenate(all_observed_pc, axis=0).astype(np.float32)
            dataset['translation'] = np.concatenate(all_translation, axis=0).astype(np.float32)
            dataset['rotation'] = np.concatenate(all_rotation, axis=0).astype(np.float32)
            dataset['scale'] = np.concatenate(all_scale, axis=0).astype(np.float32)

            with open(os.path.join(ROOT_DIR, '..', 'data', 'training_instance', 'CAMERA25_'+str(part_counter)+'.pkl'), 'wb') as f:
                cPickle.dump(dataset, f)

            all_input_dis = []
            all_input_rgb = []
            all_observed_pc = []
            all_translation = []
            all_rotation = []
            all_scale = []

            part_counter += 1
            instance_counter = 0


def annotate_real_train(data_dir):
    """ Generate gt labels for Real train data through PnP. """
    real_train = open(os.path.join(data_dir, 'real/train_list_all.txt')).read().splitlines()
    intrinsics = np.array([[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]])

    all_input_dis = []
    all_input_rgb = []
    all_observed_pc = []
    all_translation = []
    all_rotation = []
    all_scale = []

    for img_path in tqdm(real_train):
        img_full_path = os.path.join(data_dir, 'real', img_path)
        all_exist = os.path.exists(img_full_path + '_color.png') and \
                    os.path.exists(img_full_path + '_coord.png') and \
                    os.path.exists(img_full_path + '_depth.png') and \
                    os.path.exists(img_full_path + '_mask.png') and \
                    os.path.exists(img_full_path + '_meta.txt')
        if not all_exist:
            continue
        image = cv2.imread(img_full_path + '_color.png')[:, :, :3]
        image = image[:, :, ::-1]
        depth = load_depth(img_full_path)
        masks, coords, class_ids, instance_ids, model_list, bboxes, sizes = process_data(img_full_path, depth, 'train')
        if instance_ids is None:
            continue
        # Umeyama alignment of GT NOCS map with depth image
        pts, input_dis, input_rgb, scales, rotations, translations, error_messages, _ = \
            align_nocs_to_depth(masks, coords, depth, image, intrinsics, instance_ids, img_path)
        if error_messages:
            continue
        # write results
        quaternions = np.zeros((rotations.shape[0], 4))
        for k in range(pts.shape[0]):
            label = class_ids[k] - 1
            r = rotations[k]
            if label in sym_id:
                theta_x = r[0, 0] + r[2, 2]
                theta_y = r[0, 2] - r[2, 0]
                r_norm = math.sqrt(theta_x**2 + theta_y**2)
                s_map = np.array([[theta_x/r_norm, 0.0, -theta_y/r_norm],
                                    [0.0,            1.0,  0.0           ],
                                    [theta_y/r_norm, 0.0,  theta_x/r_norm]])
                r = s_map@r
            quaternions[k] = RigidTransform(rotation=r).quaternion

        all_input_dis.append(input_dis)
        all_input_rgb.append(input_rgb)
        all_observed_pc.append(pts)
        all_translation.append(translations)
        all_rotation.append(quaternions)
        all_scale.append(scales[:, np.newaxis] * sizes)

    dataset = {}
    dataset['input_dis'] = np.concatenate(all_input_dis, axis=0).astype(np.float32)
    dataset['input_rgb'] = np.concatenate(all_input_rgb, axis=0).astype(np.float32)
    dataset['observed_pc'] = np.concatenate(all_observed_pc, axis=0).astype(np.float32)
    dataset['translation'] = np.concatenate(all_translation, axis=0).astype(np.float32)
    dataset['rotation'] = np.concatenate(all_rotation, axis=0).astype(np.float32)
    dataset['scale'] = np.concatenate(all_scale, axis=0).astype(np.float32)

    with open(os.path.join(ROOT_DIR, '..', 'data', 'training_instance', 'REAL275.pkl'), 'wb') as f:
        cPickle.dump(dataset, f)


if __name__ == '__main__':

    if not os.path.exists(os.path.join(ROOT_DIR, '..', 'data', 'training_instance')):
        os.makedirs(os.path.join(ROOT_DIR, '..', 'data', 'training_instance'))

    # create list for all data
    create_img_list(DATA_DIR)
    # annotate dataset and re-write valid data to list
    annotate_camera_train(DATA_DIR)
    annotate_real_train(DATA_DIR)
