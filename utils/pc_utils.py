# Copyright (c) Gorilla Lab, SCUT.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Modified based on: https://github.com/hughw19/NOCS_CVPR2019.
"""

import cv2
import numpy as np

import math
import cmath


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


def backproject(depth, intrinsics, instance_mask):
    intrinsics_inv = np.linalg.inv(intrinsics)
    image_shape = depth.shape
    width = image_shape[1]
    height = image_shape[0]

    x = np.arange(width)
    y = np.arange(height)

    #non_zero_mask = np.logical_and(depth > 0, depth < 5000)
    non_zero_mask = (depth > 0)
    final_instance_mask = np.logical_and(instance_mask, non_zero_mask)

    idxs = np.where(final_instance_mask)
    grid = np.array([idxs[1], idxs[0]])

    # shape: height * width
    # mesh_grid = np.meshgrid(x, y) #[height, width, 2]
    # mesh_grid = np.reshape(mesh_grid, [2, -1])
    length = grid.shape[1]
    ones = np.ones([1, length])
    uv_grid = np.concatenate((grid, ones), axis=0) # [3, num_pixel]

    xyz = intrinsics_inv @ uv_grid # [3, num_pixel]
    xyz = np.transpose(xyz) #[num_pixel, 3]

    z = depth[idxs[0], idxs[1]]

    # print(np.amax(z), np.amin(z))
    pts = xyz * z[:, np.newaxis]/xyz[:, -1:]
    pts[:, 0] = -pts[:, 0]
    pts[:, 1] = -pts[:, 1]

    return pts, idxs


def pc2sphericalmap(pc, img, resolution=64):
    n = pc.shape[0]
    assert pc.shape[1] == 3

    pc = pc.astype(np.float32)
    x = pc[:, 0]
    y = pc[:, 1]
    z = pc[:, 2]
    t = np.pi / resolution
    k = 2 * np.pi / resolution
    r = np.sqrt(np.sum(pc ** 2, axis=1) + 1e-10).astype(np.float32)

    phi = np.around(np.arccos(z / r) / t).astype('int') % resolution
    arr = np.arctan2(y, x)
    rho = np.zeros(n)
    rho[y > 0] = np.around(arr[y > 0] / k)
    rho[y < 0] = np.around((arr[y < 0] + 2 * np.pi) / k)
    rho = rho.astype('int') % resolution
    f1 = np.zeros([resolution, resolution, 1], dtype='float32')
    f2 = np.zeros([resolution, resolution, 3], dtype='float32')

    for i in range(pc.shape[0]):
        tmp = np.real(cmath.rect(r[i], 0))
        if f1[rho[i], phi[i], 0] <= tmp:
            f1[rho[i], phi[i], 0] = tmp
            f2[rho[i], phi[i], :] = img[i]

    return f1[np.newaxis, :, :, :], f2[np.newaxis, :, :, :]