"""
    RANSAC for Similarity Transformation Estimation
    Modified from https://github.com/hughw19/NOCS_CVPR2019 & https://github.com/mentian/object-deformnet
    Originally Written by Srinath Sridhar
"""
import time
import numpy as np

from pc_utils import backproject, pc2sphericalmap


def estimateSimilarityUmeyama(SourceHom, TargetHom):
    # Copy of original paper is at: http://web.stanford.edu/class/cs273/refs/umeyama.pdf
    SourceCentroid = np.mean(SourceHom[:3, :], axis=1)
    TargetCentroid = np.mean(TargetHom[:3, :], axis=1)
    nPoints = SourceHom.shape[1]
    CenteredSource = SourceHom[:3, :] - np.tile(SourceCentroid, (nPoints, 1)).transpose()
    CenteredTarget = TargetHom[:3, :] - np.tile(TargetCentroid, (nPoints, 1)).transpose()
    CovMatrix = np.matmul(CenteredTarget, np.transpose(CenteredSource)) / nPoints
    if np.isnan(CovMatrix).any():
        print('nPoints:', nPoints)
        print(SourceHom.shape)
        print(TargetHom.shape)
        raise RuntimeError('There are NANs in the input.')

    U, D, Vh = np.linalg.svd(CovMatrix, full_matrices=True)
    d = (np.linalg.det(U) * np.linalg.det(Vh)) < 0.0
    if d:
        D[-1] = -D[-1]
        U[:, -1] = -U[:, -1]
    # rotation
    Rotation = np.matmul(U, Vh)
    # scale
    varP = np.var(SourceHom[:3, :], axis=1).sum()
    Scale = 1 / varP * np.sum(D)
    # translation
    Translation = TargetHom[:3, :].mean(axis=1) - SourceHom[:3, :].mean(axis=1).dot(Scale*Rotation.T)
    # transformation matrix
    OutTransform = np.identity(4)
    OutTransform[:3, :3] = Scale * Rotation
    OutTransform[:3, 3] = Translation

    return Scale, Rotation, Translation, OutTransform


def estimateSimilarityTransform(source: np.array, target: np.array, verbose=False):
    """ Add RANSAC algorithm to account for outliers.
    """
    assert source.shape[0] == target.shape[0], 'Source and Target must have same number of points.'
    SourceHom = np.transpose(np.hstack([source, np.ones([source.shape[0], 1])]))
    TargetHom = np.transpose(np.hstack([target, np.ones([target.shape[0], 1])]))
    # Auto-parameter selection based on source heuristics
    # Assume source is object model or gt nocs map, which is of high quality
    SourceCentroid = np.mean(SourceHom[:3, :], axis=1)
    nPoints = SourceHom.shape[1]
    CenteredSource = SourceHom[:3, :] - np.tile(SourceCentroid, (nPoints, 1)).transpose()
    SourceDiameter = 2 * np.amax(np.linalg.norm(CenteredSource, axis=0))
    InlierT = SourceDiameter / 10.0  # 0.1 of source diameter
    maxIter = 128
    confidence = 0.99

    if verbose:
        print('Inlier threshold: ', InlierT)
        print('Max number of iterations: ', maxIter)

    BestInlierRatio = 0
    BestInlierIdx = np.arange(nPoints)
    for i in range(0, maxIter):
        # Pick 5 random (but corresponding) points from source and target
        RandIdx = np.random.randint(nPoints, size=5)
        Scale, _, _, OutTransform = estimateSimilarityUmeyama(SourceHom[:, RandIdx], TargetHom[:, RandIdx])
        PassThreshold = Scale * InlierT    # propagate inlier threshold to target scale
        Diff = TargetHom - np.matmul(OutTransform, SourceHom)
        ResidualVec = np.linalg.norm(Diff[:3, :], axis=0)
        InlierIdx = np.where(ResidualVec < PassThreshold)[0]
        nInliers = InlierIdx.shape[0]
        InlierRatio = nInliers / nPoints
        # update best hypothesis
        if InlierRatio > BestInlierRatio:
            BestInlierRatio = InlierRatio
            BestInlierIdx = InlierIdx
        if verbose:
            print('Iteration: ', i)
            print('Inlier ratio: ', BestInlierRatio)
        # early break
        if (1 - (1 - BestInlierRatio ** 5) ** i) > confidence:
            break

    if(BestInlierRatio < 0.1):
        print('[ WARN ] - Something is wrong. Small BestInlierRatio: ', BestInlierRatio)
        return None, None, None, None

    SourceInliersHom = SourceHom[:, BestInlierIdx]
    TargetInliersHom = TargetHom[:, BestInlierIdx]
    Scale, Rotation, Translation, OutTransform = estimateSimilarityUmeyama(SourceInliersHom, TargetInliersHom)

    if verbose:
        print('BestInlierRatio:', BestInlierRatio)
        print('Rotation:\n', Rotation)
        print('Translation:\n', Translation)
        print('Scale:', Scale)

    return Scale, Rotation, Translation, OutTransform


def align_nocs_to_depth(masks, coords, depth, image, intrinsics, instance_ids, img_path, verbose=False):
    num_instances = len(instance_ids)
    error_messages = ''
    elapses = []

    instance_pts = np.zeros((num_instances, 1024, 3))
    dis_smaps = np.zeros((num_instances, 64, 64, 1))
    rgb_smaps = np.zeros((num_instances, 64, 64, 3))

    scales = np.zeros(num_instances)
    rotations = np.zeros((num_instances, 3, 3))
    translations = np.zeros((num_instances, 3))


    for i in range(num_instances):
        mask = masks[:, :, i]
        coord = coords[:, :, i, :]
        pts, idxs = backproject(depth, intrinsics, mask)
        coord_pts = coord[idxs[0], idxs[1], :] - 0.5
        try:
            start = time.time()
            s, R, T, outtransform = estimateSimilarityTransform(coord_pts, pts, False)
            elapsed = time.time() - start
            if verbose:
                print('elapsed: ', elapsed)
            elapses.append(elapsed)
        except Exception as e:
            message = '[ Error ] aligning instance {} in {} fails. Message: {}.'.format(instance_ids[i], img_path, str(e))
            print(message)
            error_messages += message + '\n'
            s = 1.0
            R = np.eye(3)
            T = np.zeros(3)
            outtransform = np.identity(4, dtype=np.float32)

        pts = pts / 1000.0
        centroid = np.mean(pts, axis=0)
        pts = pts - centroid[np.newaxis, :]
        T = T / 1000.0 - centroid

        img = image[idxs[0], idxs[1], :]
        img = (img-np.array([123.7, 116.8, 103.9])[np.newaxis, :])/255.0
        dis_map, rgb_map = pc2sphericalmap(
                                pts, img, resolution=64)

        if pts.shape[0]>1024:
            pts = pts[np.random.choice(
                pts.shape[0], 1024, replace=False), :]
        else:
            pts = pts[np.random.choice(
                pts.shape[0], 1024), :]

        instance_pts[i, :, :] = pts
        dis_smaps[i, :, :, :] = dis_map[0]
        rgb_smaps[i, :, :, :] = rgb_map[0]

        scales[i] = s / 1000.0
        rotations[i, :, :] = R.transpose()
        translations[i, :] = T

    return instance_pts, dis_smaps, rgb_smaps, scales, rotations, translations, error_messages, elapses