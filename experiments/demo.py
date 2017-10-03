#!/usr/bin/env python

import argparse
import chainer
from chainercv.utils.bbox.bbox_iou import bbox_iou
from chainercv.utils import non_maximum_suppression
import cupy
import fcis
import numpy as np
import os
import os.path as osp
import scipy.misc
import yaml


filepath = osp.abspath(osp.dirname(__file__))
mean_bgr = np.array([103.06, 115.90, 123.15])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default=0)
    args = parser.parse_args()

    # chainer config for demo
    gpu = args.gpu
    chainer.cuda.get_device_from_id(gpu).use()
    chainer.global_config.train = False
    chainer.global_config.enable_backprop = False

    cfgpath = osp.join(filepath, 'cfg', 'demo.yaml')
    with open(cfgpath, 'r') as f:
        config = yaml.load(f)

    n_class = config['n_class']
    score_thresh = config['score_thresh']
    nms_thresh = config['nms_thresh']
    mask_merge_thresh = config['mask_merge_thresh']
    binary_thresh = config['binary_thresh']

    model = fcis.models.FCISResNet101(n_class)
    modelpath = osp.join(filepath, '../models/fcis_coco.npz')
    chainer.serializers.load_npz(modelpath, model)
    model.to_gpu(gpu)

    imagedir = osp.join(filepath, 'images')
    images = os.listdir(imagedir)
    for image_name in images:
        imagepath = osp.join(imagedir, image_name)
        img = scipy.misc.imread(imagepath)
        H, W, _ = img.shape
        img = scipy.misc.imresize(img, tuple(config['image_size']))
        scale = img.shape[1] / float(W)

        # inference
        x_data = img.copy()
        x_data = x_data.astype(np.float32)
        x_data = x_data[:, :, ::-1]  # RGB -> BGR
        x_data -= mean_bgr
        x_data = x_data.transpose(2, 0, 1)  # H, W, C -> C, H, W
        x = chainer.Variable(np.array([x_data], dtype=np.float32))
        x.to_gpu(gpu)
        model(x)

        # batch_size = 1
        rois = model.rois
        rois = rois / scale
        roi_cls_probs = model.roi_cls_probs
        roi_seg_probs = model.roi_seg_probs
        # shape: (n_rois, H, W)
        roi_mask_probs = roi_seg_probs[:, 1, :, :]

        # shape: (n_rois, 4)
        rois[:, 0::2] = cupy.clip(rois[:, 0::2], 0, H)
        rois[:, 1::2] = cupy.clip(rois[:, 1::2], 0, W)

        # voting
        # cpu voting is only implemented
        rois = chainer.cuda.to_cpu(rois)
        roi_cls_probs = chainer.cuda.to_cpu(roi_cls_probs)
        roi_mask_probs = chainer.cuda.to_cpu(roi_mask_probs)

        bboxes = []
        cls_probs = []
        mask_size = roi_mask_probs.shape[-1]

        for l in range(1, n_class):
            # shape: (n_rois,)
            roi_cls_probs_l = roi_cls_probs[:, l]
            thresh_mask = roi_cls_probs_l >= 0.001
            rois_l = rois[thresh_mask]
            roi_cls_probs_l = roi_cls_probs_l[thresh_mask]
            keep = non_maximum_suppression(
                rois_l, nms_thresh, roi_cls_probs_l, limit=100)
            bbox = rois_l[keep]
            bboxes.append(bbox)
            cls_probs.append(roi_cls_probs_l[keep])

        voted_masks = []
        voted_bboxes = []
        for l in range(0, n_class - 1):
            n_bboxes = len(bboxes[l])
            voted_mask = np.zeros((n_bboxes, mask_size, mask_size))
            voted_bbox = np.zeros((n_bboxes, 4))

            for i, bbox in enumerate(bboxes[l]):
                iou = bbox_iou(rois, bbox[np.newaxis, :])
                idx = np.where(iou > mask_merge_thresh)[0]
                mask_weights = roi_cls_probs[idx, l]
                mask_weights = mask_weights / mask_weights.sum()
                mask_probs_l = [roi_mask_probs[j] for j in idx.tolist()]
                rois_l = rois[idx]
                orig_mask, voted_bbox[i] = mask_aggregation(
                    rois_l, mask_probs_l, mask_weights, H, W, binary_thresh)
                voted_mask[i] = scipy.misc.imresize(
                    orig_mask.astype(np.float32), (mask_size, mask_size))

            cls_probs_l = cls_probs[l]
            score_thresh_mask = cls_probs_l > score_thresh
            voted_mask = voted_mask[score_thresh_mask]
            voted_bbox = voted_bbox[score_thresh_mask]
            voted_masks.append(voted_mask)
            voted_bboxes.append(voted_bbox)


def mask_aggregation(
        bboxes, mask_probs, mask_weights, H, W, binary_thresh):
    assert bboxes.shape[0] == len(mask_probs)
    assert bboxes.shape[0] == mask_weights.shape[0]
    mask = np.zeros((H, W))
    for bbox, mask_prob, mask_weight in zip(bboxes, mask_probs, mask_weights):
        bbox = np.round(bbox).astype(np.int)
        mask_prob = scipy.misc.imresize(
            mask_prob, (bbox[2] - bbox[0], bbox[3] - bbox[1]))
        mask_mask = (mask_prob >= binary_thresh).astype(np.float)
        mask[bbox[0]:bbox[2], bbox[1]:bbox[3]] += mask_mask * mask_weight

    y_idx, x_idx = np.where(mask >= binary_thresh)
    if len(y_idx) == 0 or len(x_idx) == 0:
        min_y = np.ceil(H / 2.0).astype(np.int)
        min_x = np.ceil(W / 2.0).astype(np.int)
        max_y = min_y
        max_x = min_x
    else:
        min_y = y_idx.min()
        min_x = x_idx.min()
        max_y = y_idx.max()
        max_x = x_idx.max()

    clipped_mask = mask[min_y:max_y+1, min_x:max_x+1]
    clipped_bbox = np.array((min_y, min_x, max_y, max_x), dtype=np.float32)
    return clipped_mask, clipped_bbox


if __name__ == '__main__':
    main()
