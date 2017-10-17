#!/usr/bin/env python

import argparse
import chainer
import cupy
import cv2
from easydict import EasyDict
import fcis
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as osp
import yaml


filepath = osp.abspath(osp.dirname(__file__))
mean_bgr = np.array([103.06, 115.90, 123.15])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default=0)
    parser.add_argument('-m', '--modelpath', default=None)
    args = parser.parse_args()

    # chainer config for demo
    gpu = args.gpu
    chainer.cuda.get_device_from_id(gpu).use()
    chainer.global_config.train = False
    chainer.global_config.enable_backprop = False

    cfgpath = osp.join(filepath, 'cfg', 'demo.yaml')
    with open(cfgpath, 'r') as f:
        config = EasyDict(yaml.load(f))

    # load config
    target_height = config.target_height
    max_width = config.max_width
    score_thresh = config.score_thresh
    nms_thresh = config.nms_thresh
    mask_merge_thresh = config.mask_merge_thresh
    binary_thresh = config.binary_thresh

    label_yamlpath = osp.join(filepath, 'cfg', config.label_yaml)
    with open(label_yamlpath, 'r') as f:
        label_names = yaml.load(f)

    n_class = len(label_names) + 1
    model = fcis.models.FCISResNet101(n_class)

    modelpath = args.modelpath
    if modelpath is None:
        modelpath = model.download()
    chainer.serializers.load_npz(modelpath, model)
    model.to_gpu(gpu)

    imagedir = osp.join(filepath, 'images')
    image_names = sorted(os.listdir(imagedir))

    for image_name in image_names:
        imagepath = osp.join(imagedir, image_name)
        print('image: {}'.format(imagepath))
        orig_img = cv2.imread(
            imagepath, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        orig_H, orig_W, _ = orig_img.shape
        resize_scale = target_height / float(orig_H)
        if orig_W * resize_scale > max_width:
            resize_scale = max_width / float(orig_W)
        img = cv2.resize(
            orig_img, None, None,
            fx=resize_scale, fy=resize_scale,
            interpolation=cv2.INTER_LINEAR)
        scale = img.shape[1] / float(orig_W)

        # inference
        x_data = img.copy()
        x_data = x_data.astype(np.float32)
        x_data -= mean_bgr
        x_data = x_data.transpose((2, 0, 1))  # H, W, C -> C, H, W
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
        rois[:, 0::2] = cupy.clip(rois[:, 0::2], 0, orig_H)
        rois[:, 1::2] = cupy.clip(rois[:, 1::2], 0, orig_W)

        # voting
        # cpu voting is only implemented
        rois = chainer.cuda.to_cpu(rois)
        roi_cls_probs = chainer.cuda.to_cpu(roi_cls_probs)
        roi_mask_probs = chainer.cuda.to_cpu(roi_mask_probs)

        v_labels, v_masks, v_bboxes, v_cls_probs = fcis.mask.mask_voting(
            rois, roi_cls_probs, roi_mask_probs, n_class, orig_H, orig_W,
            score_thresh, nms_thresh, mask_merge_thresh, binary_thresh)

        fcis.utils.visualize_mask(
            orig_img[:, :, ::-1], v_labels, v_masks, v_bboxes,
            v_cls_probs, label_names, binary_thresh)
        plt.show()


if __name__ == '__main__':
    main()
