#!/usr/bin/env python

import argparse
import chainer
from chainercv.links.model.faster_rcnn.utils.loc2bbox import loc2bbox
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

        rois = model.rois
        rois = rois / scale
        roi_locs = model.roi_locs.data
        roi_cls_probs = model.roi_cls_probs.data
        roi_seg_probs = model.roi_seg_probs.data

        mean = cupy.tile(cupy.asarray(model.loc_normalize_mean), 2)
        std = cupy.tile(cupy.asarray(model.loc_normalize_std), 2)
        roi_locs = (roi_locs * std + mean).astype(cupy.float32)
        roi_locs = roi_locs.reshape((-1, 2, 4))
        rois = cupy.broadcast_to(rois[:, None], roi_locs.shape)
        bboxes = loc2bbox(rois.reshape((-1, 4)),
                          roi_locs.reshape(-1, 4))
        bboxes = bboxes.reshape((-1, 2*4))
        bboxes[:, 0::2] = cupy.clip(bboxes[:, 0::2], 0, H)
        bboxes[:, 1::2] = cupy.clip(bboxes[:, 1::2], 0, W)


if __name__ == '__main__':
    main()
