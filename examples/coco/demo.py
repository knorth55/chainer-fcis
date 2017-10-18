#!/usr/bin/env python

import argparse
import chainer
import easydict
import fcis
import matplotlib.pyplot as plt
import os
import os.path as osp
import yaml


filepath = osp.abspath(osp.dirname(__file__))


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

    # load config
    cfgpath = osp.join(filepath, 'cfg', 'demo.yaml')
    with open(cfgpath, 'r') as f:
        config = easydict.EasyDict(yaml.load(f))

    target_height = config.target_height
    max_width = config.max_width
    score_thresh = config.score_thresh
    nms_thresh = config.nms_thresh
    mask_merge_thresh = config.mask_merge_thresh
    binary_thresh = config.binary_thresh

    # load label_names
    label_names = fcis.datasets.coco.coco_utils.coco_label_names
    n_class = len(label_names)

    # load model
    model = fcis.models.FCISResNet101(n_class)
    modelpath = args.modelpath
    if modelpath is None:
        modelpath = model.download()
    chainer.serializers.load_npz(modelpath, model)
    model.to_gpu(gpu)

    # load input images
    imgdir = osp.join(filepath, 'images')
    img_names = sorted(os.listdir(imgdir))
    imgpaths = [osp.join(imgdir, name) for name in img_names]
    orig_imgs = fcis.utils.read_images(imgpaths, channel_order='BGR')

    for orig_img in orig_imgs:
        # prediction
        masks, bboxes, labels, cls_probs = model.predict(
            [orig_img], target_height, max_width, score_thresh,
            nms_thresh, mask_merge_thresh, binary_thresh)

        # batch size = 1
        masks = masks[0]
        bboxes = bboxes[0]
        labels = labels[0]
        cls_probs = cls_probs[0]

        # visualization
        fcis.utils.visualize_mask(
            orig_img[:, :, ::-1], masks, bboxes, labels,
            cls_probs, label_names)
        plt.show()


if __name__ == '__main__':
    main()
