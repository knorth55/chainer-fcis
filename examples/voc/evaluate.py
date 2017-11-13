#!/usr/bin/env python

import argparse
import chainer
from easydict import EasyDict
import numpy as np
import os.path as osp
import time
import yaml

import fcis
from fcis.datasets.voc.voc_utils import voc_label_names
from fcis.evaluations.eval_instance_segmentation_voc import \
    eval_instance_segmentation_voc


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
        config = EasyDict(yaml.load(f))

    target_height = config.target_height
    max_width = config.max_width
    score_thresh = 1e-3
    nms_thresh = config.nms_thresh
    mask_merge_thresh = config.mask_merge_thresh
    binary_thresh = config.binary_thresh
    min_drop_size = config.min_drop_size
    iter2 = True

    # load label_names
    n_class = len(voc_label_names)

    # load model
    model = fcis.models.FCISResNet101(
        n_class,
        ratios=(0.5, 1.0, 2.0),
        anchor_scales=(8, 16, 32),
        rpn_min_size=16)
    modelpath = args.modelpath
    if modelpath is None:
        modelpath = model.download('voc')
    chainer.serializers.load_npz(modelpath, model)
    model.to_gpu(gpu)

    dataset = fcis.datasets.sbd.SBDInstanceSegmentationDataset(split='val')

    sizes = list()
    pred_bboxes = list()
    pred_masks = list()
    pred_labels = list()
    pred_scores = list()
    gt_bboxes = list()
    gt_masks = list()
    gt_labels = list()

    print('start')
    start = time.time()
    for i in range(0, len(dataset)):
        img, gt_bbox, gt_whole_mask, gt_label = dataset[i]
        _, H, W = img.shape
        gt_whole_mask = gt_whole_mask.astype(bool)
        gt_mask = fcis.utils.whole_mask2mask(
            gt_whole_mask, gt_bbox)
        del gt_whole_mask
        sizes.append((H, W))
        gt_bboxes.append(gt_bbox)
        gt_masks.append(gt_mask)
        gt_labels.append(gt_label)

        # prediction
        outputs = model.predict(
            [img], target_height, max_width, score_thresh,
            nms_thresh, mask_merge_thresh, binary_thresh,
            min_drop_size, iter2=iter2)
        del img
        pred_bbox = outputs[0][0]
        pred_whole_mask = outputs[1][0]
        pred_whole_mask = pred_whole_mask.astype(bool)
        pred_mask = fcis.utils.whole_mask2mask(
            pred_whole_mask, pred_bbox)
        del pred_whole_mask
        pred_bboxes.append(pred_bbox)
        pred_masks.append(pred_mask)
        pred_labels.append(outputs[2][0])
        pred_scores.append(outputs[3][0])

        if (i + 1) % 100 == 0:
            print('{} / {}, avg iter/sec={:.2f}'.format(
                i, len(dataset), (i + 1) / (time.time() - start)))

    for iou_thresh in (0.5, 0.7):
        results = eval_instance_segmentation_voc(
            sizes, pred_bboxes, pred_masks, pred_labels, pred_scores,
            gt_bboxes, gt_masks, gt_labels, None,
            iou_thresh=iou_thresh, use_07_metric=False)

        print('================================')
        print('iou_thresh={}'.format(iou_thresh))
        print('map@{}={}'.format(iou_thresh, results['map']))
        for i, label_name in enumerate(voc_label_names):
            if i == 0:
                continue
            try:
                print('ap@{}/{:s}={}'.format(
                    iou_thresh, label_name, results['ap'][i]))
            except IndexError:
                print('ap@{}/{:s}={}'.format(
                    iou_thresh, label_name, np.nan))
        print('================================')


if __name__ == '__main__':
    main()
