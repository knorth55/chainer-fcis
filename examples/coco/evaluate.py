#!/usr/bin/env python

import argparse
import chainer
from easydict import EasyDict
import numpy as np
import os.path as osp
import time
import yaml

import fcis
from fcis.datasets.coco.coco_utils import coco_label_names
from fcis.evaluations.eval_instance_segmentation_coco import \
    eval_instance_segmentation_coco


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

    min_size = config.min_size
    max_size = config.max_size
    score_thresh = 1e-3
    nms_thresh = config.nms_thresh
    mask_merge_thresh = config.mask_merge_thresh
    binary_thresh = config.binary_thresh

    # load label_names
    n_class = len(coco_label_names)

    # load model
    model = fcis.models.FCISResNet101(n_class)
    modelpath = args.modelpath
    if modelpath is None:
        modelpath = model.download('coco')
    chainer.serializers.load_npz(modelpath, model)
    model.to_gpu(gpu)

    dataset = fcis.datasets.coco.COCOInstanceSegmentationDataset(
        split='minival2014', use_crowded=True,
        return_crowded=True, return_area=True)

    print('start')
    start = time.time()

    def inference_generator(model, dataset):
        for i in range(0, len(dataset)):
            img, gt_bbox, gt_whole_mask, gt_label, gt_crowded, gt_area = \
                dataset[i]
            _, H, W = img.shape
            size = (H, W)
            gt_mask = fcis.utils.whole_mask2mask(gt_whole_mask, gt_bbox)

            # prediction
            outputs = model.predict(
                [img], min_size, max_size, score_thresh,
                nms_thresh, mask_merge_thresh, binary_thresh)
            pred_bbox = outputs[0][0]
            pred_whole_mask = outputs[1][0]
            pred_mask = fcis.utils.whole_mask2mask(
                pred_whole_mask, pred_bbox)
            pred_label = outputs[2][0]
            pred_score = outputs[3][0]

            if i % 100 == 0:
                print('{} / {}, avg iter/sec={:.2f}'.format(
                    i, len(dataset), (i + 1) / (time.time() - start)))

            yield i, size, pred_bbox, pred_mask, pred_label, pred_score, \
                gt_bbox, gt_mask, gt_label, gt_crowded, gt_area

    generator = inference_generator(model, dataset)
    results = eval_instance_segmentation_coco(generator)

    keys = [
        'ap/iou=0.50:0.95/area=all/maxDets=100',
        'ap/iou=0.50/area=all/maxDets=100',
        'ap/iou=0.75/area=all/maxDets=100',
        'ap/iou=0.50:0.95/area=small/maxDets=100',
        'ap/iou=0.50:0.95/area=medium/maxDets=100',
        'ap/iou=0.50:0.95/area=large/maxDets=100',
    ]
    print('================================')
    for key in keys:
        print('m{}={}'.format(key, results['m' + key]))
        for i, label_name in enumerate(coco_label_names):
            if i == 0:
                continue
            try:
                print('{}/{:s}={}'.format(
                    key, label_name, results[key][i - 1]))
            except IndexError:
                print('{}/{:s}={}'.format(
                    key, label_name, np.nan))
        print('================================')


if __name__ == '__main__':
    main()
