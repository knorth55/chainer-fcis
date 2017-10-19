#!/usr/bin/env python

import argparse
import chainer
from easydict import EasyDict
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
    parser.add_argument('--data-dir')
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
    score_thresh = 0.05
    nms_thresh = config.nms_thresh
    mask_merge_thresh = config.mask_merge_thresh
    binary_thresh = config.binary_thresh

    # load label_names
    n_class = len(coco_label_names)

    # load model
    model = fcis.models.FCISResNet101(n_class)
    modelpath = args.modelpath
    if modelpath is None:
        modelpath = model.download()
    chainer.serializers.load_npz(modelpath, model)
    model.to_gpu(gpu)

    dataset = fcis.datasets.coco.COCOInstanceSegmentationDataset(
        data_dir=args.data_dir, split='minival',
        use_crowded=True, return_crowded=True, return_area=True)

    sizes = list()
    pred_bboxes = list()
    pred_masks = list()
    pred_labels = list()
    pred_scores = list()
    gt_bboxes = list()
    gt_masks = list()
    gt_labels = list()
    gt_crowdeds = list()
    gt_areas = list()

    print('start')
    start = time.time()
    for i in range(len(dataset)):
        img, gt_bbox, gt_mask, gt_label, gt_crowded, gt_area = dataset[i]
        _, H, W = img.shape
        sizes.append((H, W))
        gt_bboxes.append(gt_bbox)
        gt_masks.append(gt_mask)
        gt_labels.append(gt_label)
        gt_crowdeds.append(gt_crowded)
        gt_areas.append(gt_area)

        # prediction
        outputs = model.predict(
            [img], target_height, max_width, score_thresh,
            nms_thresh, mask_merge_thresh, binary_thresh)
        pred_bboxes.append(outputs[0][0])
        pred_masks.append(outputs[1][0])
        pred_labels.append(outputs[2][0])
        pred_scores.append(outputs[3][0])

        if i % 100 == 0:
            print('{} / {},   avg speed={:.2f}s'.format(
                i, len(dataset), (time.time() - start) / (i + 1)))

    results = eval_instance_segmentation_coco(
        sizes, pred_bboxes, pred_masks, pred_labels, pred_scores,
        gt_bboxes, gt_masks, gt_labels, gt_crowdeds, gt_areas)

    keys = [
        'ap/iou=0.50:0.95/area=all/maxDets=100',
        'ap/iou=0.50/area=all/maxDets=100',
        'ap/iou=0.75/area=all/maxDets=100',
        'ap/iou=0.50:0.95/area=small/maxDets=100',
        'ap/iou=0.50:0.95/area=medium/maxDets=100',
        'ap/iou=0.50:0.95/area=large/maxDets=100',
    ]
    for key in keys:
        print('m{}={}'.format(key, results['m' + key]))


if __name__ == '__main__':
    main()
