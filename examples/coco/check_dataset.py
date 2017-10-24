#!/usr/bin/env python

import chainercv
# import cv2
import fcis
from fcis.datasets.coco import COCOInstanceSegmentationDataset
import numpy as np
from train import get_keep_indices
from train import remove_zero_bbox


def check(dataset, model, i, target_height, max_width):
    img_id = dataset.ids[i]
    bboxes, masks, labels, _, _ = dataset._get_annotations(i)

    orig_H = dataset.img_props[img_id]['height']
    orig_W = dataset.img_props[img_id]['width']
    resize_scale = fcis.utils.get_resize_scale(
        (orig_H, orig_W), target_height, max_width)
    resize_scale = fcis.utils.get_resize_scale(
        (orig_H, orig_W), target_height, max_width)
    H = int(round(resize_scale * orig_H))
    W = int(round(resize_scale * orig_W))
    scale = H / orig_H

    bboxes = chainercv.transforms.resize_bbox(
        bboxes, (orig_H, orig_W), (H, W))

    indices = get_keep_indices(bboxes)
    if len(indices) != len(bboxes):
        print(indices)
    bboxes = bboxes[indices, :]
    labels = labels[indices]

    assert len(bboxes) == len(labels)
    assert len(bboxes) > 0

    for bbox in bboxes:
        bbox = np.round(bbox).astype(np.int32)
        mask_height = bbox[2] - bbox[0]
        mask_width = bbox[3] - bbox[1]

        if mask_height == 0 or mask_width == 0:
            print('i: {}'.format(i))
            print('labels: {}'.format(labels))
            print('scale: {}'.format(scale))
            print('bboxes')
            print(bboxes)
            print('bbox')
            print(bbox)


def main():
    target_height = 600
    max_width = 1000

    print('preparing')
    print('train_dataset load')
    train_dataset = COCOInstanceSegmentationDataset(split='train')
    print('train_dataset remove zero bbox')
    train_dataset = remove_zero_bbox(train_dataset, target_height, max_width)
    print('model load')
    model = fcis.models.FCISResNet101()
    print('finish preparing')

    print('check running')

    print('checking train dataset')
    for i in range(0, len(train_dataset)):
        check(train_dataset, model, i, target_height, max_width)
        if i % 5000 == 0:
            print('checking {}'.format(i))
    print('finish checking train dataset')


if __name__ == '__main__':
    main()
