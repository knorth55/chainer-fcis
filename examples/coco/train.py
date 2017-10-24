#!/usr/bin/env python
from __future__ import division

import argparse
import chainer
from chainer.datasets import TransformDataset
import chainercv
import cv2
import datetime
import easydict
import fcis
from fcis.datasets.coco.coco_utils import coco_label_names
from fcis.datasets.coco import COCOInstanceSegmentationDataset
from fcis.extensions import InstanceSegmentationCOCOEvaluator
import numpy as np
import os
import os.path as osp
import yaml


filepath = osp.abspath(osp.dirname(__file__))


def remove_zero_bbox(dataset, target_height, max_width):
    remove_ids = []

    for i in range(0, len(dataset)):
        img_id = dataset.ids[i]
        bboxes, _, labels, _, _ = dataset._get_annotations(i)
        if len(bboxes) == 0:
            remove_ids.append(img_id)
        if len(labels) == 0:
            remove_ids.append(img_id)

        orig_H = dataset.img_props[img_id]['height']
        orig_W = dataset.img_props[img_id]['width']
        resize_scale = fcis.utils.get_resize_scale(
            (orig_H, orig_W), target_height, max_width)
        H = int(round(resize_scale * orig_H))
        W = int(round(resize_scale * orig_W))

        resized_bboxes = chainercv.transforms.resize_bbox(
            bboxes, (orig_H, orig_W), (H, W))
        resized_bboxes = np.round(resized_bboxes).astype(np.int32)

        # check if there is too small bbox
        shapes = (resized_bboxes[:, 2:] - resized_bboxes[:, :2])
        masked_shapes = [shape for shape in shapes if shape.min() > 0]
        if len(masked_shapes) == 0:
            remove_ids.append(img_id)

    remove_ids = list(set(remove_ids))
    dataset.ids = [i for i in dataset.ids if i not in remove_ids]
    return dataset


def get_keep_indices(bboxes):
    indices = []
    for i, bbox in enumerate(bboxes):
        bbox = np.round(bbox).astype(np.int32)
        mask_height = bbox[2] - bbox[0]
        mask_width = bbox[3] - bbox[1]
        if mask_height > 0 and mask_width > 0:
            indices.append(i)
    return np.array(indices, dtype=np.int32)


class Transform(object):

    def __init__(self, model, target_height, max_width):
        self.model = model
        self.target_height = target_height
        self.max_width = max_width

    def __call__(self, in_data):
        orig_img, bboxes, masks, labels = in_data
        _, orig_H, orig_W = orig_img.shape
        img = self.model.prepare(
            orig_img, self.target_height, self.max_width)
        _, H, W = img.shape
        scale = H / orig_H

        bboxes = chainercv.transforms.resize_bbox(
            bboxes, (orig_H, orig_W), (H, W))

        indices = get_keep_indices(bboxes)
        resized_masks = []
        for i, (bbox, mask) in enumerate(zip(bboxes, masks)):
            if i not in indices:
                continue
            bbox = np.round(bbox).astype(np.int32)
            mask_height = bbox[2] - bbox[0]
            mask_width = bbox[3] - bbox[1]
            resized_mask = cv2.resize(
                mask.astype(np.int32),
                (mask_width, mask_height),
                interpolation=cv2.INTER_NEAREST)
            resized_masks.append(resized_mask)
        bboxes = bboxes[indices, :]
        labels = labels[indices]

        whole_masks = fcis.utils.mask2whole_mask(
            resized_masks, bboxes, (H, W))

        img, params = chainercv.transforms.random_flip(
            img, x_random=True, return_param=True)
        whole_masks = fcis.utils.flip_mask(
            whole_masks, x_flip=params['x_flip'])
        bboxes = chainercv.transforms.flip_bbox(
            bboxes, (H, W), x_flip=params['x_flip'])

        return img, bboxes, whole_masks, labels, scale


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default=0)
    parser.add_argument('--out', '-o', default=None)
    parser.add_argument('--config', default=None)
    args = parser.parse_args()

    # gpu
    gpu = args.gpu
    chainer.cuda.get_device_from_id(gpu).use()

    # out
    out = args.out
    if out is None:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        out = osp.join(filepath, 'out', timestamp)
        os.makedirs(out)

    # config
    cfgpath = args.config
    if cfgpath is None:
        cfgpath = osp.join(filepath, 'cfg', 'train.yaml')
    with open(cfgpath, 'r') as f:
        config = easydict.EasyDict(yaml.load(f))

    target_height = config.target_height
    max_width = config.max_width
    random_seed = config.random_seed
    lr = float(config.lr)
    # lr_step_epoch = config.lr_step_epoch
    max_epoch = config.max_epoch

    # set random seed
    np.random.seed(random_seed)

    # dataset
    train_dataset = COCOInstanceSegmentationDataset(split='train')
    train_dataset = remove_zero_bbox(train_dataset, target_height, max_width)
    test_dataset = COCOInstanceSegmentationDataset(split='val')
    # lr_step_size = int(round(lr_step_epoch * len(train_dataset)))

    # model
    n_class = len(coco_label_names)
    fcis_model = fcis.models.FCISResNet101(n_class)
    fcis_model.init_weight()
    model = fcis.models.FCISTrainChain(fcis_model)
    model.to_gpu()

    # optimizer
    optimizer = chainer.optimizers.MomentumSGD(lr=lr, momentum=0.9)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(rate=0.0005))

    # disable update
    model.fcis.res1.disable_update(True, True)
    model.fcis.res2.disable_update(True, True)
    model.fcis.res3.disable_update(False, True)
    model.fcis.res4.disable_update(False, True)
    model.fcis.res5.disable_update(False, True)

    # psroi_conv1 lr
    update_rule = chainer.optimizers.momentum_sgd.MomentumSGDRule(
        lr=optimizer.lr * 3.0, momentum=0.9)
    model.fcis.psroi_conv1.W.update_rule = update_rule
    model.fcis.psroi_conv1.b.update_rule = update_rule

    train_dataset = TransformDataset(
        train_dataset,
        Transform(model.fcis, target_height, max_width))

    # iterator
    train_iter = chainer.iterators.SerialIterator(
        train_dataset, batch_size=1)
    test_iter = chainer.iterators.SerialIterator(
        test_dataset, batch_size=1, repeat=False, shuffle=False)
    updater = chainer.training.updater.StandardUpdater(
        train_iter, optimizer, device=gpu)

    trainer = chainer.training.Trainer(
        updater, (max_epoch, 'epoch'), out=out)

    # trainer.extend(
    #     chainer.training.extensions.ExponentialShift('lr', 0.1),
    #     trigger=(lr_step_size, 'iteration'))

    # interval
    save_interval = 10000, 'iteration'
    log_interval = 20, 'iteration'
    plot_interval = 3000, 'iteration'
    print_interval = 20, 'iteration'
    test_interval = 8, 'epoch'

    # logging
    model_name = model.fcis.__class__.__name__
    trainer.extend(
        chainer.training.extensions.snapshot(
            savefun=chainer.serializers.save_npz,
            filename='%s_trainer_iter_{.updater.iteration}.npz' % model_name),
        trigger=save_interval)

    trainer.extend(
        chainer.training.extensions.snapshot_object(
            model.fcis,
            savefun=chainer.serializers.save_npz,
            filename='%s_model_iter_{.updater.iteration}.npz' % model_name),
        trigger=save_interval)
    trainer.extend(chainer.training.extensions.observe_lr(),
                   trigger=log_interval)
    trainer.extend(chainer.training.extensions.LogReport(
        log_name='log.json',
        trigger=log_interval))
    trainer.extend(chainer.training.extensions.PrintReport([
        'iteration',
        'epoch',
        'elapsed_time',
        'lr',
        'main/loss',
        'main/rpn_loc_loss',
        'main/rpn_cls_loss',
        'main/fcis_loc_loss',
        'main/fcis_cls_loss',
        'main/fcis_mask_loss',
        'validation/main/mAP[0.50:0.95]',
    ]), trigger=print_interval)
    trainer.extend(chainer.training.extensions.ProgressBar(update_interval=10))

    if chainer.training.extensions.PlotReport.available():
        trainer.extend(
            chainer.training.extensions.PlotReport(
                ['main/loss'],
                file_name='loss.png', trigger=plot_interval),
            trigger=plot_interval)

    trainer.extend(
        InstanceSegmentationCOCOEvaluator(
            test_iter, model.fcis,
            coco_label_names),
        trigger=test_interval)

    trainer.extend(chainer.training.extensions.dump_graph('main/loss'))

    trainer.run()


if __name__ == '__main__':
    main()
