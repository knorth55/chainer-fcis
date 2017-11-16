#!/usr/bin/env python
from __future__ import division

import argparse
import datetime
import easydict
import numpy as np
import os
import os.path as osp
import shutil
import yaml

import chainer
from chainer.datasets import TransformDataset
import chainercv
import cupy
import cv2

import fcis
from fcis.datasets.sbd import SBDInstanceSegmentationDataset
from fcis.datasets.voc.voc_utils import voc_label_names
from fcis.datasets.voc import VOCInstanceSegmentationDataset


filepath = osp.abspath(osp.dirname(__file__))


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

    def __init__(self, model, target_height, max_width, flip=True):
        self.model = model
        self.target_height = target_height
        self.max_width = max_width
        self.flip = flip

    def __call__(self, in_data):
        orig_img, bboxes, whole_mask, labels = in_data
        _, orig_H, orig_W = orig_img.shape
        img = self.model.prepare(
            orig_img, self.target_height, self.max_width)
        img = img.astype(np.float32)
        del orig_img
        _, H, W = img.shape
        scale = H / orig_H

        bboxes = chainercv.transforms.resize_bbox(
            bboxes, (orig_H, orig_W), (H, W))

        indices = get_keep_indices(bboxes)
        bboxes = bboxes[indices, :]
        whole_mask = whole_mask[indices, :, :]
        labels = labels[indices]

        whole_mask = whole_mask.transpose((1, 2, 0))
        whole_mask = cv2.resize(
            whole_mask.astype(np.uint8), (W, H),
            interpolation=cv2.INTER_NEAREST)
        if whole_mask.ndim < 3:
            whole_mask = whole_mask.reshape((H, W, 1))
        whole_mask = whole_mask.transpose((2, 0, 1))

        if self.flip:
            img, params = chainercv.transforms.random_flip(
                img, x_random=True, return_param=True)
            whole_mask = fcis.utils.flip_mask(
                whole_mask, x_flip=params['x_flip'])
            bboxes = chainercv.transforms.flip_bbox(
                bboxes, (H, W), x_flip=params['x_flip'])

        return img, bboxes, whole_mask, labels, scale


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default=0)
    parser.add_argument('--out', '-o', default=None)
    parser.add_argument('--config', default=None)
    parser.add_argument('--resume', default=None)
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

    shutil.copy(cfgpath, osp.join(out, 'train.yaml'))

    target_height = config.target_height
    max_width = config.max_width
    random_seed = config.random_seed
    max_epoch = config.max_epoch
    lr = config.lr
    cooldown_epoch = config.cooldown_epoch
    lr = config.lr
    lr_cooldown_factor = config.lr_cooldown_factor

    # set random seed
    np.random.seed(random_seed)
    cupy.random.seed(random_seed)

    # dataset
    if config.use_sbd:
        dataset_class = SBDInstanceSegmentationDataset
    else:
        dataset_class = VOCInstanceSegmentationDataset
    train_dataset = dataset_class(split='train')
    test_dataset = dataset_class(split='val')

    # model
    n_class = len(voc_label_names)
    fcis_model = fcis.models.FCISResNet101(
        n_class,
        ratios=(0.5, 1.0, 2.0),
        anchor_scales=(8, 16, 32),
        rpn_min_size=16)
    if args.resume is None:
        fcis_model.init_weight()
    else:
        chainer.serializers.load_npz(args.resume, fcis_model)
    model = fcis.models.FCISTrainChain(
        fcis_model,
        n_sample=128,
        bg_iou_thresh_lo=0.1)
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
        lr=lr * 3.0, momentum=0.9)
    model.fcis.psroi_conv1.W.update_rule = update_rule
    model.fcis.psroi_conv1.b.update_rule = update_rule

    train_dataset = TransformDataset(
        train_dataset,
        Transform(model.fcis, target_height, max_width))
    test_dataset = TransformDataset(
        test_dataset,
        Transform(model.fcis, target_height, max_width, flip=False))

    # iterator
    train_iter = chainer.iterators.SerialIterator(
        train_dataset, batch_size=1)
    test_iter = chainer.iterators.SerialIterator(
        test_dataset, batch_size=1, repeat=False, shuffle=False)
    updater = chainer.training.updater.StandardUpdater(
        train_iter, optimizer, converter=fcis.dataset.concat_examples,
        device=gpu)

    trainer = chainer.training.Trainer(
        updater, (max_epoch, 'epoch'), out=out)

    # lr scheduler
    trainer.extend(
        chainer.training.extensions.ExponentialShift('lr', lr_cooldown_factor),
        trigger=chainer.training.triggers.ManualScheduleTrigger(
            [cooldown_epoch], 'epoch'))

    # interval
    save_interval = 1, 'epoch'
    log_interval = 100, 'iteration'
    print_interval = 20, 'iteration'
    test_interval = 8, 'epoch'

    # logging
    model_name = model.fcis.__class__.__name__

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
        'main/rpn_acc',
        'main/fcis_cls_acc',
        'main/fcis_fg_acc',
        'validation/main/rpn_acc',
        'validation/main/fcis_cls_acc',
        'validation/main/fcis_fg_acc',
    ]), trigger=print_interval)
    trainer.extend(chainer.training.extensions.ProgressBar(update_interval=10))

    trainer.extend(
        chainer.training.extensions.Evaluator(
            test_iter, model, converter=fcis.dataset.concat_examples,
            device=gpu),
        trigger=test_interval)

    trainer.extend(chainer.training.extensions.dump_graph('main/loss'))

    trainer.run()


if __name__ == '__main__':
    main()
