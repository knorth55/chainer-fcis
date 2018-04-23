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
from chainercv.links.model.ssd import GradientScaling
import chainermn
import cupy as cp
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

    def __init__(self, model, min_size, max_size, flip=True):
        self.model = model
        self.min_size = min_size
        self.max_size = max_size
        self.flip = flip

    def __call__(self, in_data):
        orig_img, bboxes, whole_mask, labels = in_data
        _, orig_H, orig_W = orig_img.shape
        img = self.model.prepare(
            orig_img, self.min_size, self.max_size)
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
    parser.add_argument('--out', '-o', default=None)
    parser.add_argument('--config', default=None)
    parser.add_argument('--resume', default=None)
    args = parser.parse_args()

    # gpu communicator
    comm = chainermn.create_communicator('hierarchical')
    device = comm.intra_rank
    chainer.cuda.get_device_from_id(device).use()

    # out
    out = args.out
    if out is None:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        out = osp.join(filepath, 'out', timestamp)

    # config
    cfgpath = args.config
    if cfgpath is None:
        cfgpath = osp.join(filepath, 'cfg', 'train.yaml')
    with open(cfgpath, 'r') as f:
        config = easydict.EasyDict(yaml.load(f))

    if comm.rank == 0:
        os.makedirs(out)
        shutil.copy(cfgpath, osp.join(out, 'train.yaml'))

    min_size = config.min_size
    max_size = config.max_size
    random_seed = config.random_seed
    if 'max_epoch' in config:
        max_epoch = config.max_epoch
        max_iter = None
    else:
        max_epoch = None
        max_iter = config.max_iter
    lr = config.lr
    if 'cooldown_epoch' in config:
        cooldown_epoch = config.cooldown_epoch
        cooldown_iter = None
    else:
        cooldown_epoch = None
        cooldown_iter = config.cooldown_iter
    lr = config.lr
    lr_cooldown_factor = config.lr_cooldown_factor

    # set random seed
    np.random.seed(random_seed)
    cp.random.seed(random_seed)

    # model
    n_class = len(voc_label_names)
    fcis_model = fcis.models.FCISResNet101(
        n_class,
        ratios=(0.5, 1.0, 2.0),
        anchor_scales=(8, 16, 32),
        rpn_min_size=16)
    if args.resume is None:
        fcis_model.extractor.init_weight()
    else:
        chainer.serializers.load_npz(args.resume, fcis_model)
    model = fcis.models.FCISTrainChain(
        fcis_model,
        n_sample=128,
        bg_iou_thresh_lo=0.1)
    model.to_gpu()

    # optimizer
    optimizer = chainermn.create_multi_node_optimizer(
        chainer.optimizers.MomentumSGD(lr=lr, momentum=0.9),
        comm)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(rate=0.0005))

    # disable update
    model.fcis.extractor.res1.disable_update(True, True)
    model.fcis.extractor.res2.disable_update(True, True)
    model.fcis.extractor.res3.disable_update(False, True)
    model.fcis.extractor.res4.disable_update(False, True)
    model.fcis.extractor.res5.disable_update(False, True)

    # psroi_conv1 lr
    model.fcis.head.psroi_conv1.W.update_rule.add_hook(GradientScaling(3.0))
    model.fcis.head.psroi_conv1.b.update_rule.add_hook(GradientScaling(3.0))

    # dataset
    if comm.rank == 0:
        if config.use_sbd:
            dataset_class = SBDInstanceSegmentationDataset
        else:
            dataset_class = VOCInstanceSegmentationDataset
        train_dataset = dataset_class(split='train')
        test_dataset = dataset_class(split='val')

        train_dataset = TransformDataset(
            train_dataset,
            Transform(model.fcis, min_size, max_size))
        test_dataset = TransformDataset(
            test_dataset,
            Transform(model.fcis, min_size, max_size, flip=False))
    else:
        train_dataset = None
        test_dataset = None

    train_dataset = chainermn.scatter_dataset(
        train_dataset, comm, shuffle=True)
    test_dataset = chainermn.scatter_dataset(
        test_dataset, comm, shuffle=False)

    # iterator
    train_iter = chainer.iterators.SerialIterator(
        train_dataset, batch_size=1)
    test_iter = chainer.iterators.SerialIterator(
        test_dataset, batch_size=1, repeat=False, shuffle=False)
    updater = chainer.training.updater.StandardUpdater(
        train_iter, optimizer, converter=fcis.dataset.concat_examples,
        device=device)

    # interval
    if max_epoch is not None:
        max_interval = max_epoch, 'epoch'
    else:
        max_interval = max_iter, 'iteration'

    if cooldown_epoch is not None:
        cooldown_interval = cooldown_epoch, 'epoch'
    else:
        cooldown_interval = cooldown_iter, 'iteration'

    save_interval = 1, 'epoch'
    log_interval = 100, 'iteration'
    print_interval = 20, 'iteration'
    test_interval = 8, 'epoch'

    # trainer
    trainer = chainer.training.Trainer(
        updater, max_interval, out=out)

    # lr scheduler
    trainer.extend(
        chainer.training.extensions.ExponentialShift(
            'lr', lr_cooldown_factor, init=lr),
        trigger=chainer.training.triggers.ManualScheduleTrigger(
            *cooldown_interval))

    # evaluator
    trainer.extend(
        chainermn.create_multi_node_evaluator(
            chainer.training.extensions.Evaluator(
                test_iter, model, converter=fcis.dataset.concat_examples,
                device=device),
            comm),
        trigger=test_interval)

    # logging
    if comm.rank == 0:
        snapshot_filename = '{}_model_iter_{{.updater.iteration}}.npz'.format(
            model.fcis.__class__.__name__)

        trainer.extend(
            chainer.training.extensions.snapshot_object(
                model.fcis,
                savefun=chainer.serializers.save_npz,
                filename=snapshot_filename),
            trigger=save_interval)
        trainer.extend(
            chainer.training.extensions.observe_lr(),
            trigger=log_interval)
        trainer.extend(
            chainer.training.extensions.LogReport(
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
        trainer.extend(
            chainer.training.extensions.ProgressBar(update_interval=10))
        trainer.extend(
            chainer.training.extensions.dump_graph('main/loss'))

    trainer.run()

    if comm.rank == 0:
        print('log is saved in {}'.format(out))


if __name__ == '__main__':
    main()
