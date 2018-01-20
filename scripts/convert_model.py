#!/usr/bin/env python

import argparse
import chainer
import fcis
import os.path as osp

import _init_paths  # NOQA

from utils.load_model import load_param


filepath = osp.abspath(osp.dirname(__file__))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset')
    args = parser.parse_args()

    if args.dataset == 'coco':
        n_class = 81
        model = fcis.models.FCISResNet101(n_class)
        prefix = filepath + '/../model/fcis_coco'
        epoch = 0
    elif args.dataset == 'voc':
        n_class = 21
        model = fcis.models.FCISResNet101(
            n_class,
            ratios=(0.5, 1.0, 2.0),
            anchor_scales=(8, 16, 32),
            rpn_min_size=16)
        prefix = filepath + '/../model/e2e'
        epoch = 21
    else:
        print('dataset must be coco or voc')
    arg_params, aux_params = load_param(
        prefix, epoch, process=True)
    model = convert(model, arg_params, aux_params)
    chainer.serializers.save_npz(
        './fcis_{}.npz'.format(args.dataset), model)


def convert(model, arg_params, aux_params):
    conv_branch = {
        'branch2a': 'conv1',
        'branch2b': 'conv2',
        'branch2c': 'conv3',
        'branch1': 'conv4',
    }

    bn_branch = {
        'branch2a': 'bn1',
        'branch2b': 'bn2',
        'branch2c': 'bn3',
        'branch1': 'bn4',
    }

    # convolution weight
    for name, value in arg_params.items():
        value = value.asnumpy()
        # ResNetC1
        if name.startswith('conv1'):
            value = value[:, ::-1, :, :]
            layer = model.extractor.res1.conv1
            assert layer.W.array.shape == value.shape, name
            layer.W.array = value
        # ResNetC2-5
        elif name.startswith('res'):
            block_name, branch_name, _ = name.split('_')
            res_name = block_name[:4]
            if block_name[4] == 'a':
                bottle_num = block_name[4:]
            elif block_name[4] == 'b':
                bottle_num = block_name[4:]
                if bottle_num == 'b':
                    bottle_num = 'b1'
            elif block_name[4] == 'c':
                bottle_num = 'b2'
            bottle_name = '{0}_{1}'.format(res_name, bottle_num)
            res = getattr(model.extractor, res_name)
            bottle = getattr(res, bottle_name)
            layer = getattr(bottle, conv_branch[branch_name])
            assert layer.W.array.shape == value.shape, name
            layer.W.array = value
        # RPN
        elif name.startswith('rpn'):
            _, layer_name, _, data_type = name.split('_')
            if layer_name == 'conv':
                layer = model.rpn.conv1
            elif layer_name == 'cls':
                layer = model.rpn.score
            elif layer_name == 'bbox':
                layer = model.rpn.loc

            if data_type == 'weight':
                if layer_name == 'cls':
                    value = value.reshape((2, -1, 512, 1, 1))
                    value = value.transpose((1, 0, 2, 3, 4))
                    value = value.reshape((-1, 512, 1, 1))
                elif layer_name == 'bbox':
                    value = value.reshape((-1, 4, 512, 1, 1))
                    value = value[:, [1, 0, 3, 2]]
                    value = value.reshape((-1, 512, 1, 1))
                assert layer.W.array.shape == value.shape, name
                layer.W.array = value
            elif data_type == 'bias':
                if layer_name == 'cls':
                    value = value.reshape((2, -1))
                    value = value.transpose((1, 0))
                    value = value.reshape((-1,))
                elif layer_name == 'bbox':
                    value = value.reshape((-1, 4))
                    value = value[:, [1, 0, 3, 2]]
                    value = value.reshape((-1,))
                assert layer.b.array.shape == value.shape, name
                layer.b.array = value
        # psroi_conv1
        elif name.startswith('conv_new'):
            data_type = name.split('_')[3]
            layer = model.head.psroi_conv1
            if data_type == 'weight':
                assert layer.W.array.shape == value.shape, name
                layer.W.array = value
            elif data_type == 'bias':
                assert layer.b.array.shape == value.shape, name
                layer.b.array = value
        # psroi_conv2
        elif name.startswith('fcis_cls_seg'):
            data_type = name.split('_')[3]
            layer = model.head.psroi_conv2
            if data_type == 'weight':
                assert layer.W.array.shape == value.shape, name
                layer.W.array = value
            elif data_type == 'bias':
                assert layer.b.array.shape == value.shape, name
                layer.b.array = value
        # psroi_conv3
        elif name.startswith('fcis_bbox'):
            data_type = name.split('_')[2]
            layer = model.head.psroi_conv3
            if data_type == 'weight':
                value = value.reshape((2, 4, 7 * 7, 1024, 1, 1))
                value = value[:, [1, 0, 3, 2]]
                value = value.reshape((-1, 1024, 1, 1))
                assert layer.W.array.shape == value.shape, name
                layer.W.array = value
            elif data_type == 'bias':
                value = value.reshape((2, 4, 7 * 7))
                value = value[:, [1, 0, 3, 2]]
                value = value.reshape((-1,))
                assert layer.b.array.shape == value.shape, name
                layer.b.array = value
        else:
            layer_name, branch_name, data_type = name.split('_')
            if layer_name == 'bn':
                layer = model.extractor.res1.bn1
                if data_type == 'beta':
                    assert layer.beta.array.shape == value.shape
                    layer.beta.array = value
                elif data_type == 'gamma':
                    assert layer.gamma.array.shape == value.shape
                    layer.gamma.array = value
            else:
                res_name = 'res{}'.format(layer_name[2])
                block_name = layer_name[3:]
                if block_name[0] == 'a':
                    bottle_num = block_name
                elif block_name[0] == 'b':
                    bottle_num = block_name
                    if bottle_num == 'b':
                        bottle_num = 'b1'
                elif block_name[0] == 'c':
                    bottle_num = 'b2'
                bottle_name = '{0}_{1}'.format(res_name, bottle_num)
                res = getattr(model.extractor, res_name)
                bottle = getattr(res, bottle_name)
                layer = getattr(bottle, bn_branch[branch_name])
                if data_type == 'beta':
                    assert layer.beta.array.shape == value.shape, name
                    layer.beta.array = value
                elif data_type == 'gamma':
                    assert layer.gamma.array.shape == value.shape, name
                    layer.gamma.array = value

    for name, value in aux_params.items():
        value = value.asnumpy()
        layer_name, branch_name, _, data_type = name.split('_')
        if layer_name == 'bn':
            layer = model.extractor.res1.bn1
            if data_type == 'var':
                assert layer.avg_var.shape == value.shape, name
                layer.avg_var = value
            elif data_type == 'mean':
                assert layer.avg_mean.shape == value.shape, name
                layer.avg_mean = value
        else:
            res_name = 'res{}'.format(layer_name[2])
            block_name = layer_name[3:]
            if block_name[0] == 'a':
                bottle_num = block_name
            elif block_name[0] == 'b':
                bottle_num = block_name
                if bottle_num == 'b':
                    bottle_num = 'b1'
            elif block_name[0] == 'c':
                bottle_num = 'b2'
            bottle_name = '{0}_{1}'.format(res_name, bottle_num)
            res = getattr(model.extractor, res_name)
            bottle = getattr(res, bottle_name)
            layer = getattr(bottle, bn_branch[branch_name])
            if data_type == 'var':
                assert layer.avg_var.shape == value.shape, name
                layer.avg_var = value
            elif data_type == 'mean':
                assert layer.avg_mean.shape == value.shape, name
                layer.avg_mean = value
    return model


if __name__ == '__main__':
    main()
