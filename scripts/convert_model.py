#!/usr/bin/env python

import chainer
import fcis
import os.path as osp

import _init_paths  # NOQA

from utils.load_model import load_param


filepath = osp.abspath(osp.dirname(__file__))


def main():
    arg_params, aux_params = load_param(
        filepath + '/../model/fcis_coco', 0, process=True)

    n_class = 81

    model = fcis.models.FCISResNet101(n_class)

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
        # ResNetC1
        if name.startswith('conv1'):
            assert model.res1.conv1.W.data.shape == value.asnumpy().shape, name
            model.res1.conv1.W.data = value.asnumpy()
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
            res = getattr(model, res_name)
            bottle = getattr(res, bottle_name)
            layer = getattr(bottle, conv_branch[branch_name])
            assert layer.W.data.shape == value.asnumpy().shape, name
            layer.W.data = value.asnumpy()
        # RPN
        elif name.startswith('rpn'):
            _, layer_name, _, data_type = name.split('_')
            v = value.asnumpy()
            if layer_name == 'conv':
                layer = model.rpn.conv1
            elif layer_name == 'cls':
                layer = model.rpn.score
            elif layer_name == 'bbox':
                layer = model.rpn.loc

            if data_type == 'weight':
                assert layer.W.data.shape == v.shape, name
                if layer_name == 'cls':
                    v = v.reshape((2, -1, 512, 1, 1))
                    v = v.transpose((1, 0, 2, 3, 4))
                    v = v.reshape((-1, 512, 1, 1))
                elif layer_name == 'bbox':
                    v = v.reshape((-1, 4, 512, 1, 1))
                    v = v[:, [1, 0, 3, 2]]
                    v = v.reshape((-1, 512, 1, 1))
                layer.W.data = v
            elif data_type == 'bias':
                assert layer.b.data.shape == v.shape, name
                if layer_name == 'cls':
                    v = v.reshape((2, -1))
                    v = v.transpose((1, 0))
                    v = v.reshape((-1,))
                elif layer_name == 'bbox':
                    v = v.reshape((-1, 4))
                    v = v[:, [1, 0, 3, 2]]
                    v = v.reshape((-1,))
                layer.b.data = v
        # psroi_conv1
        elif name.startswith('conv_new'):
            data_type = name.split('_')[3]
            layer = model.psroi_conv1
            if data_type == 'weight':
                assert layer.W.data.shape == value.asnumpy().shape, name
                layer.W.data = value.asnumpy()
            elif data_type == 'bias':
                assert layer.b.data.shape == value.asnumpy().shape, name
                layer.b.data = value.asnumpy()
        # psroi_conv2
        elif name.startswith('fcis_cls_seg'):
            data_type = name.split('_')[3]
            layer = model.psroi_conv2
            if data_type == 'weight':
                assert layer.W.data.shape == value.asnumpy().shape, name
                layer.W.data = value.asnumpy()
            elif data_type == 'bias':
                assert layer.b.data.shape == value.asnumpy().shape, name
                layer.b.data = value.asnumpy()
        # psroi_conv3
        elif name.startswith('fcis_bbox'):
            data_type = name.split('_')[2]
            layer = model.psroi_conv3
            if data_type == 'weight':
                assert layer.W.data.shape == value.asnumpy().shape, name
                layer.W.data = value.asnumpy()
            elif data_type == 'bias':
                assert layer.b.data.shape == value.asnumpy().shape, name
                layer.b.data = value.asnumpy()
        else:
            layer_name, branch_name, data_type = name.split('_')
            if layer_name == 'bn':
                layer = model.res1.bn1
                if data_type == 'beta':
                    assert layer.beta.data.shape == value.asnumpy().shape
                    layer.beta.data = value.asnumpy()
                elif data_type == 'gamma':
                    assert layer.gamma.data.shape == value.asnumpy().shape
                    layer.gamma.data = value.asnumpy()
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
                res = getattr(model, res_name)
                bottle = getattr(res, bottle_name)
                layer = getattr(bottle, bn_branch[branch_name])
                if data_type == 'beta':
                    assert layer.beta.data.shape == value.asnumpy().shape, name
                    layer.beta.data = value.asnumpy()
                elif data_type == 'gamma':
                    assert layer.gamma.shape == value.asnumpy().shape, name
                    layer.gamma.data = value.asnumpy()

    for name, value in aux_params.items():
        layer_name, branch_name, _, data_type = name.split('_')
        if layer_name == 'bn':
            layer = model.res1.bn1
            if data_type == 'var':
                assert layer.avg_var.shape == value.asnumpy().shape, name
                layer.avg_var = value.asnumpy()
            elif data_type == 'mean':
                assert layer.avg_mean.shape == value.asnumpy().shape, name
                layer.avg_mean = value.asnumpy()
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
            res = getattr(model, res_name)
            bottle = getattr(res, bottle_name)
            layer = getattr(bottle, bn_branch[branch_name])
            if data_type == 'var':
                assert layer.avg_var.shape == value.asnumpy().shape, name
                layer.avg_var = value.asnumpy()
            elif data_type == 'mean':
                assert layer.avg_mean.shape == value.asnumpy().shape, name
                layer.avg_mean = value.asnumpy()

    chainer.serializers.save_npz('./fcis_coco.npz', model)


if __name__ == '__main__':
    main()
