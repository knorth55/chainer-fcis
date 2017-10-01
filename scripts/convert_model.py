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

    bottlea_branch = {
        'branch1': 'conv1',
        'branch2a': 'conv2',
        'branch2b': 'conv3',
        'branch2c': 'conv4',
    }

    bottleb_branch = {
        'branch2a': 'conv1',
        'branch2b': 'conv2',
        'branch2c': 'conv3',
    }

    # # convolution weight
    for name, value in arg_params.items():
        # ResNetC1
        if name.startswith('conv1'):
            model.res1.conv1.W.data = value.asnumpy()
        # ResNetC2-5
        elif name.startswith('res'):
            block_name, branch_name, _ = name.split('_')
            res_name = block_name[:4]
            if block_name[4] == 'a':
                branch_dict = bottlea_branch
                bottle_num = block_name[4:]
            elif block_name[4] == 'b':
                branch_dict = bottleb_branch
                bottle_num = block_name[4:]
                if bottle_num == 'b':
                    bottle_num = 'b1'
            elif block_name[4] == 'c':
                branch_dict = bottleb_branch
                bottle_num = 'b2'
            bottle_name = '{0}_{1}'.format(res_name, bottle_num)
            res = getattr(model, res_name)
            bottle = getattr(res, bottle_name)
            layer = getattr(bottle, branch_dict[branch_name])
            layer.W.data = value.asnumpy()
        # RPN
        elif name.startswith('rpn'):
            _, layer_name, _, data_type = name.split('_')
            if layer_name == 'conv':
                layer = model.rpn.conv1
            elif layer_name == 'cls':
                layer = model.rpn.score
            else:  # layer_name == 'bbox'
                layer = model.rpn.loc
            if data_type == 'weight':
                layer.W.data = value.asnumpy()
            elif data_type == 'bias':
                layer.b.data = value.asnumpy()
        # psroi_conv1
        elif name.startswith('conv_new'):
            data_type = name.split('_')[3]
            layer = model.psroi_conv1
            if data_type == 'weight':
                layer.W.data = value.asnumpy()
            elif data_type == 'bias':
                layer.b.data = value.asnumpy()
        # psroi_conv2
        elif name.startswith('fcis_cls_seg'):
            data_type = name.split('_')[3]
            layer = model.psroi_conv2
            if data_type == 'weight':
                layer.W.data = value.asnumpy()
            elif data_type == 'bias':
                layer.b.data = value.asnumpy()
        # psroi_conv3
        elif name.startswith('fcis_bbox'):
            data_type = name.split('_')[2]
            layer = model.psroi_conv3
            if data_type == 'weight':
                layer.W.data = value.asnumpy()
            elif data_type == 'bias':
                layer.b.data = value.asnumpy()
        else:
            if not name.startswith('bn'):
                print(name)

    for name, value in aux_params.items():
        layer_name, branch_name, _, data_type = name.split('_')
        if layer_name == 'bn':
            layer = model.res1.bn1
            if data_type == 'var':
                layer.running_var = value.asnumpy()
            elif data_type == 'mean':
                layer.running_mean = value.asnumpy()
        else:
            res_name = 'res{}'.format(layer_name[2])
            block_name = layer_name[3:]
            if block_name[0] == 'a':
                branch_dict = bottlea_branch
                bottle_num = block_name
            elif block_name[0] == 'b':
                branch_dict = bottleb_branch
                bottle_num = block_name
                if bottle_num == 'b':
                    bottle_num = 'b1'
            elif block_name[0] == 'c':
                branch_dict = bottleb_branch
                bottle_num = 'b2'
            bottle_name = '{0}_{1}'.format(res_name, bottle_num)
            res = getattr(model, res_name)
            bottle = getattr(res, bottle_name)
            layer = getattr(bottle, branch_dict[branch_name])
            if data_type == 'var':
                layer.running_var = value.asnumpy()
            elif data_type == 'mean':
                layer.running_mean = value.asnumpy()

    chainer.serializers.save_npz('./fcis_coco.npz', model)


if __name__ == '__main__':
    main()
