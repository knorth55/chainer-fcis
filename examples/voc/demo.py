#!/usr/bin/env python
import argparse
import datetime
import chainer
import os
import os.path as osp

import fcis


filepath = osp.abspath(osp.dirname(__file__))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default=0)
    parser.add_argument('-m', '--modelpath', default=None)
    parser.add_argument('--imgdir', default=None)
    args = parser.parse_args()

    # config path
    cfgpath = osp.join(filepath, 'cfg', 'demo.yaml')

    # chainer config for demo
    gpu = args.gpu
    chainer.cuda.get_device_from_id(gpu).use()
    chainer.global_config.train = False
    chainer.global_config.enable_backprop = False

    # load label_names
    label_names = fcis.datasets.voc.voc_utils.voc_label_names
    n_class = len(label_names)

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

    # load input images
    if args.imgdir is None:
        imgdir = osp.join(filepath, 'images')
    else:
        imgdir = args.imgdir
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    savepath = osp.join(filepath, 'vis_demo', timestamp)
    if not osp.exists(savepath):
        os.makedirs(savepath)

    fcis.utils.vis_demo(model, cfgpath, imgdir, label_names, savepath)


if __name__ == '__main__':
    main()
