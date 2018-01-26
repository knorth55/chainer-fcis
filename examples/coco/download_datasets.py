#!/usr/bin/env python

import argparse
from fcis.datasets.coco.coco_utils import get_coco
import os.path as osp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--data-dir', default=None,
        help='Dataset Destination: default $HOME/data/datasets/coco')
    parser.add_argument('--train2014', action='store_true')
    parser.add_argument('--val2014', action='store_true')
    parser.add_argument('--minival2014', action='store_true')
    parser.add_argument('--valminusminival2014', action='store_true')
    parser.add_argument('--test2014', action='store_true')
    parser.add_argument('--test2015', action='store_true')
    parser.add_argument('--all', action='store_true')
    args = parser.parse_args()

    data_dir = args.data_dir
    if data_dir is None:
        data_dir = osp.expanduser('~/data/datasets/coco')

    if args.train2014 or args.all:
        print('Downloading train2014 datasets')
        get_coco('train2014', 'train2014', data_dir)
    if args.val2014 or args.all:
        print('Downloading val2014 datasets')
        get_coco('val2014', 'val2014', data_dir)
    if args.minival2014 or args.all:
        print('Downloading minival2014 datasets')
        get_coco('minival2014', 'val2014', data_dir)
    if args.valminusminival2014 or args.all:
        print('Downloading valminusminival2014 datasets')
        get_coco('valminusminival2014', 'val2014', data_dir)
    if args.test2014 or args.all:
        print('Downloading test2014 datasets')
        get_coco('test2014', 'test2014', data_dir)
    if args.test2015 or args.all:
        print('Downloading test2015 datasets')
        get_coco('test2015', 'test2015', data_dir)


if __name__ == '__main__':
    main()
