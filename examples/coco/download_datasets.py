#!/usr/bin/env python

import argparse
from fcis.datasets.coco.coco_utils import get_coco
import os.path as osp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--data-dir', default=None,
        help='Dataset Destination: default $HOME/data/datasets/coco')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--val', action='store_true')
    parser.add_argument('--minival', action='store_true')
    parser.add_argument('--valminusminival', action='store_true')
    parser.add_argument('--all', action='store_true')
    args = parser.parse_args()

    data_dir = args.data_dir
    if data_dir is None:
        data_dir = osp.expanduser('~/data/datasets/coco')

    if args.train or args.all:
        print('Downloading train datasets')
        get_coco('train', 'train', data_dir)
    if args.val or args.all:
        print('Downloading val datasets')
        get_coco('val', 'val', data_dir)
    if args.minival or args.all:
        print('Downloading minival datasets')
        get_coco('minival', 'val', data_dir)
    if args.valminusminival or args.all:
        print('Downloading valminusminival datasets')
        get_coco('valminusminival', 'val', data_dir)


if __name__ == '__main__':
    main()
