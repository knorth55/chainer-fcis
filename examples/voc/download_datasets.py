#!/usr/bin/env python

import argparse
from fcis.datasets.sbd.sbd_utils import get_sbd
from fcis.datasets.voc.voc_utils import get_voc
import os.path as osp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--data-dir', default=None,
        help='Dataset Destination: default $HOME/data/datasets/VOC')
    parser.add_argument('--voc', action='store_true')
    parser.add_argument('--sbd', action='store_true')
    args = parser.parse_args()

    data_dir = args.data_dir
    if data_dir is None:
        data_dir = osp.expanduser('~/data/datasets/VOC')

    download_funcs = []
    if args.voc:
        download_funcs.append(get_voc)
    if args.sbd:
        download_funcs.append(get_sbd)

    for get_dataset in download_funcs:
        print('Downloading datasets')
        get_dataset(data_dir)


if __name__ == '__main__':
    main()
