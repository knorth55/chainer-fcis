import os
import os.path as osp

from chainer.dataset import download
from chainercv import utils
import fcn


root = 'pfnet/chainercv/sbd'
url = 'http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz'  # NOQA
val_url = 'https://drive.google.com/uc?id=1Lz9QIiQ4v0Qt7uYqktsnO47ZEkU8DtV7'
train_url = 'https://drive.google.com/uc?id=1GN67nRRM-U9YwUPFuy2Ep2nZspDh7ZLA'


def get_sbd(data_dir=None):
    if data_dir is None:
        data_dir = download.get_dataset_directory(root)
    label_dir = osp.join(data_dir, 'fcis_label')
    if not osp.exists(label_dir):
        os.makedirs(label_dir)

    fcn.data.cached_download(
        url=val_url,
        path=osp.join(label_dir, 'val.txt'),
        md5='905db61182fcaaf6b981af6ae6dd7ff2'
    )
    fcn.data.cached_download(
        url=train_url,
        path=osp.join(label_dir, 'train.txt'),
        md5='79bff800c5f0b1ec6b21080a3c066722'
    )

    base_path = osp.join(data_dir, 'benchmark_RELEASE/dataset')
    if osp.exists(base_path):
        return base_path

    download_file_path = utils.cached_download(url)
    ext = osp.splitext(url)[1]
    utils.extractall(download_file_path, data_dir, ext)

    return base_path
