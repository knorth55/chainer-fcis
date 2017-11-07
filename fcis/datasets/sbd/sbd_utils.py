import os.path as osp

from chainer.dataset import download
from chainercv import utils


root = 'pfnet/chainercv/sbd'
url = 'http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz'  # NOQA


def get_sbd(data_dir=None):
    if data_dir is None:
        data_dir = download.get_dataset_directory(root)
    base_path = osp.join(data_dir, 'benchmark_RELEASE/dataset')
    if osp.exists(base_path):
        return base_path

    download_file_path = utils.cached_download(url)
    ext = osp.splitext(url)[1]
    utils.extractall(download_file_path, data_dir, ext)
    return base_path
