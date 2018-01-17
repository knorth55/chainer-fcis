import numpy as np
import os.path as osp

from chainer.dataset import download
from chainercv import utils


root = 'pfnet/chainercv/voc'
url = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar'  # NOQA


def prepare_data(seg_img, ins_img):
    labels = []
    bboxes = []
    masks = []
    instances = np.unique(ins_img)
    for inst_id in instances[instances != -1]:
        mask_inst = ins_img == inst_id
        instance_class = np.unique(seg_img[mask_inst])[0]

        assert inst_id not in [-1]
        assert instance_class not in [-1, 0]

        where = np.argwhere(mask_inst)
        (y1, x1), (y2, x2) = where.min(0), where.max(0) + 1

        labels.append(instance_class)
        bboxes.append((y1, x1, y2, x2))
        masks.append(mask_inst)
    labels = np.array(labels)
    bboxes = np.array(bboxes)
    masks = np.array(masks)
    return bboxes, masks, labels


def get_voc(data_dir=None):
    if data_dir is None:
        data_dir = download.get_dataset_directory(root)
    base_path = osp.join(data_dir, 'VOCdevkit/VOC2012')
    if osp.exists(base_path):
        return base_path

    download_file_path = utils.cached_download(url)
    ext = osp.splitext(url)[1]
    utils.extractall(download_file_path, data_dir, ext)
    return base_path


voc_label_names = [
    '__background__',
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'potted plant',
    'sheep',
    'sofa',
    'train',
    'tv/monitor',
]
