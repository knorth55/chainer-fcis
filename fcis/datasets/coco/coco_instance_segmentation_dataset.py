from collections import defaultdict
import json
import numpy as np
import os
import os.path as osp

import chainer

from chainercv import utils

from fcis.datasets.coco.coco_utils import coco_label_names
from fcis.datasets.coco.coco_utils import get_coco
from fcis.utils import visualize_mask
from fcis.utils import whole_mask2mask
import matplotlib.pyplot as plt

try:
    from pycocotools import mask as coco_mask
    _available = True
except ImportError:
    _available = False


class COCOInstanceSegmentationDataset(chainer.dataset.DatasetMixin):

    def __init__(self, data_dir=None, split='train',
                 use_crowded=False, return_crowded=False,
                 return_area=False):
        if not _available:
            raise ValueError(
                'Please install pycocotools\n'
                'pip install -e'
                '\'git+https://github.com/cocodataset/cocoapi.git'
                '#egg=pycocotools&subdirectory=PythonAPI\'')

        self.use_crowded = use_crowded
        self.return_crowded = return_crowded
        self.return_area = return_area
        if split in ['val', 'minival', 'valminusminival']:
            img_split = 'val'
        else:
            img_split = 'train'

        if data_dir is None:
            data_dir = osp.expanduser('~/data/datasets/coco')
        elif data_dir == 'auto':
            data_dir = get_coco(split, img_split)

        if not osp.exists(data_dir):
            raise ValueError(
                'Please download coco2014 dataset first')

        self.img_root = os.path.join(
            data_dir, '{}2014'.format(img_split))
        anno_fn = os.path.join(
            data_dir, 'annotations', 'instances_{}2014.json'.format(split))

        self.data_dir = data_dir
        anno = json.load(open(anno_fn, 'r'))

        self.img_props = dict()
        for img in anno['images']:
            self.img_props[img['id']] = img
        self.ids = list(self.img_props.keys())

        cats = anno['categories']
        self.cat_ids = [0] + [cat['id'] for cat in cats]

        self.anns = dict()
        self.imgToAnns = defaultdict(list)
        for ann in anno['annotations']:
            self.imgToAnns[ann['image_id']].append(ann)
            self.anns[ann['id']] = ann

    @property
    def labels(self):
        labels = list()
        for i in range(len(self)):
            label = self._get_annotations(i)[2]
            labels.append(label)
        return labels

    def _get_annotations(self, i):
        img_id = self.ids[i]
        # List[{'segmentation', 'area', 'iscrowd',
        #       'image_id', 'bbox', 'category_id', 'id'}]
        annotation = self.imgToAnns[img_id]
        H = self.img_props[img_id]['height']
        W = self.img_props[img_id]['width']
        bbox = np.array([ann['bbox'] for ann in annotation],
                        dtype=np.float32)
        if len(bbox) == 0:
            bbox = np.zeros((0, 4), dtype=np.float32)
        # (x, y, width, height)  -> (x_min, y_min, x_max, y_max)
        bbox[:, 2] = bbox[:, 0] + bbox[:, 2]
        bbox[:, 3] = bbox[:, 1] + bbox[:, 3]
        # (x_min, y_min, x_max, y_max) -> (y_min, x_min, y_max, x_max)
        bbox = bbox[:, [1, 0, 3, 2]]
        label = np.array([self.cat_ids.index(ann['category_id'])
                          for ann in annotation], dtype=np.int32)

        if len(bbox) > 0:
            whole_mask = np.stack(
                [self._segm_to_mask(anno['segmentation'], (H, W))
                 for anno in annotation])
        else:
            whole_mask = np.zeros((0, H, W), dtype=np.bool)

        crowded = np.array([ann['iscrowd']
                            for ann in annotation], dtype=np.bool)

        area = np.array([ann['area']
                         for ann in annotation], dtype=np.float32)

        # Sanitize boxes using image shape
        bbox[:, :2] = np.maximum(bbox[:, :2], 0)
        bbox[:, 2] = np.minimum(bbox[:, 2], H)
        bbox[:, 3] = np.minimum(bbox[:, 3], W)

        # Remove invalid boxes
        bbox_area = np.prod(bbox[:, 2:] - bbox[:, :2], axis=1)
        keep_mask = np.logical_and(bbox[:, 0] <= bbox[:, 2],
                                   bbox[:, 1] <= bbox[:, 3])
        keep_mask = np.logical_and(keep_mask, bbox_area > 0)
        bbox = bbox[keep_mask]
        label = label[keep_mask]
        crowded = crowded[keep_mask]
        whole_mask = whole_mask[keep_mask]
        area = area[keep_mask]
        return bbox, whole_mask, label, crowded, area

    def _segm_to_mask(self, segm, size):
        # Copied from pycocotools.coco.COCO.annToMask
        H, W = size
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = coco_mask.frPyObjects(segm, H, W)
            rle = coco_mask.merge(rles)
        elif isinstance(segm['counts'], list):
            rle = coco_mask.frPyObjects(segm, H, W)
        else:
            rle = segm
        mask = coco_mask.decode(rle)
        return mask.astype(np.bool)

    def __len__(self):
        return len(self.ids)

    def get_example(self, i):
        img_id = self.ids[i]
        img_fn = os.path.join(
            self.img_root, self.img_props[img_id]['file_name'])
        img = utils.read_image(img_fn, dtype=np.float32, color=True)
        img = img[::-1, :, :]  # RGB -> BGR
        _, H, W = img.shape

        bbox, whole_mask, label, crowded, area = self._get_annotations(i)

        if not self.use_crowded:
            bbox = bbox[np.logical_not(crowded)]
            label = label[np.logical_not(crowded)]
            whole_mask = whole_mask[np.logical_not(crowded)]
            area = area[np.logical_not(crowded)]
            crowded = crowded[np.logical_not(crowded)]

        example = [img, bbox, whole_mask, label]
        if self.return_crowded:
            example += [crowded]
        if self.return_area:
            example += [area]
        return tuple(example)

    def visualize(self, i):
        img, bbox, whole_mask, label = self.get_example(i)
        img = img.transpose(1, 2, 0)
        img = img[:, :, ::-1]
        scores = np.ones(len(label))
        mask = whole_mask2mask(whole_mask, bbox)
        visualize_mask(img, mask, bbox, label, scores, coco_label_names)
        plt.show()


def _index_list_by_mask(mask, mask_mask):
    indices = np.where(mask_mask)[0]
    mask = [mask[idx] for idx in indices]
    return mask
