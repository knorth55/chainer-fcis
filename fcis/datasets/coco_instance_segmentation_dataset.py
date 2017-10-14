from collections import defaultdict
import json
import numpy as np
import os

import chainer

from chainercv import utils

from fcis.datasets.coco_utils import get_coco

try:
    from pycocotools import mask as coco_mask
    _availabel = True
except ImportError:
    _availabel = False


class COCOInstanceSegmentationDataset(chainer.dataset.DatasetMixin):

    def __init__(self, data_dir='auto', split='train',
                 use_crowded=False, return_crowded=False):
        if not _availabel:
            raise ValueError(
                'Please install pycocotools \n'
                'pip install -e \'git+https://github.com/pdollar/coco.git'
                '#egg=pycocotools&subdirectory=PythonAPI\'')

        self.use_crowded = use_crowded
        self.return_crowded = return_crowded
        if split in ['val', 'minival', 'valminusminival']:
            img_split = 'val'
        else:
            img_split = 'train'
        if data_dir == 'auto':
            data_dir = get_coco(split, img_split)

        self.img_root = os.path.join(
            data_dir, 'images', '{}2014'.format(img_split))
        anno_fn = os.path.join(
            data_dir, 'annotations', 'instances_{}2014.json'.format(split))

        self.data_dir = data_dir
        anno = json.load(open(anno_fn, 'r'))

        self.img_props = dict()
        for img in anno['images']:
            self.img_props[img['id']] = img
        self.ids = list(self.img_props.keys())

        cats = anno['categories']
        self.cat_ids = [cat['id'] for cat in cats]

        self.anns = dict()
        self.imgToAnns = defaultdict(list)
        for ann in anno['annotations']:
            self.imgToAnns[ann['image_id']].append(ann)
            self.anns[ann['id']] = ann

    @property
    def labels(self):
        labels = list()
        for i in range(len(self)):
            _, label, _, _ = self._get_annotations(i)
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

        mask = list()
        for anno, bb in zip(annotation, bbox):
            m = self._segm_to_mask(anno['segmentation'], (H, W))
            bb = bb.astype(np.int32)
            m = m[bb[0]:bb[2], bb[1]:bb[3]]
            mask.append(m)

        crowded = np.array([ann['iscrowd']
                            for ann in annotation], dtype=np.bool)

        # Sanitize boxes using image shape
        bbox[:, :2] = np.maximum(bbox[:, :2], 0)
        bbox[:, 2] = np.minimum(bbox[:, 2], H)
        bbox[:, 3] = np.minimum(bbox[:, 3], W)

        # Remove invalid boxes
        area = np.prod(bbox[:, 2:] - bbox[:, :2], axis=1)
        keep_mask = np.logical_and(bbox[:, 0] <= bbox[:, 2],
                                   bbox[:, 1] <= bbox[:, 3])
        keep_mask = np.logical_and(keep_mask, area > 0)
        bbox = bbox[keep_mask]
        label = label[keep_mask]
        crowded = crowded[keep_mask]
        mask = _index_list_by_mask(mask, keep_mask)
        return bbox, label, mask, crowded

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
        _, H, W = img.shape

        bbox, label, mask, crowded = self._get_annotations(i)

        if not self.use_crowded:
            bbox = bbox[np.logical_not(crowded)]
            label = label[np.logical_not(crowded)]
            mask = _index_list_by_mask(mask, np.logical_not(crowded))
            crowded = crowded[np.logical_not(crowded)]

        if self.return_crowded:
            return img, bbox, label, mask, crowded
        return img, bbox, label, mask


def _index_list_by_mask(l, mask):
    indices = np.where(mask)[0]
    l = [l[idx] for idx in indices]
    return l
