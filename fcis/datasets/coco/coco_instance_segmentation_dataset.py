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

try:
    from pycocotools import mask as coco_mask
    _available = True
except ImportError:
    _available = False


class COCOInstanceSegmentationDataset(chainer.dataset.DatasetMixin):

    def __init__(self, data_dir=None, split='train2014',
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
        if split in ['val2014', 'minival2014', 'valminusminival2014']:
            img_splits = ['val2014']
            splits = [split]
        elif split == 'trainval2014':
            img_splits = ['train2014', 'val2014']
            splits = ['train2014', 'valminusminival2014']
        elif 'test2014' in split:
            img_splits = ['test2014']
            splits = [split]
        elif 'test2015' in split:
            img_splits = ['test2015']
            splits = [split]
        else:
            img_splits = ['train2014']
            splits = [split]

        if data_dir is None:
            data_dir = osp.expanduser('~/data/datasets/coco')
        elif data_dir == 'auto':
            data_dir = None

        for split, img_split in zip(splits, img_splits):
            data_dir = get_coco(split, img_split, data_dir)

        if not osp.exists(data_dir):
            raise ValueError('Please download coco dataset first')
        self.data_dir = data_dir

        self.img_props = dict()
        self.ids = list()
        self.cat_ids = list()
        self.anns = dict()
        self.imgToAnns = defaultdict(list)
        self.img_dirs = dict()
        for sp, img_sp in zip(splits, img_splits):
            data = self._load_data(data_dir, sp, img_sp)
            self.img_props.update(data[0])
            self.ids.extend(data[1])
            self.cat_ids.extend(data[2])
            self.anns.update(data[3])
            self.imgToAnns.update(data[4])
            self.img_dirs.update(data[5])

    def _load_data(self, data_dir, split, img_split):
        if 'test' in split:
            anno_prefix = 'image_info'
        else:
            anno_prefix = 'instances'
        anno_fn = os.path.join(
            data_dir, 'annotations', '{0}_{1}.json'.format(anno_prefix, split))
        anno = json.load(open(anno_fn, 'r'))

        img_props = dict()
        for img in anno['images']:
            img_props[img['id']] = img
        ids = list(img_props.keys())

        cats = anno['categories']
        cat_ids = [0] + [cat['id'] for cat in cats]

        anns = dict()
        imgToAnns = defaultdict(list)
        if 'annotations' in anno:
            for ann in anno['annotations']:
                imgToAnns[ann['image_id']].append(ann)
                anns[ann['id']] = ann

        img_dirs = {x: img_split for x in ids}
        return img_props, ids, cat_ids, anns, imgToAnns, img_dirs

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
        img_root = os.path.join(
            self.data_dir, 'images', self.img_dirs[img_id])
        img_fn = os.path.join(
            img_root, self.img_props[img_id]['file_name'])
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
        import matplotlib.pyplot as plt
        img, bbox, whole_mask, label = self.get_example(i)
        img = img.transpose(1, 2, 0)
        img = img[:, :, ::-1]
        scores = np.ones(len(label))
        visualize_mask(img, whole_mask, bbox, label, scores, coco_label_names)
        plt.show()


def _index_list_by_mask(mask, mask_mask):
    indices = np.where(mask_mask)[0]
    mask = [mask[idx] for idx in indices]
    return mask
