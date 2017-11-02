import cv2
from fcis.datasets.voc import VOCInstanceSegmentationDataset
import numpy as np
import os.path as osp
import scipy


this_dir = osp.dirname(osp.realpath(__file__))


class SBDInstanceSegmentationDataset(VOCInstanceSegmentationDataset):

    data_dir = osp.expanduser('~/data/datasets/VOC/benchmark_RELEASE/dataset')
    imgsets_dir = osp.join(this_dir, 'data/')

    def _load_data(self, data_id):
        imgpath = osp.join(
            self.data_dir, 'img/{}.jpg'.format(data_id))
        seg_imgpath = osp.join(
            self.data_dir, 'cls/{}.mat'.format(data_id))
        ins_imgpath = osp.join(
            self.data_dir, 'inst/{}.mat'.format(data_id))
        img = cv2.imread(imgpath)
        img = img.transpose((2, 0, 1))
        mat = scipy.io.loadmat(seg_imgpath)
        seg_img = mat['GTcls'][0]['Segmentation'][0].astype(np.int32)
        seg_img = np.array(seg_img, dtype=np.int32)
        seg_img[seg_img == 255] = -1
        mat = scipy.io.loadmat(ins_imgpath)
        ins_img = mat['GTinst'][0]['Segmentation'][0].astype(np.int32)
        ins_img[ins_img == 255] = -1
        ins_img[np.isin(seg_img, [-1, 0])] = -1
        return img, seg_img, ins_img
