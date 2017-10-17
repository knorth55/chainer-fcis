import chainer
import chainer.functions as F
import chainer.links as L
from chainercv.links.model.faster_rcnn.region_proposal_network import \
    RegionProposalNetwork
from chainercv.links.model.faster_rcnn.utils.loc2bbox import loc2bbox
import cupy
import fcis.functions
from fcis.models.resnet101 import ResNet101C1
from fcis.models.resnet101 import ResNet101C2
from fcis.models.resnet101 import ResNet101C3
from fcis.models.resnet101 import ResNet101C4
from fcis.models.resnet101 import ResNet101C5
import fcn
import numpy as np
import os.path as osp


filepath = osp.abspath(osp.dirname(__file__))


class FCISResNet101(chainer.Chain):

    feat_stride = 16
    mean_bgr = np.array([103.06, 115.90, 123.15])
    pretrained_model = osp.expanduser(
        '~/data/models/chainer/fcis_coco.npz')

    def __init__(
            self, n_class,
            ratios=[0.5, 1, 2],
            anchor_scales=[4, 8, 16, 32],
            n_train_pre_nms=6000, n_train_post_nms=300,
            n_test_pre_nms=6000, n_test_post_nms=300,
            nms_thresh=0.7, rpn_min_size=16,
            group_size=7, roi_size=21,
            loc_normalize_mean=(0.0, 0.0, 0.0, 0.0),
            loc_normalize_std=(0.2, 0.2, 0.5, 0.5),
    ):
        super(FCISResNet101, self).__init__()
        proposal_creator_params = {
            'nms_thresh': nms_thresh,
            'n_train_pre_nms': n_train_pre_nms,
            'n_train_post_nms': n_train_post_nms,
            'n_test_pre_nms': n_test_pre_nms,
            'n_test_post_nms': n_test_post_nms,
            'force_cpu_nms': False,
            'min_size': 2
        }

        self.n_class = n_class
        self.spatial_scale = 1. / self.feat_stride
        self.group_size = group_size
        self.roi_size = roi_size
        self.loc_normalize_mean = loc_normalize_mean
        self.loc_normalize_std = loc_normalize_std

        with self.init_scope():
            # ResNet
            self.res1 = ResNet101C1()
            self.res2 = ResNet101C2()
            self.res3 = ResNet101C3()
            self.res4 = ResNet101C4()
            self.res5 = ResNet101C5()

            # RPN
            self.rpn = RegionProposalNetwork(
                1024, 512,
                ratios=ratios,
                anchor_scales=anchor_scales,
                feat_stride=self.feat_stride,
                proposal_creator_params=proposal_creator_params
            )

            # PSROI Pooling
            self.psroi_conv1 = L.Convolution2D(2048, 1024, 1, 1, 0)
            self.psroi_conv2 = L.Convolution2D(
                1024, group_size*group_size*self.n_class*2, 1, 1, 0)
            self.psroi_conv3 = L.Convolution2D(
                1024, group_size*group_size*2*4, 1, 1, 0)

    def __call__(self, x, scale=1.0):
        img_size = x.shape[2:]

        # Feature Extractor
        h = self.res1(x)
        h = self.res2(h)
        h = self.res3(h)
        h = self.res4(h)
        self.res4_h = h

        # RPN
        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.rpn(
            h, img_size, scale)
        roi_indices = roi_indices.astype(np.float32)
        indices_and_rois = self.xp.concatenate(
            (roi_indices[:, None], rois), axis=1)
        self.rois1 = rois

        h = self.res5(h)
        self.res5_h = h

        h = F.relu(self.psroi_conv1(h))
        self.psroi_conv1_h = h
        h_seg = self.psroi_conv2(h)
        self.psroi_conv2_h = h_seg
        h_locs = self.psroi_conv3(h)
        self.psroi_conv3_h = h_locs

        roi_locs, roi_cls_probs, roi_seg_probs = self._pool_and_predict(
            indices_and_rois, h_seg, h_locs)

        # Iter2
        roi_locs = roi_locs.data
        roi_locs = roi_locs.reshape((-1, 2, 4))
        roi_locs = roi_locs[:, 1, :]
        mean = self.xp.array(self.loc_normalize_mean)
        std = self.xp.array(self.loc_normalize_std)
        roi_locs = roi_locs * std + mean
        rois2 = loc2bbox(rois, roi_locs)
        H, W = img_size
        rois2[:, 0::2] = self.xp.clip(rois2[:, 0::2], 0, H)
        rois2[:, 1::2] = self.xp.clip(rois2[:, 1::2], 0, W)

        self.rois2 = rois2

        indices_and_rois2 = self.xp.concatenate(
            (roi_indices[:, None], rois2), axis=1)
        indices_and_rois2 = indices_and_rois2.astype(self.xp.float32)
        _, roi_cls_probs2, roi_seg_probs2 = self._pool_and_predict(
            indices_and_rois2, h_seg, h_locs)

        rois = self.xp.concatenate((rois, rois2))
        roi_indices = self.xp.concatenate((roi_indices, roi_indices))
        roi_cls_probs = self.xp.concatenate(
            (roi_cls_probs.data, roi_cls_probs2.data))
        roi_seg_probs = self.xp.concatenate(
            (roi_seg_probs.data, roi_seg_probs2.data))

        self.rois = rois
        self.roi_indices = roi_indices
        self.roi_cls_probs = roi_cls_probs
        self.roi_seg_probs = roi_seg_probs

    def _pool_and_predict(self, indices_and_rois, h_seg, h_locs):
        # PSROI Pooling
        # shape: (n_rois, n_class*2, H, W)
        pool_seg = _psroi_pooling_2d_yx(
            h_seg, indices_and_rois, self.roi_size, self.roi_size,
            self.spatial_scale, group_size=self.group_size,
            output_dim=self.n_class*2)
        # shape: (n_rois, n_class, 2, H, W)
        pool_seg = pool_seg.reshape(
            (-1, self.n_class, 2, self.roi_size, self.roi_size))
        # shape: (n_rois, 2*4, H, W)
        pool_locs = _psroi_pooling_2d_yx(
            h_locs, indices_and_rois, self.roi_size, self.roi_size,
            self.spatial_scale, group_size=self.group_size,
            output_dim=2*4)

        # Classfication
        # Group Max
        # shape: (n_rois, n_class, H, W)
        h_cls = F.max(pool_seg, axis=2)

        n_rois, n_class, _, _ = h_cls.shape
        # Global pooling (vote)
        # shape: (n_rois, n_class)
        roi_cls_scores = F.average(h_cls, axis=(2, 3))
        roi_cls_probs = F.softmax(roi_cls_scores)

        # Bbox Regression
        # shape: (n_rois, 2*4)
        roi_locs = F.average(pool_locs, axis=(2, 3))

        # Mask Regression
        # shape: (n_rois, n_class, 2, H, W)
        roi_seg_probs = F.softmax(pool_seg, axis=2)

        # Group Pick by Score
        max_cls_idx = roi_cls_probs.data.argmax(axis=1)
        roi_seg_probs = roi_seg_probs[np.arange(len(max_cls_idx)), max_cls_idx]

        return roi_locs, roi_cls_probs, roi_seg_probs

    def prepare(self, img, target_height, max_width, gpu=0):
        resized_img = fcis.utils.resize_image(img, target_height, max_width)
        x_data = resized_img.copy()
        x_data = x_data.astype(np.float32)
        x_data -= self.mean_bgr
        x_data = x_data.transpose((2, 0, 1))  # H, W, C -> C, H, W
        x = chainer.Variable(np.array([x_data], dtype=np.float32))
        x.to_gpu(gpu)
        scale = resized_img.shape[0] / float(img.shape[0])
        return x, scale

    def predict(
            self, orig_imgs, gpu=0,
            target_height=600, max_width=1000,
            score_thresh=0.7, nms_thresh=0.3,
            mask_merge_thresh=0.5, binary_thresh=0.4):

        masks = []
        bboxes = []
        labels = []
        cls_probs = []

        for orig_img in orig_imgs:
            orig_H, orig_W, _ = orig_img.shape
            x, scale = self.prepare(
                orig_img, target_height, max_width, gpu=gpu)

            # inference
            self.__call__(x)

            # assume that batch_size = 1
            rois = self.rois
            rois = rois / scale
            roi_cls_probs = self.roi_cls_probs
            roi_seg_probs = self.roi_seg_probs

            # shape: (n_rois, H, W)
            roi_mask_probs = roi_seg_probs[:, 1, :, :]

            # shape: (n_rois, 4)
            rois[:, 0::2] = cupy.clip(rois[:, 0::2], 0, orig_H)
            rois[:, 1::2] = cupy.clip(rois[:, 1::2], 0, orig_W)

            # voting
            # cpu voting is only implemented
            rois = chainer.cuda.to_cpu(rois)
            roi_cls_probs = chainer.cuda.to_cpu(roi_cls_probs)
            roi_mask_probs = chainer.cuda.to_cpu(roi_mask_probs)

            mask_prob, bbox, label, cls_prob = fcis.mask.mask_voting(
                rois, roi_mask_probs, roi_cls_probs, self.n_class,
                orig_H, orig_W, score_thresh, nms_thresh, mask_merge_thresh,
                binary_thresh)
            mask = fcis.utils.mask_probs2mask(mask_prob, bbox, binary_thresh)
            masks.append(mask)
            bboxes.append(bbox)
            labels.append(label)
            cls_probs.append(cls_prob)

        return masks, bboxes, labels, cls_probs

    @classmethod
    def download(cls):
        return fcn.data.cached_download(
            url='https://drive.google.com/uc?id=0B5DV6gwLHtyJZTR0NFllNGlwS3M',  # NOQA
            path=cls.pretrained_model,
            md5='689f9f01e7ee37f591b218e49c6686fb',
        )


def _psroi_pooling_2d_yx(
        x, indices_and_rois, outh, outw,
        spatial_scale, group_size, output_dim):
    xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
    pool = fcis.functions.psroi_pooling_2d(
        x, xy_indices_and_rois, outh, outw, spatial_scale,
        group_size, output_dim)
    return pool
