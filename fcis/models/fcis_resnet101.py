import chainer
import chainer.functions as F
import chainer.links as L
from chainercv.links.model.faster_rcnn.region_proposal_network import \
    RegionProposalNetwork
from chainercv.links.model.faster_rcnn.utils.loc2bbox import loc2bbox
import fcis.functions
from fcis.models.resnet101 import ResNet101C1
from fcis.models.resnet101 import ResNet101C2
from fcis.models.resnet101 import ResNet101C3
from fcis.models.resnet101 import ResNet101C4
from fcis.models.resnet101 import ResNet101C5
import numpy as np


class FCISResNet101(chainer.Chain):

    feat_stride = 16

    def __init__(
            self, n_class, ratios=[0.5, 1, 2],
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
            indices_and_rois, h_seg, h_locs, True)

        # Iter2
        roi_locs = roi_locs.data
        roi_locs = roi_locs[:, 4:]
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
        # shape: (n_rois, n_class, H*W)
        h_cls = h_cls.reshape((n_rois, n_class, -1))
        # Global pooling (vote)
        # shape: (n_rois, n_class)
        roi_cls_scores = F.average(h_cls, axis=2)
        roi_cls_probs = F.softmax(roi_cls_scores)

        # Bbox Regression
        # shape: (n_rois, 2*4, H*W)
        pool_locs = pool_locs.reshape((n_rois, 2*4, -1))
        # shape: (n_rois, 2*4)
        roi_locs = F.average(pool_locs, axis=2)

        # Mask Regression
        # shape: (n_rois, n_class, 2, H, W)
        roi_seg_probs = F.softmax(pool_seg, axis=2)

        # Group Pick by Score
        max_cls_idx = roi_cls_probs.data.argmax(axis=1)
        roi_seg_probs = roi_seg_probs[np.arange(len(max_cls_idx)), max_cls_idx]

        return roi_locs, roi_cls_probs, roi_seg_probs


def _psroi_pooling_2d_yx(
        x, indices_and_rois, outh, outw,
        spatial_scale, group_size, output_dim):
    xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
    pool = fcis.functions.psroi_pooling_2d(
        x, xy_indices_and_rois, outh, outw, spatial_scale,
        group_size, output_dim)
    return pool
