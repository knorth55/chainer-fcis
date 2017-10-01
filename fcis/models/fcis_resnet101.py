import chainer
import chainer.functions as F
import chainer.links as L
from chainercv.links.model.faster_rcnn.region_proposal_network import \
    RegionProposalNetwork
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
    ):
        super(FCISResNet101, self).__init__()
        proposal_creator_params = {
            'nms_thresh': nms_thresh,
            'n_train_pre_nms': n_train_pre_nms,
            'n_train_post_nms': n_train_post_nms,
            'n_test_pre_nms': n_test_pre_nms,
            'n_test_post_nms': n_test_post_nms,
            'force_cpu_nms': False,
            'min_size': 16
        }

        self.n_class = n_class
        self.spatial_scale = 1. / self.feat_stride
        self.group_size = group_size
        self.roi_size = roi_size

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

        # RPN
        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.rpn(
            h, img_size, scale)
        roi_indices = roi_indices.astype(np.float32)
        indices_and_rois = self.xp.concatenate(
            (roi_indices[:, None], rois), axis=1)

        h = self.res5(h)
        h = F.relu(self.psroi_conv1(h))
        h_cls_seg = self.psroi_conv2(h)
        h_bbox = self.psroi_conv3(h)

        # PSROI Pooling
        # shape: (n_rois, n_class*2, H, W)
        h_cls_seg = _psroi_pooling_2d_yx(
            h_cls_seg, indices_and_rois, self.roi_size, self.roi_size,
            self.spatial_scale, group_size=self.group_size,
            output_dim=self.n_class*2)
        # shape: (n_rois, n_class, 2, H, W)
        h_cls_seg = h_cls_seg.reshape((-1, self.n_class, 2,
                                       self.roi_size, self.roi_size))
        # shape: (n_rois, 2*4, H, W)
        h_bbox = _psroi_pooling_2d_yx(
            h_bbox, indices_and_rois, self.roi_size, self.roi_size,
            self.spatial_scale, group_size=self.group_size,
            output_dim=2*4)

        # Classfication
        # Group Max
        # shape: (n_rois, n_class, H, W)
        h_cls = F.max(h_cls_seg, axis=2)
        h_cls = h_cls.reshape(
            (h_cls.shape[0], h_cls.shape[1], h_cls.shape[2] * h_cls.shape[3]))
        # Global pooling (vote)
        # shape: (n_rois, n_class)
        cls_score = F.average(h_cls, axis=2)
        cls_prob = F.softmax(cls_score)

        # Bbox Regression
        # shape: (n_rois, 2*4)
        h_bbox = h_bbox.reshape(
            (h_bbox.shape[0], h_bbox.shape[1],
             h_bbox.shape[2] * h_bbox.shape[3]))
        bbox_pred = F.average(h_bbox, axis=2)

        # Mask Regression
        # Group Softmax
        # shape: (n_rois, n_class, 2, H, W)
        h_seg = F.softmax(h_cls_seg, axis=2)

        # Group Pick by Score
        max_cls_prob = cls_prob.data.argmax(axis=1)
        max_cls_mask = self.xp.zeros(h_seg.shape[:2])
        max_cls_mask[:, max_cls_prob] = 1
        seg_pred = h_seg[max_cls_mask.astype(bool)]
        # shape: (n_rois, 2, H, W)
        seg_pred = seg_pred.reshape((-1, 2, self.roi_size, self.roi_size))

        return rois, roi_indices, cls_prob, bbox_pred, seg_pred


def _psroi_pooling_2d_yx(
        x, indices_and_rois, outh, outw,
        spatial_scale, group_size, output_dim):
    xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
    pool = fcis.functions.psroi_pooling_2d(
        x, xy_indices_and_rois, outh, outw, spatial_scale,
        group_size, output_dim)
    return pool
