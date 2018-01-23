from __future__ import division

import chainer
import chainer.functions as F
import chainer.links as L
from chainercv.links.model.faster_rcnn.region_proposal_network import \
    RegionProposalNetwork
from chainercv.links.model.faster_rcnn.utils.loc2bbox import loc2bbox
import fcn
import numpy as np
import os.path as osp

import fcis.functions
from fcis.models import FCIS
from fcis.models.resnet101 import ResNet101C1
from fcis.models.resnet101 import ResNet101C2
from fcis.models.resnet101 import ResNet101C3
from fcis.models.resnet101 import ResNet101C4
from fcis.models.resnet101 import ResNet101C5


class ResNet101Extractor(chainer.Chain):

    def __init__(self):
        super(ResNet101Extractor, self).__init__()
        with self.init_scope():
            # ResNet
            self.res1 = ResNet101C1()
            self.res2 = ResNet101C2()
            self.res3 = ResNet101C3()
            self.res4 = ResNet101C4()
            self.res5 = ResNet101C5()

    def __call__(self, x):
        with chainer.using_config('train', False):
            with chainer.function.no_backprop_mode():
                h = self.res1(x)
                h = self.res2(h)
            h = self.res3(h)
            res4 = self.res4(h)
            res5 = self.res5(res4)
        return res4, res5

    def init_weight(self, pretrained_model=None):
        if pretrained_model is None:
            pretrained_model = chainer.links.ResNet101Layers(
                pretrained_model='auto')

        n_layer_dict = {
            'res2': 3,
            'res3': 4,
            'res4': 23,
            'res5': 3
        }

        def copy_conv(conv, orig_conv):
            assert conv is not orig_conv
            assert conv.W.array.shape == orig_conv.W.array.shape
            conv.W.array[:] = orig_conv.W.array

        def copy_bn(bn, orig_bn):
            assert bn is not orig_bn
            assert bn.gamma.array.shape == orig_bn.gamma.array.shape
            assert bn.beta.array.shape == orig_bn.beta.array.shape
            assert bn.avg_var.shape == orig_bn.avg_var.shape
            assert bn.avg_mean.shape == orig_bn.avg_mean.shape
            bn.gamma.array[:] = orig_bn.gamma.array
            bn.beta.array[:] = orig_bn.beta.array
            bn.avg_var[:] = orig_bn.avg_var
            bn.avg_mean[:] = orig_bn.avg_mean

        def copy_bottleneck(bottle, orig_bottle, n_conv):
            for i in range(0, n_conv):
                conv_name = 'conv{}'.format(i + 1)
                conv = getattr(bottle, conv_name)
                orig_conv = getattr(orig_bottle, conv_name)
                copy_conv(conv, orig_conv)

                bn_name = 'bn{}'.format(i + 1)
                bn = getattr(bottle, bn_name)
                orig_bn = getattr(orig_bottle, bn_name)
                copy_bn(bn, orig_bn)

        def copy_block(block, orig_block, res_name):
            n_layer = n_layer_dict[res_name]
            bottle = getattr(block, '{}_a'.format(res_name))
            copy_bottleneck(bottle, orig_block.a, 4)
            for i in range(1, n_layer):
                bottle = getattr(block, '{0}_b{1}'.format(res_name, i))
                orig_bottle = getattr(orig_block, 'b{}'.format(i))
                copy_bottleneck(bottle, orig_bottle, 3)

        copy_conv(self.res1.conv1, pretrained_model.conv1)
        copy_bn(self.res1.bn1, pretrained_model.bn1)
        copy_block(self.res2, pretrained_model.res2, 'res2')
        copy_block(self.res3, pretrained_model.res3, 'res3')
        copy_block(self.res4, pretrained_model.res4, 'res4')
        copy_block(self.res5, pretrained_model.res5, 'res5')


class FCISResNet101Head(chainer.Chain):

    def __init__(
            self,
            n_class, in_size, mid_size,
            spatial_scale, group_size, roi_size,
            loc_normalize_mean, loc_normalize_std,
    ):
        super(FCISResNet101Head, self).__init__()

        self.n_class = n_class
        self.spatial_scale = spatial_scale
        self.group_size = group_size
        self.roi_size = roi_size
        self.loc_normalize_mean = loc_normalize_mean
        self.loc_normalize_std = loc_normalize_std

        initialW = chainer.initializers.Normal(0.01)
        with self.init_scope():
            self.psroi_conv1 = L.Convolution2D(
                in_size, mid_size, 1, 1, 0, initialW=initialW)
            self.psroi_conv2 = L.Convolution2D(
                mid_size, group_size * group_size * n_class * 2,
                1, 1, 0, initialW=initialW)
            self.psroi_conv3 = L.Convolution2D(
                mid_size, group_size * group_size * 2 * 4,
                1, 1, 0, initialW=initialW)

    def __call__(
            self,
            x, rois, roi_indices,
            img_size, iter2,
            gt_roi_labels=None
    ):
        roi_indices = roi_indices.astype(np.float32)
        indices_and_rois = self.xp.concatenate(
            (roi_indices[:, None], rois), axis=1)

        h = F.relu(self.psroi_conv1(x))
        h_cls_seg = self.psroi_conv2(h)
        h_locs = self.psroi_conv3(h)

        # PSROI pooling and regression
        roi_seg_scores, roi_cls_locs, roi_cls_scores = self._pool(
            indices_and_rois, h_cls_seg, h_locs,
            gt_roi_labels=gt_roi_labels)
        if iter2:
            # 2nd Iteration
            # get rois2 for more precise prediction
            roi_cls_locs = roi_cls_locs.array
            roi_locs = roi_cls_locs[:, 1, :]
            mean = self.xp.array(self.loc_normalize_mean, np.float32)
            std = self.xp.array(self.loc_normalize_std, np.float32)
            roi_locs = roi_locs * std + mean
            rois2 = loc2bbox(rois, roi_locs)
            H, W = img_size
            rois2[:, 0::2] = self.xp.clip(rois2[:, 0::2], 0, H)
            rois2[:, 1::2] = self.xp.clip(rois2[:, 1::2], 0, W)

            # PSROI pooling and regression
            indices_and_rois2 = self.xp.concatenate(
                (roi_indices[:, None], rois2), axis=1)
            roi_seg_scores2, roi_cls_locs2, roi_cls_scores2 = self._pool(
                indices_and_rois2, h_cls_seg, h_locs,
                gt_roi_labels=gt_roi_labels)

            # concat 1st and 2nd iteration results
            rois = self.xp.concatenate((rois, rois2))
            roi_indices = self.xp.concatenate((roi_indices, roi_indices))
            roi_cls_scores = F.concat(
                (roi_cls_scores, roi_cls_scores2), axis=0)
            roi_cls_locs = F.concat(
                (roi_cls_locs, roi_cls_locs2), axis=0)
            roi_seg_scores = F.concat(
                (roi_seg_scores, roi_seg_scores2), axis=0)
        return rois, roi_indices, roi_seg_scores, roi_cls_locs, roi_cls_scores

    def _pool(
            self, indices_and_rois, h_cls_seg, h_locs, gt_roi_labels=None):
        # PSROI Pooling
        # shape: (n_rois, n_class*2, roi_size, roi_size)
        pool_cls_seg = _psroi_pooling_2d_yx(
            h_cls_seg, indices_and_rois, self.roi_size, self.roi_size,
            self.spatial_scale, group_size=self.group_size,
            output_dim=self.n_class * 2)
        # shape: (n_rois, n_class, 2, roi_size, roi_size)
        pool_cls_seg = pool_cls_seg.reshape(
            (-1, self.n_class, 2, self.roi_size, self.roi_size))
        # shape: (n_rois, 2*4, roi_size, roi_size)
        pool_locs = _psroi_pooling_2d_yx(
            h_locs, indices_and_rois, self.roi_size, self.roi_size,
            self.spatial_scale, group_size=self.group_size,
            output_dim=2 * 4)

        # Classfication
        # Group Max
        # shape: (n_rois, n_class, roi_size, roi_size)
        h_cls = pool_cls_seg.transpose((0, 1, 3, 4, 2))
        h_cls = F.max(h_cls, axis=4)

        # Global pooling (vote)
        # shape: (n_rois, n_class)
        roi_cls_scores = _global_average_pooling_2d(h_cls)

        # Bbox Regression
        # shape: (n_rois, 2*4)
        roi_cls_locs = _global_average_pooling_2d(pool_locs)
        n_rois = roi_cls_locs.shape[0]
        roi_cls_locs = roi_cls_locs.reshape((n_rois, 2, 4))

        # Mask Regression
        # shape: (n_rois, n_class, 2, roi_size, roi_size)
        # Group Pick by Score
        if gt_roi_labels is None:
            max_cls_idx = roi_cls_scores.array.argmax(axis=1)
        else:
            max_cls_idx = gt_roi_labels
        # shape: (n_rois, 2, roi_size, roi_size)
        roi_seg_scores = pool_cls_seg[np.arange(len(max_cls_idx)), max_cls_idx]

        return roi_seg_scores, roi_cls_locs, roi_cls_scores


class FCISResNet101(FCIS):

    feat_stride = 16
    mean_bgr = np.array([103.06, 115.90, 123.15], dtype=np.float32)
    model_dir = osp.expanduser('~/data/models/chainer/')

    def __init__(
            self, n_class=81,
            ratios=[0.5, 1, 2],
            anchor_scales=[4, 8, 16, 32],
            n_train_pre_nms=6000, n_train_post_nms=300,
            n_test_pre_nms=6000, n_test_post_nms=300,
            nms_thresh=0.7, rpn_min_size=2,
            group_size=7, roi_size=21,
            loc_normalize_mean=(0.0, 0.0, 0.0, 0.0),
            loc_normalize_std=(0.2, 0.2, 0.5, 0.5),
    ):
        proposal_creator_params = {
            'nms_thresh': nms_thresh,
            'n_train_pre_nms': n_train_pre_nms,
            'n_train_post_nms': n_train_post_nms,
            'n_test_pre_nms': n_test_pre_nms,
            'n_test_post_nms': n_test_post_nms,
            'force_cpu_nms': False,
            'min_size': rpn_min_size,
        }

        self.n_class = n_class
        self.loc_normalize_mean = loc_normalize_mean
        self.loc_normalize_std = loc_normalize_std
        spatial_scale = 1. / self.feat_stride

        extractor = ResNet101Extractor()
        rpn = RegionProposalNetwork(
            1024, 512,
            ratios=ratios,
            anchor_scales=anchor_scales,
            feat_stride=self.feat_stride,
            initialW=chainer.initializers.Normal(0.01),
            proposal_creator_params=proposal_creator_params
        )
        head = FCISResNet101Head(
            n_class, 2048, 1024,
            spatial_scale, group_size, roi_size,
            loc_normalize_mean, loc_normalize_std)

        super(FCISResNet101, self).__init__(
            extractor, rpn, head)

    @classmethod
    def download(cls, dataset='coco'):
        if dataset == 'voc_converted':
            url = 'https://drive.google.com/uc?id=1qFEV3txP_TSd9N0ZVmR9gaS5NoTi9MIr'  # NOQA
            path = osp.join(cls.model_dir, 'fcis_voc_converted.npz')
            md5 = '95a4029fe1e0ae6100cca8a3971c687c'
        elif dataset == 'voc':
            url = 'https://drive.google.com/uc?id=1PscvchtzYsT_xsNX8EsmY1j0Kju6j0r0'  # NOQA
            path = osp.join(cls.model_dir, 'fcis_voc_trained.npz')
            md5 = 'aa3206d755abde94bfb2af99cfd4b9bf'
        else:
            url = 'https://drive.google.com/uc?id=1j98jQp2ATBdiQ51p0YsWGkC5mOmduzPW'  # NOQA
            path = osp.join(cls.model_dir, 'fcis_coco.npz')
            md5 = 'f71a7213b32c2a7ef4522561bc917577'
        return fcn.data.cached_download(url=url, path=path, md5=md5)


def _psroi_pooling_2d_yx(
        x, indices_and_rois, outh, outw,
        spatial_scale, group_size, output_dim):
    xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
    pool = fcis.functions.psroi_pooling_2d(
        x, xy_indices_and_rois, outh, outw, spatial_scale,
        group_size, output_dim)
    return pool


def _global_average_pooling_2d(x):
    n_rois, n_channel, H, W = x.array.shape
    h = F.average_pooling_2d(x, (H, W), stride=1)
    h = F.reshape(h, (n_rois, n_channel))
    return h
