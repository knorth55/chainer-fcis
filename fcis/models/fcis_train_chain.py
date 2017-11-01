# _smooth_l1_loss and _fast_rcnn_loc_loss
# are originally from https://github.com/chainer/chainercv
# and worked by Yusuke Niitani (@yuyu2172)

import chainer
import chainer.functions as F
from chainercv.links.model.faster_rcnn.utils.anchor_target_creator import\
    AnchorTargetCreator
from fcis.proposal_target_creator import ProposalTargetCreator
import numpy as np


class FCISTrainChain(chainer.Chain):

    def __init__(self, fcis, rpn_sigma=3.0, roi_sigma=1.0):
        super(FCISTrainChain, self).__init__()
        with self.init_scope():
            self.fcis = fcis
        self.rpn_sigma = rpn_sigma
        self.roi_sigma = roi_sigma

        self.loc_normalize_mean = fcis.loc_normalize_mean
        self.loc_normalize_std = fcis.loc_normalize_std

        self.anchor_target_creator = AnchorTargetCreator()
        self.proposal_target_creator = ProposalTargetCreator(
            loc_normalize_mean=self.loc_normalize_mean,
            loc_normalize_std=self.loc_normalize_std)

    def __call__(self, x, bboxes, whole_mask, labels, scale):
        if isinstance(bboxes, chainer.Variable):
            bboxes = bboxes.data
        if isinstance(whole_mask, chainer.Variable):
            whole_mask = whole_mask.data
        if isinstance(labels, chainer.Variable):
            labels = labels.data
        if isinstance(scale, chainer.Variable):
            scale = scale.data
        scale = chainer.cuda.to_cpu(scale)[0]
        n = bboxes.shape[0]
        # batch size = 1
        if n != 1:
            raise ValueError('Currently only batch size 1 is supported.')

        _, _, H, W = x.shape
        img_size = (H, W)
        assert img_size == whole_mask.shape[2:]

        with chainer.using_config('train', False):
            h = self.fcis.res1(x)
            h = self.fcis.res2(h)
        h = self.fcis.res3(h)
        h = self.fcis.res4(h)

        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.fcis.rpn(
            h, (H, W), scale)

        h = self.fcis.res5(h)

        h = F.relu(self.fcis.psroi_conv1(h))
        h_seg = self.fcis.psroi_conv2(h)
        h_locs = self.fcis.psroi_conv3(h)

        # batch size = 1
        bboxes = bboxes[0]
        whole_mask = whole_mask[0]
        labels = labels[0]
        rpn_scores = rpn_scores[0]
        rpn_locs = rpn_locs[0]

        # Sample RoIs and forward
        sample_rois, gt_roi_locs, gt_roi_masks, gt_roi_labels = \
            self.proposal_target_creator(rois, bboxes, whole_mask, labels)

        sample_roi_indices = self.xp.zeros(
            (len(sample_rois),), dtype=np.float32)
        sample_indices_and_rois = self.xp.concatenate(
            (sample_roi_indices[:, None], sample_rois), axis=1)

        roi_seg_scores, roi_locs, roi_cls_scores = \
            self.fcis._pool_and_predict(
                sample_indices_and_rois, h_seg, h_locs,
                gt_roi_labels=gt_roi_labels)
        roi_locs = roi_locs.reshape((len(roi_locs), -1))

        # RPN losses
        gt_rpn_locs, gt_rpn_labels = self.anchor_target_creator(
            bboxes, anchor, img_size)
        rpn_loc_loss = _fast_rcnn_loc_loss(
            rpn_locs, gt_rpn_locs, gt_rpn_labels, self.rpn_sigma)
        rpn_cls_loss = F.softmax_cross_entropy(
            rpn_scores, gt_rpn_labels)
        rpn_loss = rpn_loc_loss + rpn_cls_loss

        # FCIS losses
        fcis_loc_loss = _fast_rcnn_loc_loss(
            roi_locs, gt_roi_locs, gt_roi_labels, self.roi_sigma)
        fcis_cls_loss = F.softmax_cross_entropy(
            roi_cls_scores, gt_roi_labels)
        fcis_mask_loss = F.softmax_cross_entropy(
            roi_seg_scores, gt_roi_masks)
        fcis_loss = fcis_loc_loss + fcis_cls_loss + 10.0 * fcis_mask_loss

        # Total loss
        loss = rpn_loss + fcis_loss
        chainer.reporter.report({
            'loss': loss,
            'rpn_loc_loss': rpn_loc_loss,
            'rpn_cls_loss': rpn_cls_loss,
            'fcis_loc_loss': fcis_loc_loss,
            'fcis_cls_loss': fcis_cls_loss,
            'fcis_mask_loss': fcis_mask_loss
        }, self)
        return loss


def _smooth_l1_loss(x, t, in_weight, sigma):
    sigma2 = sigma ** 2
    diff = in_weight * (x - t)
    abs_diff = F.absolute(diff)
    flag = (abs_diff.data < (1. / sigma2)).astype(np.float32)

    y = (flag * (sigma2 / 2.) * F.square(diff) +
         (1 - flag) * (abs_diff - 0.5 / sigma2))

    return F.sum(y)


def _fast_rcnn_loc_loss(pred_loc, gt_loc, gt_label, sigma):
    xp = chainer.cuda.get_array_module(pred_loc)

    in_weight = xp.zeros_like(gt_loc)
    # Localization loss is calculated only for positive rois.
    in_weight[gt_label > 0] = 1
    loc_loss = _smooth_l1_loss(pred_loc, gt_loc, in_weight, sigma)
    # Normalize by total number of negtive and positive rois.
    loc_loss /= xp.sum(gt_label >= 0)
    return loc_loss
