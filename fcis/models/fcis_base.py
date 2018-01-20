from __future__ import division

import chainer
import chainer.functions as F
import cupy
import numpy as np

import fcis


class FCIS(chainer.Chain):

    def __init__(self, extractor, rpn, head):
        super(FCIS, self).__init__()
        with self.init_scope():
            self.extractor = extractor
            self.rpn = rpn
            self.head = head

    def __call__(self, x, scale=1.0, iter2=True):
        img_size = x.shape[2:]

        # Feature Extractor
        rpn_features, roi_features = self.extractor(x)
        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.rpn(
            rpn_features, img_size, scale)
        rois, roi_indices, roi_seg_scores, roi_cls_locs, roi_cls_scores = \
            self.head(roi_features, rois, roi_indices, img_size, iter2=iter2)
        return rois, roi_indices, roi_seg_scores, roi_cls_locs, roi_cls_scores

    def prepare(self, orig_img, target_height=600, max_width=1000):
        img = orig_img.copy()
        img = img.transpose((1, 2, 0))  # C, H, W -> H, W, C
        img = fcis.utils.resize_image(img, target_height, max_width)
        img = img.astype(np.float32)
        img -= self.mean_bgr
        img = img.transpose((2, 0, 1))  # H, W, C -> C, H, W
        return img

    def predict(
            self, orig_imgs,
            target_height=600, max_width=1000,
            score_thresh=0.7, nms_thresh=0.3,
            mask_merge_thresh=0.5, binary_thresh=0.4,
            min_drop_size=2,
            iter2=True, mask_voting=True):

        bboxes = []
        whole_masks = []
        labels = []
        cls_probs = []

        for orig_img in orig_imgs:
            _, orig_H, orig_W = orig_img.shape
            img = self.prepare(
                orig_img, target_height, max_width)
            img = img.astype(np.float32)
            scale = img.shape[1] / orig_H
            with chainer.using_config('train', False), \
                    chainer.function.no_backprop_mode():
                # inference
                x = chainer.Variable(self.xp.array(img[None]))
                bbox, _, seg_score, _, cls_score = self.__call__(
                    x, scale, iter2=iter2)
                seg_prob = F.softmax(seg_score).array
                cls_prob = F.softmax(cls_score).array

            # assume that batch_size = 1
            bbox = bbox / scale

            # shape: (n_rois, H, W)
            mask_prob = seg_prob[:, 1, :, :]

            # shape: (n_rois, 4)
            bbox[:, 0::2] = cupy.clip(bbox[:, 0::2], 0, orig_H)
            bbox[:, 1::2] = cupy.clip(bbox[:, 1::2], 0, orig_W)

            # voting
            # cpu voting is only implemented
            bbox = chainer.cuda.to_cpu(bbox)
            cls_prob = chainer.cuda.to_cpu(cls_prob)
            mask_prob = chainer.cuda.to_cpu(mask_prob)

            if mask_voting:
                bbox, mask_prob, label, cls_prob = fcis.mask.mask_voting(
                    bbox, mask_prob, cls_prob, self.n_class,
                    orig_H, orig_W, score_thresh, nms_thresh,
                    mask_merge_thresh, binary_thresh)
            else:
                label = cls_prob.argmax(axis=1)
                label_mask = label != 0
                cls_prob = cls_prob[np.arange(len(cls_prob)), label]
                keep_indices = cls_prob > score_thresh
                keep_indices = np.logical_and(
                    label_mask, keep_indices)
                bbox = bbox[keep_indices]
                mask_prob = mask_prob[keep_indices]
                cls_prob = cls_prob[keep_indices]
                label = label[keep_indices]

            height = bbox[:, 2] - bbox[:, 0]
            width = bbox[:, 3] - bbox[:, 1]
            keep_indices = np.where(
                (height > min_drop_size) & (width > min_drop_size))[0]
            bbox = bbox[keep_indices]
            mask_prob = mask_prob[keep_indices]
            cls_prob = cls_prob[keep_indices]
            label = label[keep_indices]

            mask = fcis.utils.mask_probs2mask(mask_prob, bbox, binary_thresh)
            whole_mask = fcis.utils.mask2whole_mask(
                mask, bbox, (orig_H, orig_W))
            whole_masks.append(whole_mask)
            bboxes.append(bbox)
            labels.append(label)
            cls_probs.append(cls_prob)

        return bboxes, whole_masks, labels, cls_probs
