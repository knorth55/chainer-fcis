# this is originally from https://github.com/chainer/chainercv
# and worked by Yusuke Niitani (@yuyu2172)
# modified by Shingo Kitagawa (@knorth55)

from chainer import cuda
from chainercv.links.model.faster_rcnn.utils.bbox2loc import bbox2loc
from chainercv.utils.bbox.bbox_iou import bbox_iou
import cv2
import fcis
import numpy as np


class ProposalTargetCreator(object):
    def __init__(
            self, n_sample=128,
            loc_normalize_mean=(0., 0., 0., 0.),
            loc_normalize_std=(0.2, 0.2, 0.5, 0.5),
            fg_ratio=0.25, fg_iou_thresh=0.5,
            bg_iou_thresh_hi=0.5, bg_iou_thresh_lo=0.0,
            mask_size=21, binary_thresh=0.4):

        self.n_sample = n_sample
        self.loc_normalize_mean = loc_normalize_mean
        self.loc_normalize_std = loc_normalize_std
        self.fg_ratio = fg_ratio
        self.fg_iou_thresh = fg_iou_thresh
        self.bg_iou_thresh_hi = bg_iou_thresh_hi
        self.bg_iou_thresh_lo = bg_iou_thresh_lo
        self.mask_size = mask_size
        self.binary_thresh = binary_thresh

    def __call__(self, rois, bboxes, whole_mask, labels):

        rois = cuda.to_cpu(rois)
        bboxes = cuda.to_cpu(bboxes)
        whole_mask = cuda.to_cpu(whole_mask)
        labels = cuda.to_cpu(labels)

        n_bbox, _ = bboxes.shape

        rois = np.concatenate((rois, bboxes), axis=0)
        if self.n_sample is None:
            n_sample = rois.shape[0]
        else:
            n_sample = self.n_sample

        fg_rois_per_image = np.round(n_sample * self.fg_ratio)
        iou = bbox_iou(rois, bboxes)
        gt_assignment = iou.argmax(axis=1)
        max_iou = iou.max(axis=1)

        # Select foreground RoIs as those with >= fg_iou_thresh IoU.
        fg_indices = np.where(max_iou >= self.fg_iou_thresh)[0]
        fg_rois_per_this_image = int(min(fg_rois_per_image, fg_indices.size))
        if fg_indices.size > 0:
            fg_indices = np.random.choice(
                fg_indices, size=fg_rois_per_this_image, replace=False)

        # Select background RoIs as those within
        # [bg_iou_thresh_lo, bg_iou_thresh_hi).
        bg_indices = np.where((max_iou < self.bg_iou_thresh_hi) &
                              (max_iou >= self.bg_iou_thresh_lo))[0]
        bg_rois_per_this_image = n_sample - fg_rois_per_this_image
        bg_rois_per_this_image = int(min(bg_rois_per_this_image,
                                         bg_indices.size))
        if bg_indices.size > 0:
            bg_indices = np.random.choice(
                bg_indices, size=bg_rois_per_this_image, replace=False)

        # The indices that we're selecting (both foreground and background).
        keep_indices = np.append(fg_indices, bg_indices)

        # sample_rois
        sample_rois = rois[keep_indices]

        # locs
        # Compute offsets and scales to match sampled RoIs to the GTs.
        loc_normalize_mean = np.array(self.loc_normalize_mean, np.float32)
        loc_normalize_std = np.array(self.loc_normalize_std, np.float32)
        gt_roi_locs = bbox2loc(
            sample_rois, bboxes[gt_assignment[keep_indices]])
        gt_roi_locs = gt_roi_locs - loc_normalize_mean
        gt_roi_locs = gt_roi_locs / loc_normalize_std

        # masks
        gt_roi_masks = -1 * np.ones(
            (len(keep_indices), self.mask_size, self.mask_size),
            dtype=np.int32)

        for i, fg_index in enumerate(fg_indices):
            roi = np.round(sample_rois[i]).astype(np.int32)
            gt_roi = np.round(bboxes[gt_assignment[fg_index]])
            gt_roi = gt_roi.astype(np.int32)
            gt_mask = whole_mask[gt_assignment[fg_index]]
            gt_roi_mask = fcis.mask.intersect_bbox_mask(
                roi, gt_roi, gt_mask, self.mask_size)
            gt_roi_mask = cv2.resize(
                gt_roi_mask.astype(np.float32),
                (self.mask_size, self.mask_size))
            gt_roi_mask = gt_roi_mask >= self.binary_thresh
            gt_roi_mask = gt_roi_mask.astype(np.int32)
            gt_roi_masks[i, ...] = gt_roi_mask

        # labels
        # The label with value 0 is the background.
        gt_roi_labels = labels[gt_assignment[keep_indices]]
        # set labels of bg_rois to be 0
        gt_roi_labels[fg_rois_per_this_image:] = 0

        sample_rois = cuda.to_gpu(sample_rois)
        gt_roi_locs = cuda.to_gpu(gt_roi_locs)
        gt_roi_masks = cuda.to_gpu(gt_roi_masks)
        gt_roi_labels = cuda.to_gpu(gt_roi_labels)

        return sample_rois, gt_roi_locs, gt_roi_masks, gt_roi_labels
