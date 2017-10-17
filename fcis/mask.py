from chainercv.utils.bbox.bbox_iou import bbox_iou
from chainercv.utils import non_maximum_suppression
import cv2
import numpy as np


def mask_aggregation(
        bboxes, mask_probs, mask_weights,
        H, W, binary_thresh):
    assert bboxes.shape[0] == len(mask_probs)
    assert bboxes.shape[0] == mask_weights.shape[0]
    mask = np.zeros((H, W))
    for bbox, mask_prob, mask_weight in zip(bboxes, mask_probs, mask_weights):
        bbox = np.round(bbox).astype(np.int)
        y_min, x_min, y_max, x_max = bbox
        mask_prob = cv2.resize(
            mask_prob, (x_max - x_min, y_max - y_min))
        mask_mask = (mask_prob >= binary_thresh).astype(np.float)
        mask[y_min:y_max, x_min:x_max] += mask_mask * mask_weight

    y_idx, x_idx = np.where(mask >= binary_thresh)
    if len(y_idx) == 0 or len(x_idx) == 0:
        new_y_min = np.ceil(H / 2.0).astype(np.int)
        new_x_min = np.ceil(W / 2.0).astype(np.int)
        new_y_max = new_y_min + 1
        new_x_max = new_x_min + 1
    else:
        new_y_min = y_idx.min()
        new_x_min = x_idx.min()
        new_y_max = y_idx.max() + 1
        new_x_max = x_idx.max() + 1

    clipped_mask = mask[new_y_min:new_y_max, new_x_min:new_x_max]
    clipped_bbox = np.array([new_y_min, new_x_min, new_y_max, new_x_max],
                            dtype=np.float32)
    return clipped_mask, clipped_bbox


def mask_voting(
        rois, cls_probs, mask_probs,
        n_class, H, W,
        score_thresh=0.7,
        nms_thresh=0.3,
        mask_merge_thresh=0.5,
        binary_thresh=0.4):

    mask_size = mask_probs.shape[-1]
    v_labels = np.empty((0, ), dtype=np.int32)
    v_masks = np.empty((0, mask_size, mask_size), dtype=np.float32)
    v_bboxes = np.empty((0, 4), dtype=np.float32)
    v_cls_probs = np.empty((0, ), dtype=np.float32)

    for l in range(0, n_class - 1):
        # non maximum suppression
        cls_prob_l = cls_probs[:, l+1]
        thresh_mask = cls_prob_l >= 0.001
        bbox_l = rois[thresh_mask]
        cls_prob_l = cls_prob_l[thresh_mask]
        keep = non_maximum_suppression(
            bbox_l, nms_thresh, cls_prob_l, limit=100)
        bbox_l = bbox_l[keep]
        cls_prob_l = cls_prob_l[keep]

        n_bbox_l = len(bbox_l)
        v_mask_l = np.zeros((n_bbox_l, mask_size, mask_size))
        v_bbox_l = np.zeros((n_bbox_l, 4))

        for i, bbox in enumerate(bbox_l):
            iou = bbox_iou(rois, bbox[np.newaxis, :])
            idx = np.where(iou > mask_merge_thresh)[0]
            mask_weights = cls_probs[idx, l + 1]
            mask_weights = mask_weights / mask_weights.sum()
            mask_prob_l = mask_probs[idx]
            rois_l = rois[idx]
            clipped_mask, v_bbox_l[i] = mask_aggregation(
                rois_l, mask_prob_l, mask_weights, H, W, binary_thresh)
            v_mask_l[i] = cv2.resize(
                clipped_mask.astype(np.float32), (mask_size, mask_size))

        score_thresh_mask = cls_prob_l > score_thresh
        v_cls_prob_l = cls_prob_l[score_thresh_mask]
        v_mask_l = v_mask_l[score_thresh_mask]
        v_bbox_l = v_bbox_l[score_thresh_mask]
        v_label_l = np.repeat(l, v_bbox_l.shape[0])
        v_cls_probs = np.concatenate((v_cls_probs, v_cls_prob_l))
        v_masks = np.concatenate((v_masks, v_mask_l))
        v_bboxes = np.concatenate((v_bboxes, v_bbox_l))
        v_labels = np.concatenate((v_labels, v_label_l))
    return v_labels, v_masks, v_bboxes, v_cls_probs
