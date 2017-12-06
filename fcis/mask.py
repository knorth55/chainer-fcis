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
            mask_prob.astype(np.float32),
            (x_max - x_min, y_max - y_min))
        mask_mask = (mask_prob >= binary_thresh).astype(np.float)
        mask[y_min:y_max, x_min:x_max] += mask_mask * mask_weight

    y_idx, x_idx = np.where(mask >= binary_thresh)
    if len(y_idx) == 0 or len(x_idx) == 0:
        return None, None
    else:
        new_y_max = y_idx.max() + 1
        new_x_max = x_idx.max() + 1
        new_y_min = y_idx.min()
        new_x_min = x_idx.min()

        clipped_mask = mask[new_y_min:new_y_max, new_x_min:new_x_max]
        clipped_bbox = np.array(
            [new_y_min, new_x_min, new_y_max, new_x_max],
            dtype=np.float32)
        return clipped_bbox, clipped_mask


def mask_voting(
        rois, mask_probs, cls_probs,
        n_class, H, W,
        score_thresh=0.7,
        nms_thresh=0.3,
        mask_merge_thresh=0.5,
        binary_thresh=0.4, max_num=100):

    mask_size = mask_probs.shape[-1]
    v_labels = np.empty((0, ), dtype=np.int32)
    v_masks = np.empty((0, mask_size, mask_size), dtype=np.float32)
    v_bboxes = np.empty((0, 4), dtype=np.float32)
    v_cls_probs = np.empty((0, ), dtype=np.float32)

    tmp_all_scores = np.empty((0, ), dtype=np.float32)
    tmp_cls_probs = []
    tmp_bbox = []
    for label in range(0, n_class):
        if label == 0:
            # l == 0 is background
            continue
        # non maximum suppression
        cls_prob_l = cls_probs[:, label]
        keep_indices = non_maximum_suppression(
            rois, nms_thresh, cls_prob_l, limit=max_num)
        bbox_l = rois[keep_indices]
        cls_prob_l = cls_prob_l[keep_indices]
        tmp_bbox.append(bbox_l)
        tmp_cls_probs.append(cls_prob_l)
        tmp_all_scores = np.concatenate((tmp_all_scores, cls_prob_l))

    sorted_all_scores = np.sort(tmp_all_scores)[::-1]
    keep_num = min(len(sorted_all_scores), max_num)
    thresh = max(sorted_all_scores[keep_num - 1], 1e-3)

    for label in range(0, n_class):
        if label == 0:
            continue
        bbox_l = tmp_bbox[label - 1]
        cls_prob_l = tmp_cls_probs[label - 1]
        keep_indices = np.where(cls_prob_l >= thresh)
        bbox_l = bbox_l[keep_indices]
        cls_prob_l = cls_prob_l[keep_indices]

        v_mask_l = np.empty((0, mask_size, mask_size), dtype=np.float32)
        v_bbox_l = np.empty((0, 4), dtype=np.float32)
        v_cls_prob_l = np.empty((0, ), dtype=np.float32)

        for i, bbox in enumerate(bbox_l):
            iou = bbox_iou(rois, bbox[np.newaxis, :])
            idx = np.where(iou >= mask_merge_thresh)[0]
            mask_weights = cls_probs[idx, label]
            mask_weights = mask_weights / mask_weights.sum()
            mask_prob_l = mask_probs[idx]
            rois_l = rois[idx]
            clipped_bbox, clipped_mask = mask_aggregation(
                rois_l, mask_prob_l, mask_weights, H, W, binary_thresh)
            if clipped_bbox is not None and clipped_mask is not None:
                clipped_mask = cv2.resize(
                    clipped_mask.astype(np.float32),
                    (mask_size, mask_size))
                v_mask_l = np.concatenate((v_mask_l, clipped_mask[None]))
                v_bbox_l = np.concatenate((v_bbox_l, clipped_bbox[None]))
                v_cls_prob_l = np.concatenate(
                    (v_cls_prob_l, cls_prob_l[i][None]))

        keep_indices = v_cls_prob_l > score_thresh
        v_mask_l = v_mask_l[keep_indices]
        v_bbox_l = v_bbox_l[keep_indices]
        v_cls_prob_l = v_cls_prob_l[keep_indices]

        v_label_l = np.repeat(label, v_bbox_l.shape[0])
        v_masks = np.concatenate((v_masks, v_mask_l))
        v_bboxes = np.concatenate((v_bboxes, v_bbox_l))
        v_labels = np.concatenate((v_labels, v_label_l))
        v_cls_probs = np.concatenate((v_cls_probs, v_cls_prob_l))
    return v_bboxes, v_masks, v_labels, v_cls_probs


def intersect_bbox_mask(bbox, gt_bbox, gt_mask, mask_size=21):
    min_y = max(bbox[0], gt_bbox[0])
    min_x = max(bbox[1], gt_bbox[1])
    max_y = min(bbox[2], gt_bbox[2])
    max_x = min(bbox[3], gt_bbox[3])

    if min_y > max_y or min_x > max_x:
        return np.zeros((mask_size, mask_size))

    h = max_y - min_y
    w = max_x - min_x
    start_y = min_y - bbox[0]
    start_x = min_x - bbox[1]
    end_y = start_y + h
    end_x = start_x + w

    gt_roi_mask = np.zeros((bbox[2] - bbox[0], bbox[3] - bbox[1]))
    gt_clipped_mask = gt_mask[min_y:max_y, min_x:max_x]
    gt_roi_mask[start_y:end_y, start_x:end_x] = gt_clipped_mask
    return gt_roi_mask


def get_mask_overlap(mask1, mask2):
    intersect = np.bitwise_and(mask1, mask2).sum()
    union = np.bitwise_or(mask1, mask2).sum()
    return 1.0 * intersect / union


def mask_iou(mask1, mask2):
    if mask1.shape[1:] != mask2.shape[1:]:
        raise ValueError

    n_mask1 = len(mask1)
    n_mask2 = len(mask2)
    iou = np.zeros((n_mask1, n_mask2), dtype=np.float64)
    for i, m1 in enumerate(mask1):
        for j, m2 in enumerate(mask2):
            ov = get_mask_overlap(m1, m2)
            iou[i, j] = ov
    return iou
