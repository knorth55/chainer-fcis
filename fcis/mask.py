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
        rois, roi_cls_probs, roi_mask_probs,
        n_class, H, W,
        score_thresh=0.7,
        nms_thresh=0.3,
        mask_merge_thresh=0.5,
        binary_thresh=0.4):

    cls_probs = []
    bboxes = []
    mask_size = roi_mask_probs.shape[-1]

    for l in range(1, n_class):
        # shape: (n_rois,)
        roi_cls_probs_l = roi_cls_probs[:, l]
        thresh_mask = roi_cls_probs_l >= 0.001
        rois_l = rois[thresh_mask]
        roi_cls_probs_l = roi_cls_probs_l[thresh_mask]
        keep = non_maximum_suppression(
            rois_l, nms_thresh, roi_cls_probs_l, limit=100)
        bbox = rois_l[keep]
        bboxes.append(bbox)
        cls_probs.append(roi_cls_probs_l[keep])

    voted_cls_probs = []
    voted_masks = []
    voted_bboxes = []
    for l in range(0, n_class - 1):
        n_bboxes = len(bboxes[l])
        voted_mask = np.zeros((n_bboxes, mask_size, mask_size))
        voted_bbox = np.zeros((n_bboxes, 4))

        for i, bbox in enumerate(bboxes[l]):
            iou = bbox_iou(rois, bbox[np.newaxis, :])
            idx = np.where(iou > mask_merge_thresh)[0]
            mask_weights = roi_cls_probs[idx, l + 1]
            mask_weights = mask_weights / mask_weights.sum()
            mask_probs_l = [roi_mask_probs[j] for j in idx.tolist()]
            rois_l = rois[idx]
            orig_mask, voted_bbox[i] = mask_aggregation(
                rois_l, mask_probs_l, mask_weights, H, W, binary_thresh)
            voted_mask[i] = cv2.resize(
                orig_mask.astype(np.float32), (mask_size, mask_size))

        cls_probs_l = cls_probs[l]
        score_thresh_mask = cls_probs_l > score_thresh
        voted_cls_probs_l = cls_probs_l[score_thresh_mask]
        voted_mask = voted_mask[score_thresh_mask]
        voted_bbox = voted_bbox[score_thresh_mask]
        voted_cls_probs.append(voted_cls_probs_l)
        voted_masks.append(voted_mask)
        voted_bboxes.append(voted_bbox)
    return voted_masks, voted_bboxes, voted_cls_probs
