# Modified by:
# Shingo Kitagawa (@knorth55)
#
# Modified by:
# Kentaro Wada (@wkentaro)
#
# Original work by:
# Yusuke Niitani (@yuyu2172)
# https://github.com/chainer/chainercv/blob/master/chainercv/evaluations/eval_detection_voc.py  # NOQA

from __future__ import division

from collections import defaultdict
import itertools

from chainercv.evaluations import calc_detection_voc_ap
import fcis
import numpy as np
import six


def calc_instseg_voc_prec_rec(
        sizes, pred_bboxes, pred_masks, pred_labels, pred_scores,
        gt_bboxes, gt_masks, gt_labels,
        gt_difficults=None, iou_thresh=0.5):

    sizes = iter(sizes)
    pred_bboxes = iter(pred_bboxes)
    pred_masks = iter(pred_masks)
    pred_labels = iter(pred_labels)
    pred_scores = iter(pred_scores)
    gt_bboxes = iter(gt_bboxes)
    gt_masks = iter(gt_masks)
    gt_labels = iter(gt_labels)
    if gt_difficults is None:
        gt_difficults = itertools.repeat(None)
    else:
        gt_difficults = iter(gt_difficults)

    n_pos = defaultdict(int)
    score = defaultdict(list)
    match = defaultdict(list)

    for (size, pred_bbox, pred_mask, pred_label, pred_score,
         gt_bbox, gt_mask, gt_label, gt_difficult) in \
            six.moves.zip(
                sizes, pred_bboxes, pred_masks, pred_labels, pred_scores,
                gt_bboxes, gt_masks, gt_labels, gt_difficults):

        pred_whole_mask = fcis.utils.mask2whole_mask(
            pred_mask, pred_bbox, size)
        gt_whole_mask = fcis.utils.mask2whole_mask(gt_mask, gt_bbox, size)
        if gt_difficult is None:
            gt_difficult = np.zeros(gt_whole_mask.shape[0], dtype=bool)

        for l in np.unique(np.concatenate((pred_label, gt_label)).astype(int)):
            pred_keep_l = pred_label == l
            pred_whole_mask_l = pred_whole_mask[pred_keep_l]
            pred_score_l = pred_score[pred_keep_l]
            # sort by score
            order = pred_score_l.argsort()[::-1]
            pred_whole_mask_l = pred_whole_mask_l[order]
            pred_score_l = pred_score_l[order]

            gt_keep_l = gt_label == l
            gt_whole_mask_l = gt_whole_mask[gt_keep_l]
            gt_difficult_l = gt_difficult[gt_keep_l]

            n_pos[l] += np.logical_not(gt_difficult_l).sum()
            score[l].extend(pred_score_l)

            if len(pred_whole_mask_l) == 0:
                continue
            if len(gt_whole_mask_l) == 0:
                match[l].extend((0,) * pred_whole_mask_l.shape[0])
                continue

            iou = fcis.mask.mask_iou(pred_whole_mask_l, gt_whole_mask_l)
            gt_index = iou.argmax(axis=1)
            # set -1 if there is no matching ground truth
            gt_index[iou.max(axis=1) < iou_thresh] = -1
            del iou

            selec = np.zeros(gt_whole_mask_l.shape[0], dtype=bool)
            for gt_idx in gt_index:
                if gt_idx >= 0:
                    if gt_difficult_l[gt_idx]:
                        match[l].append(-1)
                    else:
                        if not selec[gt_idx]:
                            match[l].append(1)
                        else:
                            match[l].append(0)
                    selec[gt_idx] = True
                else:
                    match[l].append(0)

    for iter_ in (
            pred_masks, pred_labels, pred_scores,
            gt_masks, gt_labels, gt_difficults):
        if next(iter_, None) is not None:
            raise ValueError('Length of input iterables need to be same.')

    n_fg_class = max(n_pos.keys()) + 1
    prec = [None] * n_fg_class
    rec = [None] * n_fg_class

    for l in n_pos.keys():
        score_l = np.array(score[l])
        match_l = np.array(match[l], dtype=np.int8)

        order = score_l.argsort()[::-1]
        match_l = match_l[order]

        tp = np.cumsum(match_l == 1)
        fp = np.cumsum(match_l == 0)

        # If an element of fp + tp is 0,
        # the corresponding element of prec[l] is nan.
        prec[l] = tp / (fp + tp)
        # If n_pos[l] is 0, rec[l] is None.
        if n_pos[l] > 0:
            rec[l] = tp / n_pos[l]

    return prec, rec


def eval_instance_segmentation_voc(
        sizes, pred_bboxes, pred_masks, pred_labels, pred_scores,
        gt_bboxes, gt_masks, gt_labels, gt_difficults=None,
        iou_thresh=0.5, use_07_metric=False):

    prec, rec = calc_instseg_voc_prec_rec(
        sizes, pred_bboxes, pred_masks, pred_labels, pred_scores,
        gt_bboxes, gt_masks, gt_labels, gt_difficults,
        iou_thresh=iou_thresh)

    ap = calc_detection_voc_ap(prec, rec, use_07_metric=use_07_metric)

    return {'ap': ap, 'map': np.nanmean(ap)}
