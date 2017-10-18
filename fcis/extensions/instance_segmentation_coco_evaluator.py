import copy

from chainer import reporter
import chainer.training.extensions
from chainercv.utils import apply_prediction_to_iterator
from fcis.evaluations import eval_instance_segmentation_coco


class InstanceSegmentationCOCOEvaluator(chainer.training.extensions.Evaluator):
    trigger = 1, 'epoch'
    default_name = 'validation'
    priority = chainer.training.PRIORITY_WRITER

    def __init__(self, iterator, target, label_names=None):
        super(InstanceSegmentationCOCOEvaluator, self).__init__(
            iterator, target)
        self.label_names = label_names

    def evaluate(self):
        iterator = self._iterators['main']
        target = self._targets['main']

        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)

        imgs, pred_values, gt_values = apply_prediction_to_iterator(
            target.predict, it)
        # delete unused iterator explicitly
        sizes = [img.shape for img in imgs]
        del imgs

        pred_bboxes, pred_masks, pred_labels, pred_scores = pred_values
        gt_bboxes, gt_masks, gt_labels = gt_values

        result = eval_instance_segmentation_coco(
            sizes, pred_bboxes, pred_masks, pred_labels, pred_scores,
            gt_bboxes, gt_masks, gt_labels)

        report = {
            'mAP[0.50:0.95]': result['ap/iou=0.50:0.95/area=all/maxDets=100'],
            'mAP[0.50:]': result['ap/iou=0.50/area=all/maxDets=100'],
            'mAP[0.50:0.95] (small)': result['ap/iou=0.50:0.95/area=small/maxDets=100'],  # NOQA
            'mAP[0.50:0.95] (mid)': result['ap/iou=0.50:0.95/area=medium/maxDets=100'],  # NOQA
            'mAP[0.50:0.95] (large)': result['ap/iou=0.50:0.95/area=large/maxDets=100'],  # NOQA
        }

        observation = dict()
        with reporter.report_scope(observation):
            reporter.report(report, target)
        return observation
