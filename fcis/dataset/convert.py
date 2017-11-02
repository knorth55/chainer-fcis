import six

from chainer.dataset.convert import _concat_arrays
from chainer.dataset.convert import to_device


def concat_examples(batch, device=None):
    # batch: img, bboxes, whole_mask, labels, scale
    if len(batch) == 0:
        raise ValueError('batch is empty')

    first_elem = batch[0]

    result = []
    for i in six.moves.range(len(first_elem)):
        if i == 0:  # img
            result.append(to_device(device, _concat_arrays(
                [example[i] for example in batch])))
        else:
            result.append(_concat_arrays([example[i] for example in batch]))
    return tuple(result)
