import numpy
import six

from chainer import cuda


def to_device(device, x):
    """Send an array to a given device.

    This method sends a given array to a given device. This method is used in
    :func:`~chainer.dataset.concat_examples`.
    You can also use this method in a custom converter method used in
    :class:`~chainer.training.Updater` and :class:`~chainer.training.Extension`
    such as :class:`~chainer.training.StandardUpdater` and
    :class:`~chainer.training.extensions.Evaluator`.

    See also :func:`chainer.dataset.concat_examples`.

    Args:
        device (int or None): Device ID to which an array is sent. If it is
            negative value, an array is sent to CPU. If it is positive, an
            array is sent to GPU with the given ID. If it is ``None``, an
            array is left in the original device.
        x (numpy.ndarray or cupy.ndarray): An array to send.

    Returns:
        Converted array.

    """
    if device is None:
        return x
    elif device < 0:
        return cuda.to_cpu(x)
    else:
        return cuda.to_gpu(x, device, cuda.Stream.null)


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


def _concat_arrays(arrays):
    # Convert `arrays` to numpy.ndarray if `arrays` consists of the built-in
    # types such as int or float.
    if not isinstance(arrays[0], numpy.ndarray) and\
       not isinstance(arrays[0], cuda.ndarray):
        arrays = numpy.asarray(arrays)

    xp = cuda.get_array_module(arrays[0])
    with cuda.get_device_from_array(arrays[0]):
        return xp.concatenate([array[None] for array in arrays])
