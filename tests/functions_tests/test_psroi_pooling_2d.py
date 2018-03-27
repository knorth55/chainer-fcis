import unittest

import numpy

import chainer
from chainer import cuda
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition

from fcis import functions


class TestPSROIPolling2D(unittest.TestCase):

    def setUp(self):
        self.N = 3
        self.group_size = 7
        self.output_dim = 2
        self.n_channels = self.group_size * self.group_size * self.output_dim
        self.x = numpy.arange(
            self.N * self.n_channels * 30 * 40,
            dtype=numpy.float32).reshape((self.N, self.n_channels, 30, 40))
        numpy.random.shuffle(self.x)
        self.x = 2 * self.x / self.x.size - 1
        self.x = self.x.astype(numpy.float32)
        self.rois = numpy.array([
            [0, 1, 1, 6, 6],
            [2, 6, 2, 7, 11],
            [1, 3, 1, 5, 10],
            [0, 3, 3, 3, 3]
        ])
        self.rois = self.rois.astype(numpy.float32)
        self.n_rois = self.rois.shape[0]
        self.outh, self.outw = 21, 21
        self.spatial_scale = 1.0
        self.gy = numpy.random.uniform(
            -1, 1, (self.n_rois, self.output_dim,
                    self.outh, self.outw)).astype(numpy.float32)
        self.check_backward_options = {'atol': 5e-4, 'rtol': 5e-3}

    def check_forward(self, x_data, roi_data):
        x = chainer.Variable(x_data)
        rois = chainer.Variable(roi_data)
        y = functions.psroi_pooling_2d(
            x, rois, self.outh, self.outw,
            self.spatial_scale, self.group_size, self.output_dim)
        self.assertEqual(y.data.dtype, numpy.float32)
        y_data = cuda.to_cpu(y.data)
        self.assertEqual(
            (self.n_rois, self.output_dim, self.outh, self.outw), y_data.shape)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.x, self.rois)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x), cuda.to_gpu(self.rois))

    def check_backward(self, x_data, roi_data, y_grad):
        gradient_check.check_backward(
            functions.PSROIPooling2D(
                self.outh, self.outw,
                self.spatial_scale, self.group_size, self.output_dim),
            (x_data, roi_data), y_grad, no_grads=[False, True],
            **self.check_backward_options)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.rois, self.gy)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.rois),
                            cuda.to_gpu(self.gy))


testing.run_module(__name__, __file__)
