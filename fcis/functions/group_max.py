import numpy as np

from chainer import cuda
from chainer import function
from chainer.utils import type_check


class GroupMax(function.Function):

    def __init__(self, group):
        self.group = group

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)

        x_type = in_types[0]
        type_check.expect(
            x_type.dtype == np.float32,
            x_type.ndim == 4
        )

    def forward_cpu(self, inputs):
        # NOT IMPLEMENTED YET
        return

    def forward_gpu(self, inputs):
        self._bottom_data_shape = inputs[0].shape

        bottom_data = inputs[0]
        channels, height, width = bottom_data.shape[1:]
        channels_in_group = channels / self.group
        spatial_dim = height * width
        top_data = cuda.cupy.empty((self.group, height, width), np.float32)
        self.max_idx_data = cuda.cupy.zeros(top_data.shape, np.int32)

        cuda.cupy.ElementwiseKernel(
            '''
            raw float32 bottom_data, int32 channels, int32 group,
            int32 channels_in_group, int32 spatial_dim
            ''',
            'float32 top_data, int32 max_idx_data',
            '''
            int s = index % spatial_dim;
            int g = (index / spatial_dim) % group;
            int n = index / spatial_dim / group;

            float max_val = -FLT_MAX;
            int max_idx = -1;
            for (int k = 0; k < channels_in_group; ++k) {
              int c = g*channels_in_group + k;
              int bottom_index = (n*channels + c)*spatial_dim + s;
              if (bottom_data[bottom_index]>max_val) {
                max_val = bottom_data[bottom_index];
                max_idx = c;
              }
            }
            top_data = max_val;
            max_idx_data = max_idx;
            '''
        )(bottom_data, channels, self.group, channels_in_group,
          spatial_dim, top_data, self.max_idx_data)

        return top_data,

    def backward_cpu(self, inputs, gy):
        # NOT IMPLEMENTED YET
        return

    def backward_gpu(self, inputs, gy):
        channels, height, width = self._bottom_data_shape[1:]
        spatial_dim = height * width
        bottom_diff = cuda.cupy.zeros(self._bottom_data_shape, np.float32)
        cuda.cupy.ElementwiseKernel(
            '''
            raw float32 top_diff, raw int32 max_idx_data,
            int32 channels, int32 group, int32 spatial_dim
            ''',
            'float32 bottom_diff',
            '''
            int s = i % spatial_dim;
            int n = i / spatial_dim / group;

            int c = max_idx_data[i];
            int bottom_index = (n*channels + c)*spatial_dim + s;
            bottom_diff[bottom_index] = top_diff[i];
            '''
        )(gy[0], self.max_idx_data, channels, self.group, spatial_dim,
          bottom_diff)

        return bottom_diff,
