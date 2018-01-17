# Cuda Kernel Original work by Haozhi Qi (@Oh233)
# https://github.com/msracver/FCIS

# Modified by Shingo Kitagawa (@knorth55)

import numpy

from chainer import cuda
from chainer import function
from chainer.utils import type_check


class PSROIPooling2D(function.Function):

    def __init__(self, outh, outw, spatial_scale, group_size, output_dim):
        self.outh, self.outw = outh, outw
        self.spatial_scale = spatial_scale
        self.group_size = group_size
        self.output_dim = output_dim

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)

        x_type, roi_type = in_types
        type_check.expect(
            x_type.dtype == numpy.float32,
            x_type.ndim == 4,
            roi_type.dtype == numpy.float32,
            roi_type.ndim == 2,
            roi_type.shape[1] == 5,
        )

    def forward_gpu(self, inputs):
        self.retain_inputs((1,))
        self._bottom_data_shape = inputs[0].shape

        bottom_data, bottom_rois = inputs
        channels, height, width = bottom_data.shape[1:]
        n_rois = bottom_rois.shape[0]
        top_data = cuda.cupy.empty((n_rois, self.output_dim, self.outh,
                                    self.outw), dtype=numpy.float32)
        cuda.cupy.ElementwiseKernel(
            '''
            raw float32 bottom_data, float32 spatial_scale, int32 channels,
            int32 height, int32 width, int32 pooled_height, int32 pooled_width,
            int32 group_size, int32 output_dim, raw float32 bottom_rois
            ''',
            'float32 top_data',
            '''
            // pos in output filter
            int pw = i % pooled_width;
            int ph = (i / pooled_width) % pooled_height;
            int ctop = (i / pooled_width / pooled_height) % output_dim;
            int n = i / pooled_width / pooled_height / output_dim;

            int roi_batch_ind = bottom_rois[n * 5 + 0];
            float roi_start_w = static_cast<float>(
                round(bottom_rois[n * 5 + 1])) * spatial_scale;
            float roi_start_h = static_cast<float>(
                round(bottom_rois[n * 5 + 2])) * spatial_scale;
            float roi_end_w = static_cast<float>(
                round(bottom_rois[n * 5 + 3])) * spatial_scale;
            float roi_end_h = static_cast<float>(
                round(bottom_rois[n * 5 + 4])) * spatial_scale;

            // Force too small ROIs to be 1x1
            float roi_width = max(roi_end_w - roi_start_w, 0.1);  // avoid 0
            float roi_height = max(roi_end_h - roi_start_h, 0.1);

            // Compute w and h at bottom
            float bin_size_h = roi_height / static_cast<float>(pooled_height);
            float bin_size_w = roi_width / static_cast<float>(pooled_width);

            int hstart = static_cast<int>(floor(static_cast<float>(ph)
                                                * bin_size_h + roi_start_h));
            int wstart = static_cast<int>(floor(static_cast<float>(pw)
                                                * bin_size_w + roi_start_w));
            int hend = static_cast<int>(ceil(static_cast<float>(ph + 1)
                                             * bin_size_h + roi_start_h));
            int wend = static_cast<int>(ceil(static_cast<float>(pw + 1)
                                            * bin_size_w + roi_start_w));

            // Add roi offsets and clip to input boundaries
            hstart = min(max(hstart, 0), height);
            hend = min(max(hend, 0), height);
            wstart = min(max(wstart, 0), width);
            wend = min(max(wend, 0), width);
            bool is_empty = (hend <= hstart) || (wend <= wstart);

            // Compute c at bottom
            int gw = floor(
                static_cast<float>(pw) * group_size / pooled_width);
            int gh = floor(
                static_cast<float>(ph) * group_size / pooled_height);
            gw = min(max(gw, 0), group_size - 1);
            gh = min(max(gh, 0), group_size - 1);
            int c = (ctop * group_size + gh) * group_size + gw;

            int data_offset = (roi_batch_ind * channels + c) * height * width;
            float out_sum = 0;
            for (int h = hstart; h < hend; ++h){
              for (int w = wstart; w < wend; ++w){
                 int bottom_index = h * width + w;
                 out_sum += bottom_data[data_offset + bottom_index];
              }
            }

            float bin_area = (hend - hstart) * (wend - wstart);
            top_data = is_empty? (float) 0. : out_sum / bin_area;
            ''', 'psroi_pooling_2d_fwd'
        )(bottom_data, self.spatial_scale, channels, height, width,
          self.outh, self.outw, self.group_size, self.output_dim,
          bottom_rois, top_data)

        return top_data,

    def backward_gpu(self, inputs, gy):
        bottom_rois = inputs[1]
        channels, height, width = self._bottom_data_shape[1:]
        bottom_diff = cuda.cupy.zeros(self._bottom_data_shape, numpy.float32)
        cuda.cupy.ElementwiseKernel(
            '''
            raw float32 bottom_diff, int32 num_rois,
            float32 spatial_scale, int32 channels, int32 height, int32 width,
            int32 pooled_height, int32 pooled_width, int32 group_size,
            int32 output_dim, raw float32 bottom_rois
            ''',
            'float32 top_diff',
            '''
            int pw = i % pooled_width;
            int ph = (i / pooled_width) % pooled_height;
            int ctop = (i / pooled_width / pooled_height) % output_dim;
            int n = i / pooled_width / pooled_height / output_dim;

            // [start, end) interval for spatial sampling
            int roi_batch_ind = bottom_rois[n * 5];
            float roi_start_w = static_cast<float>(
                round(bottom_rois[n * 5 + 1])) * spatial_scale;
            float roi_start_h = static_cast<float>(
                round(bottom_rois[n * 5 + 2])) * spatial_scale;
            float roi_end_w = static_cast<float>(
                round(bottom_rois[n * 5 + 3])) * spatial_scale;
            float roi_end_h = static_cast<float>(
                round(bottom_rois[n * 5 + 4])) * spatial_scale;

            // Force too small ROIs to be 1x1
            float roi_width = max(roi_end_w - roi_start_w, 0.1); // avoid 0
            float roi_height = max(roi_end_h - roi_start_h, 0.1);

            // Compute w and h at bottom
            float bin_size_w = roi_width / static_cast<float>(pooled_width);
            float bin_size_h = roi_height / static_cast<float>(pooled_height);

            int wstart = floor(
                static_cast<float>(pw) * bin_size_w + roi_start_w);
            int hstart = floor(
                static_cast<float>(ph) * bin_size_h + roi_start_h);
            int wend = ceil(
                static_cast<float>(pw + 1.0) * bin_size_w + roi_start_w);
            int hend = ceil(
                static_cast<float>(ph + 1.0) * bin_size_h + roi_start_h);

            // Add roi offsets and clip to input boundaries
            wstart = min(max(wstart, 0), width);
            hstart = min(max(hstart, 0), height);
            wend = min(max(wend, 0), width);
            hend = min(max(hend, 0), height);
            bool is_empty = (hend <= hstart) || (wend <= wstart);

            // Compute c at bottom
            int gw = floor(
                static_cast<float>(pw) * group_size / pooled_width);
            int gh = floor(
                static_cast<float>(ph) * group_size / pooled_height);
            gw = min(max(gw, 0), group_size - 1);
            gh = min(max(gh, 0), group_size - 1);
            int c = (ctop * group_size + gh) * group_size + gw;

            int bottom_diff_offset = (roi_batch_ind * channels + c);
            bottom_diff_offset = bottom_diff_offset * height * width;
            float bin_area = (hend - hstart) * (wend - wstart);
            float diff_val = is_empty ? (float) 0. : top_diff / bin_area;
            for (int h = hstart; h < hend; ++h){
              for (int w = wstart; w < wend; ++w){
                int bottom_index = h * width + w;
                atomicAdd(
                    &bottom_diff[bottom_diff_offset + bottom_index], diff_val);
              }
            }
            ''', 'psroi_pooling_2d_bwd'
        )(bottom_diff, bottom_rois.shape[0], self.spatial_scale,
          channels, height, width, self.outh, self.outw, self.group_size,
          self.output_dim, bottom_rois, gy[0])

        return bottom_diff, None


def psroi_pooling_2d(
        x, rois, outh, outw, spatial_scale,
        group_size, output_dim
):
    return PSROIPooling2D(outh, outw, spatial_scale,
                          group_size, output_dim)(x, rois)
