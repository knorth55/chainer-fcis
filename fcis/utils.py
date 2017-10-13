import cv2
import fcn
import matplotlib.pyplot as plt
import numpy as np


def visualize_mask(
        img, labels, masks, bboxes, scores,
        label_names, binary_thresh,
        alpha=0.7, bbox_alpha=0.7, ax=None):

    viz_img = img.copy()
    viz_img = viz_img.astype(np.float)
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
    ax.cla()
    ax.axis("off")

    cmap = fcn.utils.label_colormap(len(bboxes))
    for color, l, mask, bbox, score in zip(
            cmap, labels, masks, bboxes, scores):
        color_uint8 = color * 255.0
        bbox = bbox.astype(np.int32)
        y_min, x_min, y_max, x_max = bbox
        if y_max > y_min and x_max > x_min:
            mask = cv2.resize(mask, (x_max - x_min, y_max - y_min))
            mask = (mask >= binary_thresh).astype(np.int32)
            mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
            colored_mask = alpha * mask * color_uint8
            sub_img = alpha * mask * viz_img[y_min:y_max, x_min:x_max, :]
            viz_img[y_min:y_max, x_min:x_max, :] += colored_mask
            viz_img[y_min:y_max, x_min:x_max, :] -= sub_img
        ax.text((x_max + x_min) / 2, y_min,
                '{:s} {:.3f}'.format(label_names[l], score),
                bbox={'facecolor': color, 'alpha': bbox_alpha},
                fontsize=8, color='white')
    viz_img = viz_img.astype(np.uint8)
    ax.imshow(viz_img)
    return ax
