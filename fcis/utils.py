import cv2
import numpy as np

import fcn


def visualize_mask(
        img, whole_masks, bboxes, labels, cls_probs,
        label_names, alpha=0.7, bbox_alpha=0.7, ax=None):
    import matplotlib.pyplot as plt
    viz_img = img.copy()
    viz_img = viz_img.astype(np.float)
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
    ax.cla()
    ax.axis("off")

    cmap = fcn.utils.label_colormap(len(bboxes) + 1)
    cmap = cmap[1:]
    for color, l, whole_mask, bbox, cls_prob in zip(
            cmap, labels, whole_masks, bboxes, cls_probs):
        color_uint8 = color * 255.0
        bbox = np.round(bbox).astype(np.int32)
        y_min, x_min, y_max, x_max = bbox
        if y_max > y_min and x_max > x_min:
            mask = whole_mask2mask(whole_mask[None], bbox[None])[0]
            mask = mask.astype(np.int32)
            mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
            colored_mask = alpha * mask * color_uint8
            sub_img = alpha * mask * viz_img[y_min:y_max, x_min:x_max, :]
            viz_img[y_min:y_max, x_min:x_max, :] += colored_mask
            viz_img[y_min:y_max, x_min:x_max, :] -= sub_img
        ax.text((x_max + x_min) / 2, y_min,
                '{:s} {:.3f}'.format(label_names[l], cls_prob),
                bbox={'facecolor': color, 'alpha': bbox_alpha},
                fontsize=8, color='white')
    viz_img = viz_img.astype(np.uint8)
    ax.imshow(viz_img)
    return ax


def mask2whole_mask(mask, bbox, size):
    """Convert list representation of instance masks to an image-sized array.

    Args:
        mask (list): [(H_1, W_1), ..., (H_R, W_R)]
        bbox (array): Array of shape (R, 4)
        size (tuple of ints): (H, W)

    Returns:
        array of shape (R, H, W)

    """
    if len(mask) != len(bbox):
        raise ValueError('The length of mask and bbox should be the same')
    R = len(mask)
    H, W = size
    whole_mask = np.zeros((R, H, W), dtype=np.bool)

    for i, (m, bb) in enumerate(zip(mask, bbox)):
        bb = np.round(bb).astype(np.int32)
        whole_mask[i, bb[0]:bb[2], bb[1]:bb[3]] = m
    return whole_mask


def whole_mask2mask(whole_mask, bbox):
    """Convert an image-sized array of instance masks into a list.

    Args:
        whole_mask (array): array of shape (R, H, W)
        bbox (array): Array of shape (R, 4)

    Returns:
        [(H_1, W_1), ..., (H_R, W_R)]

    """
    if len(whole_mask) != len(bbox):
        raise ValueError(
            'The length of whole_mask and bbox should be the same')
    mask = list()
    for whole_m, bb in zip(whole_mask, bbox):
        bb = np.round(bb).astype(np.int32)
        mask.append(whole_m[bb[0]:bb[2], bb[1]:bb[3]])
    return mask


def whole_mask2label_mask(mask):
    _, H, W = mask.shape
    label_mask = np.zeros((H, W), dtype=np.int32)
    for i, m in enumerate(mask):
        # label: 0 is for background
        label = i + 1
        label_mask[m] = label
    return label_mask


def label_mask2whole_mask(label_mask):
    H, W = label_mask.shape
    mask = np.zeros((label_mask.max() + 1, label_mask.size))
    mask[label_mask.ravel(), np.arange(label_mask.size)] = 1
    mask = mask.reshape((-1, H, W))
    # label: 0 is for background
    mask = mask[1:, :, :]
    return mask


def read_images(imgpaths, channel_order='BGR'):
    imgs = []
    for imgpath in imgpaths:
        img = cv2.imread(imgpath, cv2.IMREAD_COLOR)
        if channel_order == 'RGB':
            img = img[:, :, ::-1]
        imgs.append(img)
    return imgs


def get_resize_scale(shape, target_height, max_width):
    H, W = shape
    resize_scale = target_height / float(H)
    if W * resize_scale > max_width:
        resize_scale = max_width / float(W)
    return resize_scale


def resize_image(img, target_height, max_width):
    resize_scale = get_resize_scale(
        img.shape[:2], target_height, max_width)
    resized_img = cv2.resize(
        img, None, None,
        fx=resize_scale, fy=resize_scale,
        interpolation=cv2.INTER_LINEAR)
    return resized_img


def flip_mask(mask, x_flip=False, y_flip=False):
    if y_flip:
        mask = mask[:, ::-1, :]
    if x_flip:
        mask = mask[:, :, ::-1]
    return mask


def mask_probs2mask(mask_probs, bboxes, binary_thresh=0.4):
    masks = []
    for mask_prob, bbox in zip(mask_probs, bboxes):
        bbox = np.round(bbox).astype(np.int32)
        y_min, x_min, y_max, x_max = bbox
        mask = cv2.resize(
            mask_prob.astype(np.float32), (x_max - x_min, y_max - y_min))
        mask = mask >= binary_thresh
        masks.append(mask)
    return masks
