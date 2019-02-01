# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Blob helper functions."""

import numpy as np
# from scipy.misc import imread, imresize
import cv2

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3


def im_list_to_blob(ims):
    """Convert a list of images into a network input.

    Assumes images are already prepared (means subtracted, BGR order, ...).
    """
    max_shape = np.array([im.shape for im in ims]).max(axis=0)
    num_images = len(ims)
    blob = np.zeros((num_images, max_shape[0], max_shape[1], 3),
                    dtype=np.float32)
    for i in xrange(num_images):
        im = ims[i]
        blob[i, 0:im.shape[0], 0:im.shape[1], :] = im

    return blob

def prep_im_for_blob(im, pixel_means, target_size, max_size, usecaffe=False, smallstd=False):
    """Mean subtract and scale an image for use in a blob."""
#    print("caffe?",usecaffe)
    if not usecaffe:
        im = im.astype(np.float32, copy=False)
        # changed to use pytorch models
        im /= 255. # Convert range to [0,1]
        pixel_means = [0.485, 0.456, 0.406]
        im -= pixel_means # Minus mean
        if not smallstd:
            pixel_stdens = [0.229, 0.224, 0.225]
        else:
            pixel_stdens=[0.00392156, 0.00392156, 0.00392156] #for res10
        im /= pixel_stdens # divide by stddev
    else:
        im = im.astype(np.float32, copy=False)
        im -= pixel_means
        # im = im[:, :, ::-1]
    
    # im = im[:, :, ::-1]
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    # if np.round(im_scale * im_size_max) > max_size:
    #     im_scale = float(max_size) / float(im_size_max)
    # im = imresize(im, im_scale)
    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
                    interpolation=cv2.INTER_LINEAR)

    return im, im_scale
