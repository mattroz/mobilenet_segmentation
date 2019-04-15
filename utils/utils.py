import time
import skimage.io as io
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import keras.backend as K

# Function was forked from https://github.com/albu/albumentations/blob/master/notebooks/example_kaggle_salt.ipynb
def visualize(image, mask, original_image=None, original_mask=None, name=None, show=False):
    fontsize = 18

    if original_image is None and original_mask is None:
        f, ax = plt.subplots(2, 1, figsize=(8, 8))

        ax[0].imshow(image)
        ax[1].imshow(mask)
    else:
        f, ax = plt.subplots(2, 2, figsize=(8, 8))

        ax[0, 0].imshow(original_image)
        ax[0, 0].set_title('Original image', fontsize=fontsize)

        ax[1, 0].imshow(original_mask)
        ax[1, 0].set_title('Original mask', fontsize=fontsize)

        ax[0, 1].imshow(image)
        ax[0, 1].set_title('Original image', fontsize=fontsize)

        ax[1, 1].imshow(mask)
        ax[1, 1].set_title('Predicted mask', fontsize=fontsize)

    if name is not None:
        plt.savefig(name)

# IoU approximation for segmentaton: http://www.cs.umanitoba.ca/~ywang/papers/isvc16.pdf
def iou_metric(y_true, y_pred, smooth=1):
    y_pred = K.cast(y_pred, dtype=tf.float32)
    y_true = K.cast(y_true, dtype=tf.float32)
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2])
    union = K.sum(y_true, axis=[1,2]) + K.sum(y_pred, axis=[1,2])
    return K.mean((2. * intersection + smooth) / (union + smooth), axis=0)


def iou_for_image(y_true, y_pred, smooth=1):
    y_pred = K.get_value(y_pred)
    intersection = np.sum(np.abs(y_true * y_pred), axis=(1,2))
    union = np.sum(y_true, axis=(1,2)) + np.sum(y_pred, axis=(1,2))
    iou = (2. * intersection + smooth) / (union + smooth)
    return iou

def dice_loss(y_true, y_pred):
    return 1 - iou_metric(y_true, y_pred)


def bce_dice_loss(y_true, y_pred):
    loss = K.binary_crossentropy(y_true, y_pred) - K.log(iou_metric(y_true, y_pred))
    return loss

# Calculate IoU-over-threshold metric from Kaggle TGS Salt competition
# Excellent explanation: https://www.kaggle.com/pestipeti/explanation-of-scoring-metric
def get_precision(y_true, y_pred, threshold):
    # The main point is the following:
    #   - GT mask is empty, your prediction non-empty: (i.e. FP) ==> precision = 0
    #   - GT mask non empty, your prediction empty: (i.e. FN) ==> precision = 0
    #   - GT mask empty, your prediction empty: (i.e. TN) ==> precision = 1
    #   - GT mask non-empty, your prediction non-empty: (i.e. TP) ==> precision = [IoU(GT, pred) > threshold]
    # But here is the easiest case - binary segmentation, i.e. it is sufficient to calculate only IoU > threshold
    precision = np.asarray(iou_for_image(y_true, y_pred) > threshold, dtype=np.int8)
    return precision


def get_multi_threshold_precision(precision):
    # precision is [batch_size, n_of_thresholds]
    return np.mean(precision, axis=1)
