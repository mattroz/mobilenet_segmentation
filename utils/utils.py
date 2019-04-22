import time
import skimage.io as io
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import keras.backend as K

# Function was forked from https://github.com/albu/albumentations/blob/master/notebooks/example_kaggle_salt.ipynb
def visualize(image, mask, original_image=None, original_mask=None, name=None):
    """
    Function for two image-mask pairs visualizing.

    image: numpy array
        First image to visualize.
    mask: numpy array
        Mask for the first image.
    original_image: numpy array
        Second image to visualize
    original_mask: numpy array
        Mask for the second image
    name: str
        If specified, produced plot will be saved under this name
    """

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


# IoU approximation for segmentaton:
def iou_metric(y_true, y_pred, smooth=1):
    """Calculates Intersection-over-Union metric across batch axis.

    Paper: http://www.cs.umanitoba.ca/~ywang/papers/isvc16.pdf

    y_true: tf.Tensor
        Original labels (mask).
    y_pred: tf.Tensor
        Predicted labels (mask).
    smooth: int, float
        Smoothing constant for boundary cases.

    :return: tf.Tensor
        Mean IoU across batch axis.
    """

    y_pred = K.cast(y_pred, dtype=tf.float32)
    y_true = K.cast(y_true, dtype=tf.float32)
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2])
    union = K.sum(y_true, axis=[1,2]) + K.sum(y_pred, axis=[1,2])
    return K.mean((2. * intersection + smooth) / (union + smooth), axis=0)


def iou_for_image(y_true, y_pred, smooth=1):
    """Calculates Intersection-over-Union for two masks.

    y_true: tf.Tensor
        Original labels (mask).
    y_pred: tf.Tensor
        Predicted labels (mask).
    smooth: int, float
        Smoothing constant for boundary cases.

    :return: scalar
        IoU for two masks.
    """

    y_pred = K.get_value(y_pred)
    intersection = np.sum(np.abs(y_true * y_pred), axis=(1,2))
    union = np.sum(y_true, axis=(1,2)) + np.sum(y_pred, axis=(1,2))
    iou = (2. * intersection + smooth) / (union + smooth)
    return iou


def binary_focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    """Calculates binary focal loss.

    y_true: tf.Tensor
        Original labels (mask).
    y_pred: tf.Tensor
        Predicted labels (mask).
    alpha: float
        Tunable parameter.
    gamma: float
        Tunable parameter.

    :return: tf.Tensor.
    """

    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

    epsilon = K.epsilon()
    # clip to prevent NaN's and Inf's

    pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
    pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)

    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) \
           - K.sum((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))


def dice_loss(y_true, y_pred):
    """Calculates Dice loss, which is (1 - IoU)

    y_true: tf.Tensor
        Original labels (mask).
    y_pred: tf.Tensor
        Predicted labels (mask).

    :return: tf.Tensor.
    """

    return (1 - iou_metric(y_true, y_pred))


def bce_dice_loss(y_true, y_pred):
    """Calculates binary crossentropy dice loss.

    Calculates as [BCE - log(IoU)]

    y_true: tf.Tensor
        Original labels (mask).
    y_pred: tf.Tensor
        Predicted labels (mask).

    :return: tf.Tensor
    """

    loss = K.mean(K.binary_crossentropy(y_true, y_pred)) - K.log(iou_metric(y_true, y_pred))
    return loss


def focal_dice_loss(y_true, y_pred):
    """Calculates focal dice loss.

    Calculates as [focal_loss - log(IoU)]

    y_true: tf.Tensor
        Original labels (mask).
    y_pred: tf.Tensor
        Predicted labels (mask).

    :return: tf.Tensor
    """

    loss = binary_focal_loss(y_true, y_pred) - K.log(iou_metric(y_true, y_pred))
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
