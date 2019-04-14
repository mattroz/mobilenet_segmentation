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
    y_pred = K.cast(y_pred, dtype=tf.float32)
    y_true = K.cast(y_true, dtype=tf.float32)
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2])
    union = K.sum(y_true, axis=[1,2]) + K.sum(y_pred, axis=[1,2])
    return (2. * intersection + smooth) / (union + smooth) 


def dice_loss(y_true, y_pred):
    return 1 - iou_metric(y_true, y_pred)


def bce_dice_loss(y_true, y_pred):
    loss = K.binary_crossentropy(y_true, y_pred) - K.log(iou_metric(y_true, y_pred))
    return loss

# Calculate IoU-over-threshold metric from Kaggle TGS Salt competition
# Excellent explanation: https://www.kaggle.com/pestipeti/explanation-of-scoring-metric
def get_precision(y_true, y_pred, threshold):
    y_true_np = K.get_value(y_true)
    y_pred_np = np.where(K.get_value(y_pred) > threshold, 1, 0)

    # Get sum of pixels values ([batch, H, W])
    y_true_sum = np.sum(y_true_np, axis=[1,2])
    y_pred_sum = np.sum(y_pred_np, axis=[1,2])

    # If sum of pixels in ground truth is zero and sum of pixels in predicted != zero ==> false positive
    # false_positives = np.where(np.logical_and(y_true_sum == 0, y_pred_sum > 0), 1, 0)

    # If sum of pixels in GT is > 0 and sum of pixels in predicted is zero ==> false negative
    # false_negatives = np.where(np.logical_and(y_true_sum > 0, y_pred_sum == 0), 1, 0)

    # If sum of pixels in GT == 0 and the same for predicted ==> true negative
    true_negatives = np.where(np.logical_and(y_true_sum == 0, y_pred_sum == 0), 1, 0)

    # If sum of pixels in GT > 0 and the same is for predicted ==> measure IoU and cut off by threshold
    true_positives = np.where(np.logical_and(y_true_sum > 0, y_pred_sum > 0),
                              iou_metric(y_true, y_pred) > threshold,
                              0)

    # Now when we've obtained some statistics, it is time for precision calculating.
    # The main point is the following:
    #   - GT mask is empty, your prediction non-empty: (i.e. FP) ==> precision = 0
    #   - GT mask non empty, your prediction empty: (i.e. FN) ==> precision = 0
    #   - GT mask empty, your prediction empty: (i.e. TN) ==> precision = 1
    #   - GT mask non-empty, your prediction non-empty: (i.e. TP) ==> precision = [IoU(GT, pred) > threshold]
    precision = np.logical_or(true_negatives, true_positives)

    return precision


def get_multi_threshold_precision(precision):
    # precision is [batch_size, n_of_thresholds]
    return np.mean(precision, axis=1)
