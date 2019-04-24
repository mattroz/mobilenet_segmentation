import numpy as np
import tensorflow as tf
import keras.backend as K


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


"""
Lovasz-Softmax and Jaccard hinge loss in Tensorflow
Maxim Berman 2018 ESAT-PSI KU Leuven (MIT License)
"""

def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    gts = tf.reduce_sum(gt_sorted)
    intersection = gts - tf.cumsum(gt_sorted)
    union = gts + tf.cumsum(1. - gt_sorted)
    jaccard = 1. - intersection / union
    jaccard = tf.concat((jaccard[0:1], jaccard[1:] - jaccard[:-1]), 0)
    return jaccard


# --------------------------- BINARY LOSSES ---------------------------


def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        def treat_image(log_lab):
            log, lab = log_lab
            log, lab = tf.expand_dims(log, 0), tf.expand_dims(lab, 0)
            log, lab = flatten_binary_scores(log, lab, ignore)
            return lovasz_hinge_flat(log, lab)

        losses = tf.map_fn(treat_image, (logits, labels), dtype=tf.float32)

        # Fixed python3
        losses.set_shape((None,))

        loss = tf.reduce_mean(losses)
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss


def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """

    def compute_loss():
        labelsf = tf.cast(labels, logits.dtype)
        signs = 2. * labelsf - 1.
        errors = 1. - logits * tf.stop_gradient(signs)
        errors_sorted, perm = tf.nn.top_k(errors, k=tf.shape(errors)[0], name="descending_sort")
        gt_sorted = tf.gather(labelsf, perm)
        grad = lovasz_grad(gt_sorted)
        # loss = tf.tensordot(tf.nn.relu(errors_sorted), tf.stop_gradient(grad), 1, name="loss_non_void")
        # ELU + 1
        loss = tf.tensordot(tf.nn.elu(errors_sorted) + 1., tf.stop_gradient(grad), 1, name="loss_non_void")
        return loss

    # deal with the void prediction case (only void pixels)
    loss = tf.cond(tf.equal(tf.shape(logits)[0], 0),
                   lambda: tf.reduce_sum(logits) * 0.,
                   compute_loss,
                   strict=True,
                   name="loss"
                   )
    return loss


def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = tf.reshape(scores, (-1,))
    labels = tf.reshape(labels, (-1,))
    if ignore is None:
        return scores, labels
    valid = tf.not_equal(labels, ignore)
    vscores = tf.boolean_mask(scores, valid, name='valid_scores')
    vlabels = tf.boolean_mask(labels, valid, name='valid_labels')
    return vscores, vlabels

def lovasz_hinge_loss(y_true, y_pred):
    return lovasz_hinge(y_pred, y_true, per_image=False, ignore=None)