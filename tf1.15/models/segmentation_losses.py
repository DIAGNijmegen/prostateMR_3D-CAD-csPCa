from __future__ import unicode_literals, absolute_import, print_function, division
import numpy as np
import tensorflow as tf


'''
Prostate Cancer Detection in bpMRI
Script:         TF Segmentation Losses
Contributor:    anindox8
Target Organ:   Prostate
Target Classes: Benign(0), Malignant(1)
Update:         18/08/2020
Reference:      https://niftynet.readthedocs.io/en/dev/index.html
                https://dltk.github.io/DLTK/dev/_modules/dltk/core/metrics.html
'''



def dice_3d(predictions, labels, num_classes=1):
    dice_scores = np.zeros((num_classes))
    epsilon     = 1e-7
    for i in range(num_classes):    
        dice_num       = np.sum(predictions[labels==(i+1)])*2.0 
        dice_denom     = np.sum(predictions) + np.sum(labels)
        dice_scores[i] = (dice_num+epsilon)/(dice_denom+epsilon)
    return dice_scores.astype(np.float32)


def focal_loss_with_logits(y_pred, logits, targets, alpha=0.25, gamma=2):
    weight_a = alpha       * tf.cast((1 - y_pred), tf.float32) ** gamma * targets
    weight_b = (1 - alpha) * tf.cast(y_pred,       tf.float32) ** gamma * (1 - targets)

    return (tf.log1p(tf.exp(-tf.abs(logits))) + tf.nn.relu(-logits)) * (weight_a + weight_b) + logits * weight_b 


def labels_to_one_hot(ground_truth, num_classes=1):
    """
    Converts ground truth labels to one-hot, sparse tensors.
    Used extensively in segmentation losses.
    :param ground_truth: ground truth categorical labels (rank `N`)
    :param num_classes: A scalar defining the depth of the one hot dimension
        (see `depth` of `tf.one_hot`)
    :return: one-hot sparse tf tensor
        (rank `N+1`; new axis appended at the end)
    """
    # read input/output shapes
    if isinstance(num_classes, tf.Tensor):
        num_classes_tf = tf.to_int32(num_classes)
    else:
        num_classes_tf = tf.constant(num_classes, tf.int32)
    input_shape = tf.shape(ground_truth)
    output_shape = tf.concat(
        [input_shape, tf.reshape(num_classes_tf, (1,))], 0)

    if num_classes == 1:
        # need a sparse representation?
        return tf.reshape(ground_truth, output_shape)

    # squeeze the spatial shape
    ground_truth = tf.reshape(ground_truth, (-1,))
    # shape of squeezed output
    dense_shape = tf.stack([tf.shape(ground_truth)[0], num_classes_tf], 0)

    # create a rank-2 sparse tensor
    ground_truth = tf.to_int64(ground_truth)
    ids = tf.range(tf.to_int64(dense_shape[0]), dtype=tf.int64)
    ids = tf.stack([ids, ground_truth], axis=1)
    one_hot = tf.SparseTensor(
        indices=ids,
        values=tf.ones_like(ground_truth, dtype=tf.float32),
        dense_shape=tf.to_int64(dense_shape))

    # resume the spatial dims
    one_hot = tf.sparse_reshape(one_hot, output_shape)
    return one_hot


def cross_entropy(prediction, ground_truth, weight_map=None):
    """
    Function to calculate the cross-entropy loss function
    :param prediction: the logits (before softmax)
    :param ground_truth: the segmentation ground truth
    :param weight_map:
    :return: the cross-entropy loss
    """
    if len(ground_truth.shape) == len(prediction.shape):
        ground_truth = ground_truth[..., -1]

    # TODO trace this back:
    ground_truth = tf.cast(ground_truth, tf.int32)

    entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=prediction, labels=ground_truth)

    if weight_map is None:
        return tf.reduce_mean(entropy)

    weight_sum = tf.maximum(tf.reduce_sum(weight_map), 1e-6)
    return tf.reduce_sum(entropy * weight_map / weight_sum)


def dense_cross_entropy(prediction, ground_truth, weight_map=None):
    if weight_map is not None:
        raise NotImplementedError
    entropy = tf.nn.softmax_cross_entropy_with_logits(
        logits=prediction, labels=ground_truth)
    return tf.reduce_mean(entropy)


def generalised_dice_loss(prediction,
                          ground_truth,
                          weight_map=None,
                          type_weight='Square'):
    """
    Function to calculate the Generalised Dice Loss defined in
        Sudre, C. et. al. (2017) Generalised Dice overlap as a deep learning
        loss function for highly unbalanced segmentations. DLMIA 2017
    :param prediction: the logits
    :param ground_truth: the segmentation ground truth
    :param weight_map:
    :param type_weight: type of weighting allowed between labels (choice
        between Square (square of inverse of volume),
        Simple (inverse of volume) and Uniform (no weighting))
    :return: the loss
    """
    prediction = tf.cast(prediction, tf.float32)
    if len(ground_truth.shape) == len(prediction.shape):
        ground_truth = ground_truth[..., -1]
    one_hot = labels_to_one_hot(ground_truth, tf.shape(prediction)[-1])

    if weight_map is not None:
        num_classes = prediction.shape[1].value
        weight_map_nclasses = tf.tile(
            tf.expand_dims(tf.reshape(weight_map, [-1]), 1), [1, num_classes])
        ref_vol = tf.sparse_reduce_sum(
            weight_map_nclasses * one_hot, reduction_axes=[0])

        intersect = tf.sparse_reduce_sum(
            weight_map_nclasses * one_hot * prediction, reduction_axes=[0])
        seg_vol = tf.reduce_sum(
            tf.multiply(weight_map_nclasses, prediction), 0)
    else:
        ref_vol = tf.sparse_reduce_sum(one_hot, reduction_axes=[0])
        intersect = tf.sparse_reduce_sum(one_hot * prediction,
                                         reduction_axes=[0])
        seg_vol = tf.reduce_sum(prediction, 0)
    if type_weight == 'Square':
        weights = tf.reciprocal(tf.square(ref_vol))
    elif type_weight == 'Simple':
        weights = tf.reciprocal(ref_vol)
    elif type_weight == 'Uniform':
        weights = tf.ones_like(ref_vol)
    else:
        raise ValueError("The variable type_weight \"{}\""
                         "is not defined.".format(type_weight))
    new_weights = tf.where(tf.is_inf(weights), tf.zeros_like(weights), weights)
    weights = tf.where(tf.is_inf(weights), tf.ones_like(weights) *
                       tf.reduce_max(new_weights), weights)
    generalised_dice_numerator = \
        2 * tf.reduce_sum(tf.multiply(weights, intersect))
    generalised_dice_denominator = tf.reduce_sum(
        tf.multiply(weights, tf.maximum(seg_vol + ref_vol, 1)))
    generalised_dice_score = \
        generalised_dice_numerator / generalised_dice_denominator
    generalised_dice_score = tf.where(tf.is_nan(generalised_dice_score), 1.0,
                                      generalised_dice_score)
    return 1 - generalised_dice_score


def sparse_dice_plus_xent_loss(prediction, ground_truth, weight_map=None):
    """
    Function to calculate the loss used in https://arxiv.org/pdf/1809.10486.pdf,
    no-new net, Isenseee et al (used to win the Medical Imaging Decathlon).
    It is the sum of the cross-entropy and the Dice-loss.
    :param prediction: the logits
    :param ground_truth: the segmentation ground truth
    :param weight_map:
    :return: the loss (cross_entropy + Dice)
    """
    num_classes = tf.shape(prediction)[-1]

    prediction = tf.cast(prediction, tf.float32)
    loss_xent = cross_entropy(prediction, ground_truth, weight_map=weight_map)

    # Dice as according to the paper:
    one_hot = labels_to_one_hot(ground_truth, num_classes=num_classes)
    softmax_of_logits = tf.nn.softmax(prediction)

    if weight_map is not None:
        weight_map_nclasses = tf.tile(
            tf.reshape(weight_map, [-1, 1]), [1, num_classes])
        dice_numerator = 2.0 * tf.sparse_reduce_sum(
            weight_map_nclasses * one_hot * softmax_of_logits,
            reduction_axes=[0])
        dice_denominator = \
            tf.reduce_sum(weight_map_nclasses * softmax_of_logits,
                          reduction_indices=[0]) + \
            tf.sparse_reduce_sum(one_hot * weight_map_nclasses,
                                 reduction_axes=[0])
    else:
        dice_numerator = 2.0 * tf.sparse_reduce_sum(
            one_hot * softmax_of_logits, reduction_axes=[0])
        dice_denominator = \
            tf.reduce_sum(softmax_of_logits, reduction_indices=[0]) + \
            tf.sparse_reduce_sum(one_hot, reduction_axes=[0])

    epsilon = 0.00001
    loss_dice = -(dice_numerator + epsilon) / (dice_denominator + epsilon)
    dice_numerator = tf.Print(
        dice_denominator, [dice_numerator, dice_denominator, loss_dice])

    return loss_dice + loss_xent


def sparse_dice(prediction, ground_truth, weight_map=None):
    """
    Function to calculate the dice loss with the definition given in
        Milletari, F., Navab, N., & Ahmadi, S. A. (2016)
        V-net: Fully convolutional neural
        networks for volumetric medical image segmentation. 3DV 2016
    using a square in the denominator
    :param prediction: the logits
    :param ground_truth: the segmentation ground_truth
    :param weight_map:
    :return: the loss
    """
    prediction = tf.cast(prediction, tf.float32)
    if len(ground_truth.shape) == len(prediction.shape):
        ground_truth = ground_truth[..., -1]
    one_hot = labels_to_one_hot(ground_truth, tf.shape(prediction)[-1])

    if weight_map is not None:
        num_classes = prediction.shape[1].value
        weight_map_nclasses = tf.tile(tf.expand_dims(
            tf.reshape(weight_map, [-1]), 1), [1, num_classes])
        dice_numerator = 2.0 * tf.sparse_reduce_sum(
            weight_map_nclasses * one_hot * prediction, reduction_axes=[0])
        dice_denominator = \
            tf.reduce_sum(weight_map_nclasses * tf.square(prediction),
                          reduction_indices=[0]) + \
            tf.sparse_reduce_sum(one_hot * weight_map_nclasses,
                                 reduction_axes=[0])
    else:
        dice_numerator = 2.0 * tf.sparse_reduce_sum(
            one_hot * prediction, reduction_axes=[0])
        dice_denominator = \
            tf.reduce_sum(tf.square(prediction), reduction_indices=[0]) + \
            tf.sparse_reduce_sum(one_hot, reduction_axes=[0])
    epsilon = 0.00001

    dice_score = (dice_numerator + epsilon) / (dice_denominator + epsilon)
    return 1.0 - tf.reduce_mean(dice_score)

def dense_dice(logits, ground_truth, weight_map=None):
    """
    Computing mean-class Dice similarity.
    :param logits: last dimension should have ``num_classes``
    :param ground_truth: segmentation ground truth (encoded as a binary matrix)
        last dimension should be ``num_classes``
    :param weight_map:
    :return: ``1.0 - mean(Dice similarity per class)``
    """
    if weight_map is not None:
        raise NotImplementedError

    prediction     = tf.cast(tf.nn.softmax(logits), dtype=tf.float32)
    ground_truth   = tf.cast(ground_truth,              dtype=tf.float32)
    reduce_axes    = list(range(1,len(prediction.shape)-1))
    dice_numerator = 2.0 * tf.reduce_sum(
        prediction * ground_truth, axis=reduce_axes)
    dice_denominator = \
        tf.reduce_sum(tf.square(prediction),   axis=reduce_axes) + \
        tf.reduce_sum(tf.square(ground_truth), axis=reduce_axes)

    epsilon = 0.00001

    dice_score = (dice_numerator + epsilon) / (dice_denominator + epsilon)
    return 1.0 - tf.reduce_mean(dice_score)