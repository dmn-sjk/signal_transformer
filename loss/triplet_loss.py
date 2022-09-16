import numpy as np
import tensorflow as tf

def pairwise_distances(vector):
    return tf.sqrt(tf.reduce_sum((tf.expand_dims(vector, 1) - tf.expand_dims(vector, 0)) ** 2, 2))

def batch_all_triplet_loss(positions, embeddings, margin=0.1, dist_threshold=0.25):
    pairwise_dist_emb = pairwise_distances(embeddings)
    pairwise_dist_pos = pairwise_distances(positions)

    anchor_positive_dist = tf.expand_dims(pairwise_dist_emb, 2)
    anchor_positive_dist_pos = tf.expand_dims(pairwise_dist_pos, 2)

    anchor_negative_dist = tf.expand_dims(pairwise_dist_emb, 1)
    anchor_negative_dist_pos = tf.expand_dims(pairwise_dist_pos, 1)

    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

    positive_shape_multpl = tf.constant([1, 1, tf.shape(anchor_negative_dist_pos)[0].numpy()])
    negative_shape_multpl = tf.constant([1, tf.shape(anchor_negative_dist_pos)[0].numpy(), 1])

    anchor_positive_dist_pos_broadcast = tf.tile(anchor_positive_dist_pos, positive_shape_multpl)
    anchor_negative_dist_pos_broadcast = tf.tile(anchor_negative_dist_pos, negative_shape_multpl)

    positives_mask = anchor_positive_dist_pos_broadcast <= dist_threshold
    negatives_mask = anchor_negative_dist_pos_broadcast > dist_threshold

    indices_equal = tf.cast(tf.eye(tf.shape(positions)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)
    i_not_equal_j = tf.expand_dims(indices_not_equal, 2)
    i_not_equal_k = tf.expand_dims(indices_not_equal, 1)
    j_not_equal_k = tf.expand_dims(indices_not_equal, 0)

    distinct_indices = tf.logical_and(tf.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)

    mask = tf.logical_and(distinct_indices, positives_mask, negatives_mask)
    mask = tf.cast(mask, tf.float32)

    triplet_loss = tf.multiply(mask, triplet_loss)

    triplet_loss = tf.maximum(triplet_loss, 0.0)

    valid_triplets = tf.cast(tf.greater(triplet_loss, 1e-16), tf.float32)
    num_positive_triplets = tf.reduce_sum(valid_triplets)
    num_valid_triplets = tf.reduce_sum(mask)
    fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)

    triplet_loss = tf.reduce_sum(triplet_loss) / (num_positive_triplets + 1e-16)

    return triplet_loss, fraction_positive_triplets
