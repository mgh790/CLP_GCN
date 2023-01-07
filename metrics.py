import tensorflow as tf
# import  tensorflow from tfa.metrics.F1Score
# import tensorflow_addons as tfa

from tensorflow.python.keras.metrics import Metric
def masked_softmax_cross_entropy(preds, labels, mask,weights=None,loss_unsupervised=None):
    # """Softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    if loss_unsupervised is not None:
        loss=tf.add(loss,tf.cast(loss_unsupervised,dtype=tf.float32))

    if weights is not None:
        loss *=weights
    # loss1=loss
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)

    ## leskovekkk new loss
    # loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=preds, labels=labels)
    # mask = tf.cast(mask, dtype=tf.float32)
    # loss = tf.reduce_sum(loss * mask) / tf.reduce_sum(mask)
    # return loss


## leskovekkk
# GCN loss
# self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.outputs, labels=self.labels)
# self.loss = tf.reduce_sum(self.loss * self.label_mask) / tf.reduce_sum(self.label_mask)
#
# # LPA loss
# lpa_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.predicted_label, labels=self.labels)
# lpa_loss = tf.reduce_sum(lpa_loss * self.label_mask) / tf.reduce_sum(self.label_mask)
# self.loss += self.args.lpa_weight * lpa_loss
#

def masked_mean_square(y_input,y, mask):
    # loss = ((y_input - y) ** 2).sum(axis=1)
    loss=tf.squared_difference(y_input,y)
    mask = tf.cast(mask, dtype=tf.float32)

    loss *= mask
    # return loss
    return tf.reduce_mean(loss)

def masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    return tf.reduce_mean(accuracy_all)
