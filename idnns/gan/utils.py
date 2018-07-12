import tensorflow as tf

def init_loss(logits, func, batch_size):
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=func([batch_size, 1, 1, 1])))

def placeholder_var(size, name):
    return tf.placeholder(tf.float32, size, name=name)
