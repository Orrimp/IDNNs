import tensorflow as tf

def init_loss(logits, func, batch_size):
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=func([batch_size, 1, 1, 1])))


def placeholder_var(size, name):
    return tf.placeholder(tf.float32, shape=size, name=name)


def create_layer(layer_input, filters, is_training, ksize=3, padding='SAME'):
    conv1 = tf.layers.conv2d(layer_input, filters, [ksize, ksize], strides=(1, 1), padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer())
    batch1 = tf.layers.batch_normalization(conv1, scale=True, training=is_training)  # momentum=0.95,
    lrelu1 = tf.nn.leaky_relu(batch1, 0.2)

    return lrelu1


def create_layer_max_pooling(input, filters, is_training,  ksize=3, padding='SAME'):
    conv1 = tf.layers.conv2d(input, filters, [ksize, ksize], strides=(1, 1), padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer())
    batch1 = tf.layers.batch_normalization(conv1, scale=True, training=is_training)  # momentum=0.95,
    lrelu1 = tf.nn.leaky_relu(batch1, 0.2)
    max_pool1 = tf.nn.max_pool(lrelu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding=padding)

    return max_pool1


def create_max_pool(layer_input, ksize2x2, stride2x2):
    return tf.nn.max_pool(layer_input, ksize=[1, ksize2x2[0], ksize2x2[1], 1], strides=[1, stride2x2[0], stride2x2[1], 1], padding="VALID")

