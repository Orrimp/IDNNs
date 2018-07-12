import tensorflow as tf
import numpy as np

initializers = {'rand_norm': tf.random_normal_initializer,
                   'rand_uniform': tf.random_uniform_initializer,
                   'zero': tf.zeros_initializer,
                   'zero_float': tf.constant_initializer(0.0),
                   'truncated': tf.truncated_normal_initializer,
                   'xavier_normal': tf.glorot_normal_initializer,
                   'xavier_uniform': tf.glorot_uniform_initializer}


def create_layer(name_scope, activation_function, prev_layer, row_size, col_size):
    # Bulid layer of the network with weights and biases
    # https://www.tensorflow.org/performance/xla/shapes -> dimensions shape definition order (y, x) or (z, y, x)
    weights = get_scope_variable(name_scope=name_scope, type=tf.GraphKeys.WEIGHTS, shape=[row_size, col_size] ,initializer=intializer_func('truncated')(mean=0.0, stddev=1.0 / np.sqrt(float(row_size))))
    biases = get_scope_variable(name_scope=name_scope, type=tf.GraphKeys.BIASES, shape=[col_size], initializer=intializer_func('zero_float'))

    with tf.variable_scope(name_scope) as scope:
        layer_input = tf.matmul(prev_layer, weights) + biases
        new_layer = activation_function(layer_input, name='output')

    print("Creating " + str(name_scope) + " with ", end='')
    print(' prev_layer ' + str(prev_layer.shape) , end='')
    print(' row_size ' + str(row_size), end='')
    print(' col_size ' + str(col_size), end='')
    print(' weights ' + str(weights.shape), end='')
    print(' biases' + str(biases.shape), end='')
    print(' new_layer ' + str(new_layer.shape), end='')

    return weights, biases, new_layer


def default_activation_func():
    # Add initializer
    return {'input': tf.nn.relu, 'hidden': tf.nn.relu, 'output': tf.nn.sigmoid}


def intializer_func(name):
    return initializers[name]


def get_scope_variable(name_scope, type, shape, initializer=None):
    """Create or reuse variables and add some additional information to it"""
    """By default tensorflow places all the variable into GLOBAL_VARIABLES and TRAINABLE_VARIABLES, we override it -> fix it by adding GLOBAL and TRAINABLE or the intializer will not work"""

    with tf.variable_scope(name_scope, reuse=tf.AUTO_REUSE) as scope:
        try:
            print("Initializing " + name_scope + " " + type)
            v = tf.get_variable(name=type, collections=[type, tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.TRAINABLE_VARIABLES], shape=shape, initializer=initializer)
        except ValueError:
            print("Error initializing " + name_scope + " " + type)
            scope.reuse_variables()
            v = tf.get_variable(name=type, collections=[type, tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.TRAINABLE_VARIABLES], shape=shape, initializer=initializer)

    # TODO change to: Add varaibles to the collections later with "tf.add_to_collection(name varaible)
    return variable_summaries(v)


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

    return var


def placeholder_var(size, batch_size, name):
    return tf.placeholder(tf.float32, [batch_size, size], name=name)

def placeholder_var(size, name):
    return tf.placeholder(tf.float32, size, name=name)
