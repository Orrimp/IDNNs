import tensorflow as tf
import datetime


class Storage:

    def __init__(self, num_of_iterations, interval_to_save):
        self._activations = []
        self._weights = []
        self._biases = []
        self._saver: tf.train.Saver = None
        self._writer: tf.summary.FileWriter = None
        self._num_of_iterations = num_of_iterations
        self._interval_to_save = interval_to_save
        self._file_prefix = "saves/" + str(datetime.datetime.now()) + "/model.ckpt"
        self._global_step = tf.Variable(100, dtype=tf.int32, trainable=False, name='global_step')


    def print_state(self):
        print("Currently stored weights: " + str(len(self._weights)))
        # print("Currently stored biases: " + str(len(self.biases)))
        # print("Currently stored activations: " + str(len(self.activations)))

    def save(self, sess):
        if self._saver is None:
            self._saver = tf.train.Saver(max_to_keep=int(self._num_of_iterations / self._interval_to_save) + 1)
            self._writer = tf.summary.FileWriter('./improved_graph', sess.graph)

        self._saver.save(sess=sess, save_path=self._file_prefix, global_step=self._global_step)

        self._weights.append(tf.get_collection(tf.GraphKeys.WEIGHTS))
        self._biases.append(tf.get_collection(tf.GraphKeys.BIASES))
        self._activations.append(tf.get_collection(tf.GraphKeys.ACTIVATIONS))
        print("stored _weights: " + str(self._weights))
        print("stored _biases: " + str(self._biases))
        print("stored _activations: " + str(self._activations))



    @property
    def global_step(self):
        return self._global_step

