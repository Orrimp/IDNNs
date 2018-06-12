from idnns.forgetting.interfaces.I_Model import I_Model
from idnns.forgetting.storage import Storage
from idnns.forgetting.utils.constants import Const
from idnns.forgetting.utils.utils_network import create_layer, default_activation_func, placeholder_var
from idnns.forgetting.utils.utils_data import load_mnist_data, MNIST_TRAIN_SIZE
from idnns.forgetting.utils.config import parse, extract_dims
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)


class DnnModel(I_Model):

    def __init__(self, args, layers_params, data_set=load_mnist_data(), file_name="dnn_model", activation_func=None):
        """
        :param layers_params: The dimensions of the layers including input, hidden and output layers
        :param file_name: The name of the file to save the trained model
        """

        # Variables
        self.num_of_epochs = args.num_of_epochs
        self.layers_params = layers_params
        self.__save_file = file_name
        self.input = placeholder_var(layers_params[0], args.batch_size, 'inputs')
        self.label_placeholder = placeholder_var(layers_params[-1], args.batch_size, 'labbels')
        self.activation_func = activation_func if activation_func is not None else default_activation_func()

        self.data_set = data_set
        self.learning_rate = args.learning_rate
        self.batch_size = args.batch_size
        self.num_of_iterations = int(self.num_of_epochs * MNIST_TRAIN_SIZE / self.batch_size)

        # Other execution flow variables
        self.interval_to_print = args.interval_to_print
        self.sess = None
        self.last_layer = None
        self.store = Storage(self.num_of_iterations, args.interval_to_print)

    def create_network_layers(self):
        # Create layers but ignore the first and the lasst value inside the layer description because its the input and output dimensions.
        prev_hidden = self.input
        for layer_index in range(1, len(self.layers_params)-1):
            value, instruction = parse(self.layers_params[layer_index])

            if instruction is None:
                row_size, col_size = extract_dims(self.layers_params, layer_index)
                weights, biases, prev_hidden = create_layer('hidden_layer' + str(layer_index), self.activation_func['hidden'], prev_hidden, row_size, col_size)
                print("Creating layer with " + str(row_size) + " x " + str(col_size))

            if value is None and instruction == "dropout":
                prev_hidden = tf.layers.dropout(prev_hidden, training=True, name='hidden_layer' + str(layer_index))
                print("Creating layer dropout")

        row_size, col_size = self.layers_params[-2], self.layers_params[-1]
        weights, biases, self.last_layer = create_layer('output_layer', self.activation_func['output'], prev_hidden, row_size, col_size)

        return self

    def fill_feed_dict(self, traing_data=True):
        if traing_data:
            images, labels = self.data_set.train.next_batch(self.batch_size, fake_data=False, shuffle=True)
        else:
            images, labels = self.data_set.test.next_batch(self.batch_size, fake_data=False, shuffle=True)

        feed_dict = {self.input:images, self.label_placeholder:labels}
        return feed_dict

    def init_hyperparams(self):
        self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.label_placeholder * tf.log(self.last_layer), reduction_indices=[1]))
        self.optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.learning_rate).minimize(self.cross_entropy, global_step=self.store.global_step)

        return self

    def init_network(self):
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        file_writer = tf.summary.FileWriter(Const.LOG_LOC.value, self.sess.graph)

        return self

    def train_network(self):
        """Train the nework"""
        # Initialize the network

        #self.init_network()

        # Go over the epochs
        for j in range(0, self.num_of_iterations):
            summary, result = self.sess.run([self.optimizer, self.cross_entropy], feed_dict=self.fill_feed_dict(True))
            if (j % self.interval_to_print) == 0:
                    print("Current step %d with training loss %g" % (j, result))
                    self.extract_and_save_state()

        return self

    def evaluate_network(self, num_of_epochs=2000):
        loss = tf.losses.mean_squared_error(labels=self.label_placeholder, predictions=self.last_layer)

        # how to evaluate diff between input and output on mnist?
        accuracy = 0
        for j in range(0, num_of_epochs):
            accuracy += self.sess.run(loss, feed_dict=self.fill_feed_dict(False))

        print("The average accuracy is %g " % (accuracy/num_of_epochs))

        return self

    def extract_and_save_state(self):
        self.store.save(self.sess)
        self.store.print_state()

        return True

    def close(self):
        self.sess.close()

