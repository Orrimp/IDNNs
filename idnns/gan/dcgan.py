import os, time, imageio, itertools, pickle, argparse
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

from argsparser import ArgsParser
from utils import init_loss, placeholder_var, create_layer_max_pooling, create_layer, create_max_pool
from visual import Visual
from store import Store

tf.logging.set_verbosity(tf.logging.INFO)

"""This class implements DCGAN https://github.com/znxlwm/tensorflow-MNIST-GAN-DCGAN"""
class DCGAN:

    def __init__(self, args):
        np.random.seed(int(time.time()))
        self.store = Store()
        self.visual = Visual(self.store)
        self.image_shape = [28, 28, 1]  # 28x28 pixels and black white
        self.batch_size = args.batch_size
        self.lr = args.learning_rate
        self.train_epoch = args.train_epoch
        self.dropout_keep_probability = tf.placeholder("float")

        self.mnist = input_data.read_data_sets("MNIST_data/", one_hot=True, reshape=[])
        self.is_training = tf.placeholder(dtype=tf.bool)

        self.x = tf.placeholder(tf.float32, shape=(None, 64, 64, 1), name="X_Input")
        self.z = tf.placeholder(tf.float32, shape=(None, 1, 1, 100), name="Z")

        self.G_z = define_generator(self.z, self.is_training)

        D_real, D_real_logits = define_discriminator(self.x, self.is_training)
        D_fake, D_fake_logits = define_discriminator(self.G_z, self.is_training, reuse=True)

        D_loss_real = init_loss(D_real_logits, tf.ones, self.batch_size)
        D_loss_fake = init_loss(D_fake_logits, tf.zeros, self.batch_size)
        self.G_loss = init_loss(D_fake_logits, tf.ones, self.batch_size)
        self.D_loss = D_loss_real + D_loss_fake

        self.sess = None

    def train(self):
        D_vars = [var for var in tf.trainable_variables() if var.name.startswith('discriminator')]
        G_vars = [var for var in tf.trainable_variables() if var.name.startswith('generator')]

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            D_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5).minimize(self.D_loss, var_list=D_vars)
            G_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5).minimize(self.G_loss, var_list=G_vars)

        # start the session
        self.sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()

        train_set = tf.image.resize_images(self.mnist.train.images, [64, 64]).eval()
        train_set = (train_set - 0.5) / 0.5  # normalization; range: -1 ~ 1

        print('training start!')
        start_time = time.time()

        for epoch in range(self.train_epoch):
            G_losses = []
            D_losses = []
            epoch_start_time = time.time()
            iterations = self.mnist.train.num_examples // self.batch_size
            for iter in range(iterations):
                # update discriminator
                x_ = train_set[iter * self.batch_size:(iter + 1) * self.batch_size]
                z_ = np.random.normal(0, 1, (self.batch_size, 1, 1, 100))
                loss_d, _ = self.sess.run([self.D_loss, D_optim], {self.x: x_, self.z: z_, self.is_training: True})

                # update generator
                z_ = np.random.normal(0, 1, (self.batch_size, 1, 1, 100))
                loss_g, _ = self.sess.run([self.G_loss, G_optim], {self.z: z_, self.x: x_, self.is_training: True})

                G_losses.append(loss_g)
                D_losses.append(loss_d)
                print("Epoch " + str(epoch) + "/" + str(self.train_epoch) + " of iteration " + str(iter)+ "/" + str(iterations) + " with loss_g " + str(loss_g) + " and loss_d " + str(loss_d))

            epoch_end_time = time.time()
            per_epoch_ptime = epoch_end_time - epoch_start_time
            print('[%d/%d] - ptime: %.2f loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), self.train_epoch, per_epoch_ptime, np.mean(D_losses), np.mean(G_losses)))
            fixed_p = root + 'Fixed_results/' + model + str(epoch + 1) + '.png'
            fixed_z_ = np.random.normal(0, 1, (25, 1, 1, 100))
            test_images = self.sess.run(self.G_z, {self.z: fixed_z_, self.is_training: False})
            self.visual.show_result(test_images, num_epoch=epoch, show=False, save=True, path=fixed_p)

            self.store.hist_append('D_losses', np.mean(D_losses))
            self.store.hist_append('G_losses', np.mean(G_losses))
            self.store.hist_append('per_epoch_ptimes', per_epoch_ptime)

        # let it run and save the images
        end_time = time.time()
        total_ptime = end_time - start_time
        self.store.hist_append('total_ptime', total_ptime)

        print('Avg per epoch ptime: %.2f, total %d epochs ptime: %.2f' % (np.mean(self.store.retrieve('per_epoch_ptimes')), self.train_epoch, total_ptime))
        self.visual.show_train_hist(self.store.retrieve('D_losses'), self.store.retrieve('G_losses'), show=False, save=True, path=root + model + 'train_hist.png')

        images = []
        for e in range(self.train_epoch):
            img_name = root + 'Fixed_results/' + model + str(e + 1) + '.png'
            images.append(imageio.imread(img_name))
        imageio.mimsave(root + model + 'generation_animation.gif', images, fps=5)


    def visualize(self, layer, image_stimulation):
        '''Gives the model a layer and a image to check the activates against
        :param layer: Layer to visualize
        :param image_stimulation: Image to run through the models to see the activations
        :return: graph to display
        '''
        #https://medium.com/@awjuliani/visualizing-neural-network-layer-activation-tensorflow-tutorial-d45f8bf7bbc4
        flatten_image = np.reshape(image_stimulation, [1, 784], order='F')
        units = self.sess.run(layer, feed_dict={self.x: flatten_image, self.dropout_keep_probability: 1.0})
        self.visual.plotNNFilter(units)


    def shutdown(self):
        self.sess.close()


def define_discriminator_simplenet(input, is_training=True, reuse=False):
    with tf.variable_scope('discriminator', reuse=reuse):
        conv1 = create_layer(input, 64, is_training)
        conv2 = create_layer(conv1, 128, is_training)
        conv3 = create_layer(conv2, 128, is_training)
        conv4 = create_layer(conv3, 128, is_training)
        conv5 = create_layer_max_pooling(conv4, 192, is_training)
        conv6 = create_layer(conv5, 192, is_training)
        conv7 = create_layer(conv6, 192, is_training)
        conv8 = create_layer(conv7, 192, is_training)
        conv9 = create_layer(conv8, 192, is_training)
        conv10 = create_layer_max_pooling(conv9, 192, is_training)
        conv11 = create_layer(conv10, 288, is_training, ksize=1)
        conv12 = create_layer(conv11, 355, is_training, ksize=1)
        conv13 = create_layer(conv12, 432, is_training)

        conv14 = create_layer(conv13, 1, is_training)
        conv15 = create_max_pool(conv14, [2, 2], [2, 2])
        output = tf.nn.sigmoid(conv15)

    return output, conv14


def define_discriminator(input, is_training=True, reuse=False):
    with tf.variable_scope('discriminator', reuse=reuse):
        # 1st hidden layer

        conv1 = tf.layers.conv2d(input, 128, [4, 4], strides=(2, 2), padding='same')
        lrelu1 = tf.nn.leaky_relu(conv1, 0.2)

        # 2nd hidden layer
        conv2 = tf.layers.conv2d(lrelu1, 256, [4, 4], strides=(2, 2), padding='same')
        batch2 = tf.layers.batch_normalization(conv2, training=is_training)
        lrelu2 = tf.nn.leaky_relu(batch2, 0.2)

        # 3rd hidden layer
        conv3 = tf.layers.conv2d(lrelu2, 512, [4, 4], strides=(2, 2), padding='same')
        batch3 = tf.layers.batch_normalization(conv3, training=is_training)
        lrelu3 = tf.nn.leaky_relu(batch3, 0.2)

        # 4th hidden layer
        conv4 = tf.layers.conv2d(lrelu3, 1024, [4, 4], strides=(2, 2), padding='same')
        batch4 = tf.layers.batch_normalization(conv4, training=is_training)
        lrelu4 = tf.nn.leaky_relu(batch4, 0.2)

        # output layer
        conv5 = tf.layers.conv2d(lrelu4, 1, [4, 4], strides=(1, 1), padding='valid')
        output = tf.nn.sigmoid(conv5)

        return output, conv5

def define_generator(input, is_training=True, reuse=False):
    with tf.variable_scope('generator', reuse=reuse):

        # 1st hidden layer
        conv1 = tf.layers.conv2d_transpose(input, 1024, [4, 4], strides=(1, 1), padding='valid')
        batch1 = tf.layers.batch_normalization(conv1, training=is_training)
        lrelu1 = tf.nn.leaky_relu(batch1, 0.2)

        # 2nd hidden layer
        conv2 = tf.layers.conv2d_transpose(lrelu1, 512, [4, 4], strides=(2, 2), padding='same')
        batch2 = tf.layers.batch_normalization(conv2, training=is_training)
        lrelu2 = tf.nn.leaky_relu(batch2, 0.2)

        # 3rd hidden layer
        conv3 = tf.layers.conv2d_transpose(lrelu2, 256, [4, 4], strides=(2, 2), padding='same')
        batch3 = tf.layers.batch_normalization(conv3, training=is_training)
        lrelu3 = tf.nn.leaky_relu(batch3, 0.2)

        # 4th hidden layer
        conv4 = tf.layers.conv2d_transpose(lrelu3, 128, [4, 4], strides=(2, 2), padding='same')
        batch4 = tf.layers.batch_normalization(conv4, training=is_training)
        lrelu4 = tf.nn.leaky_relu(batch4, 0.2)

        # output layer
        conv5 = tf.layers.conv2d_transpose(lrelu4, 1, [4, 4], strides=(2, 2), padding='same')
        output = tf.nn.tanh(conv5)

        return output


def init_directories():
    root = 'MNIST_DCGAN_results/'
    model = 'MNIST_DCGAN_'
    if not os.path.isdir(root):
        os.mkdir(root)
    if not os.path.isdir(root + 'Fixed_results'):
        os.mkdir(root + 'Fixed_results')

    return root, model


if __name__ == "__main__":
    args = ArgsParser().parse()
    root, model = init_directories()

    gan = DCGAN(args)
    gan.train()
