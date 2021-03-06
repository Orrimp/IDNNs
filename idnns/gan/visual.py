import math
import os, time, itertools, imageio, pickle
import matplotlib.pyplot as plt
import matplotlib
from store import Store
import numpy as np

matplotlib.use('Agg')
plt.switch_backend('agg')


class Visual:

    def __init__(self, store: Store):
        self.store = store

    def show_result(self, test_images, num_epoch, show=False, save=False, path='result.png'):

        size_figure_grid = 5
        fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
        for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
            ax[i, j].get_xaxis().set_visible(False)
            ax[i, j].get_yaxis().set_visible(False)

        for k in range(size_figure_grid * size_figure_grid):
            i = k // size_figure_grid
            j = k % size_figure_grid
            ax[i, j].cla()
            ax[i, j].imshow(np.reshape(test_images[k], (64, 64)), cmap='gray')

        label = 'Epoch {0}'.format(num_epoch)
        fig.text(0.5, 0.04, label, ha='center')

        if save:
            plt.savefig(path)

        if show:
            plt.show()
        else:
            plt.close()

    def show_train_hist(self, hist_D_losses, hist_G_losses, show=False, save=False, path='Train_hist.png'):
        x = range(len(hist['D_losses']))

        y1 = hist_D_losses
        y2 = hist_G_losses

        plt.plot(x, y1, label='D_loss')
        plt.plot(x, y2, label='G_loss')

        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        plt.legend(loc=4)
        plt.grid(True)
        plt.tight_layout()

        if save:
            plt.savefig(path)

        if show:
            plt.show()
        else:
            plt.close()


    def plotNNFilter(self, units):
        filters = units.shape[3]
        plt.figure(1, figsize=(20, 20))
        n_columns = 6
        n_rows = math.ceil(filters / n_columns) + 1
        for i in range(filters):
            plt.subplot(n_rows, n_columns, i + 1)
            plt.title('Filter ' + str(i))
            plt.imshow(units[0, :, :, i], interpolation="nearest", cmap="gray")