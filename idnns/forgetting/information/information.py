import numpy as np
import scipy as sp
import scipy.special
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

from idnns.forgetting.information.probability import extract_probs


class Information:

    def __init__(self, num_max_epochs, num_of_samples, interval_information_display=499):
        self.interval_information_display = interval_information_display

        # The indexs of epochs that we want to calculate the information evenly distributed in logspace
        self.epochs_indexes = np.unique(np.logspace(start=1, stop=num_max_epochs, num=num_of_samples, dtype=int)) - 1

    def get_information(self, ws, input_x, label):
        """Calculate the information for the network for all the epochs and all the layers"""
        print('Start calculating the information...')

        return np.array([self.calc_information_for_epoch(index, ws[index], input_x, label) for index in range(len(ws))])

    def calc_information_for_epoch(self, iter_index, ws_iter_index, input_x, label):
        """Calculate the information for all the layers for specific epoch"""

        np.random.seed(None)
        # Generated a linear vector from -1 to 1 with 30 steps
        bins = np.linspace(start=-1, stop=1, num=30)
        label = np.array(label).astype(np.float)
        params = extract_probs(label, input_x)
        params = np.array([self.calc_information_sampling(data=ws_iter_index[i], bins=bins, label=label, parapms=params)
                           for i in range(len(ws_iter_index))])

        if np.mod(iter_index, self.interval_information_display) == 0:
            print('Calculated The information of epoch number - {0}'.format(iter_index))
        return params

    def calc_by_sampling_neurons(self, ws_iter_index, num_of_samples, label, sigma, bins, pxs):
        iter_information = []
        for j in range(len(ws_iter_index)):
            weights = ws_iter_index[j]
            new_data = np.zeros((num_of_samples * weights.shape[0], weights.shape[1]))
            labels = np.zeros((num_of_samples * label.shape[0], label.shape[1]))
            input = np.zeros((num_of_samples * weights.shape[0], 2))

            for weight_index in range(weights.shape[0]):
                cov_matrix = np.eye(weights[weight_index, :].shape[0]) * sigma
                random_samples = np.random.multivariate_normal(weights[weight_index, :], cov_matrix, num_of_samples)

                new_data[num_of_samples * weight_index:(num_of_samples * (weight_index + 1)), :] = random_samples
                labels[num_of_samples * weight_index:(num_of_samples * (weight_index + 1)), :] = label[weight_index, :]
                input[num_of_samples * weight_index:(num_of_samples * (weight_index + 1)), 0] = weight_index

            b = np.ascontiguousarray(input).view(np.dtype((np.void, input.dtype.itemsize * input.shape[1])))
            unique_array, unique_indices, unique_inverse_x, unique_counts = np.unique(b, return_index=True, return_inverse=True, return_counts=True)

            b_y = np.ascontiguousarray(labels).view(np.dtype((np.void, labels.dtype.itemsize * labels.shape[1])))
            unique_array_y, unique_indices_y, unique_inverse_y, unique_counts_y = np.unique(b_y, return_index=True, return_inverse=True, return_counts=True)
            pys1 = unique_counts_y / float(np.sum(unique_counts_y))

            iter_information.append(self.calc_information_for_layer(data=new_data, bins=bins, unique_inverse_x=unique_inverse_x, unique_inverse_y=unique_inverse_y, pxs=pxs, pys1=pys1))
            params = np.array(iter_information)
            return params

    def calc_information_for_layer(self, data, bins, unique_inverse_x, unique_inverse_y, pxs, pys):
        """understanding:"""
        bins = bins.astype(np.float32)
        digitized = bins[np.digitize(np.squeeze(data.reshape(1, -1)), bins) - 1].reshape(len(data), -1)
        b2 = np.ascontiguousarray(digitized).view(np.dtype((np.void, digitized.dtype.itemsize * digitized.shape[1])))
        unique_array, unique_inverse_t, unique_counts = np.unique(b2, return_index=False, return_inverse=True, return_counts=True)
        p_ts = unique_counts / float(sum(unique_counts))
        PXs, PYs = np.asarray(pxs).T, np.asarray(pys).T

        local_IXT, local_ITY = self.calc_information_from_mat(PXs, PYs, p_ts, digitized, unique_inverse_x, unique_inverse_y)
        return local_IXT, local_ITY

    def calc_information_from_mat(self, px, py, ps2, data, unique_inverse_x, unique_inverse_y):
        """Calculate the MI based on binning of the data"""
        H2 = -np.sum(ps2 * np.log2(ps2))
        H2X = self.calc_condtion_entropy(px, data, unique_inverse_x)
        H2Y = self.calc_condtion_entropy(py.T, data, unique_inverse_y)
        IY = H2 - H2Y
        IX = H2 - H2X
        return IX, IY

    def calc_condtion_entropy(self, px, t_data, unique_inverse_x):
        # Condition entropy of t given x
        H2X_array = np.array(Parallel(n_jobs=8)(delayed(self.calc_entropy_for_specipic_t)(t_data[unique_inverse_x == i, :], px[i]) for i in range(px.shape[0])))
        H2X = np.sum(H2X_array)
        return H2X

    def calc_entropy_for_specipic_t(self, current_ts, px_i):
        """Calc entropy for specipic t"""
        b2 = np.ascontiguousarray(current_ts).view(np.dtype((np.void, current_ts.dtype.itemsize * current_ts.shape[1])))
        unique_array, unique_inverse_t, unique_counts = np.unique(b2, return_index=False, return_inverse=True, return_counts=True)
        p_current_ts = unique_counts / float(sum(unique_counts))
        p_current_ts = np.asarray(p_current_ts, dtype=np.float64).T
        H2X = px_i * (-np.sum(p_current_ts * np.log2(p_current_ts)))
        return H2X

    def calc_information_sampling(self, data, bins, label, params):
        """calcinformation"""
        bins = bins.astype(np.float32)
        digitized = bins[np.digitize(np.squeeze(data.reshape(1, -1)), bins) - 1].reshape(len(data), -1)
        b2 = np.ascontiguousarray(digitized).view(np.dtype((np.void, digitized.dtype.itemsize * digitized.shape[1])))
        unique_array, unique_inverse_index, unique_counts = np.unique(b2, return_index=False, return_inverse=True, return_counts=True)
        p_ts = unique_counts / float(sum(unique_counts))
        PXs, PYs = np.asarray(params['pxs']).T, np.asarray(params['pys1']).T

        local_IXT, local_ITY = self.calc_information_from_mat(PXs, PYs, p_ts, digitized, params['unique_inverse_x'], params['unique_inverse_y'], unique_array)

        params = {'local_IXT': local_IXT, 'local_ITY': local_ITY}
        return params


if __name__ == "__main__":
    mu, sigma = 0, 1
    array1 = np.random.normal(1, np.sqrt(1), 100)
    array2 = np.random.uniform(0, 1, 100)

    hist_values_1, bins_endges_1 = np.histogram(array1, bins=25)
    hist_values_2, bins_endges_2 = np.histogram(array2, bins=25)
    print(array1)
    print("------------------")
    print(array2)

    count, bins, ignored = plt.hist(array2, 30, normed=True)
    plt.plot(bins, 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (bins - mu) ** 2 / (2 * sigma ** 2)), linewidth=2, color='r')
    plt.ylabel("Probability")
    plt.show()

    print("Entropy")
    result = sp.special.kl_div(array1, array2)

    plt.plot(result)
    plt.show()

    # Test Lullback-Leibler Divergence by creating two different distributes and sees the results of DKL
    # How will X comapre to Y with DKL

