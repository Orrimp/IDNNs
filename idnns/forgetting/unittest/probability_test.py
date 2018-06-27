import unittest
from idnns.forgetting.information.probability import extract_probs, extract_probs_flat
import numpy as np
import matplotlib.pyplot as plt


class ProbabilityTest(unittest.TestCase):

    def test(self):
        mean = [7, 1]
        cov = [[5, 1], [9, 3]]
        x, y = np.random.multivariate_normal(mean, cov, (784, 6500)).T
        z, d = np.random.multivariate_normal(mean, cov, (10, 6500)).T
        print("x-Shape: " + str(x.shape))
        print("y-Shape: " + str(y.shape))
        print("z-Shape: " + str(z.shape))

        params = extract_probs(z, x)
        print("p_y_given_x: " + str(params))
        print("pys: " + str(params['pys'].shape))
        print("pxs: " + str(params['pxs'].shape))
        print("p_y_given_x: " + str(params["p_y_given_x"].shape))

        print("itemsize: " + str(x.dtype.itemsize))
        print("shape: " + str(x.shape[1]))

        plt.plot(params['pys'], params['p_y_given_x'], 'x')
        plt.axis('equal')
        plt.show()

        print(params['p_y_given_x'].shape)
        self.assertNotEquals(params['pys'].all, params['p_y_given_x'].all)

        # x = np.array([[0.1, 0.2], [0.1, 0.4]], dtype=np.float32)
        # x1 = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
        # x2 = np.array([[0.5, 0.5], [0.3, 0.7]], dtype=np.float32)
        #
        # print("x:")
        # print(x)
        # print(sys.getsizeof(x[0]))
        #
        # y = x.view(np.dtype(np.uint8))
        # print("y")
        # print(y)
        #
        # # z = x.view(np.dtype((np.uint8, x.dtype.itemsize * x.shape[1])))
        # # print("z")
        # # print(z)
        #
        # # print(np.array_equal(y, z))
        # print('')
        # print(extract_probs(x1, x2))

        print('Single', end='\n')
        x_single = np.array([1, 4, 9, 25, 4])
        y_single = np.array([6, 10, 7, 7, 7])
        params = extract_probs_flat(y_single, x_single)
        print("pys: " + str(params['pys']))
        print("pxs: " + str(params['pxs']))
        print("p_y_given_x: " + str(params['p_y_given_x']))

        test_array = np.array([[10, 10], [30, 10], [45, 30]])
        unique_array, unique_counts = np.unique(test_array, return_counts=True)
        for indexes in range(0, len(unique_array)):
            py_x_current = np.mean(test_array[indexes, :], axis=0)
            k = test[indexes, :]


if __name__ == "__main__":
    prob_test = ProbabilityTest()
    prob_test.test()

