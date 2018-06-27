import numpy as np
from idnns.forgetting.utils.utils_python import to_contiguous_array
import sys


def extract_probs(label, input_x):
    """calculate the probabilities of the given data and labels p(x), p(y) and (y|x)"""
    """For binary data we can count the number of black and white pixels. Then we calculate 
        the probability against the index of the output.
    """

    contiguous_input = to_contiguous_array(input_x)
    pxs, unique_array, unique_counts = _calc_prob_unique(contiguous_input)

    p_y_given_x = []
    for indexes in range(0, len(unique_array)):
        py_x_current = np.mean(label[indexes, :], axis=0)
        p_y_given_x.append(py_x_current)

    p_y_given_x = np.array(p_y_given_x).T

    contiguous_label = to_contiguous_array(label)
    pys, unique_array, unique_counts = _calc_prob_unique(contiguous_label)

    return {'pys': pys, 'pxs': pxs, 'p_y_given_x': p_y_given_x}


def extract_probs_flat(label, input_x):
    """calculate the probabilities of the given data and labels p(x), p(y) and (y|x)"""

    pxs = _calc_prob_flat(input_x)
    pys = _calc_prob_flat(label)

    """P(B|A) = P(A and B) / P(A)"""
    p_y_given_x = _calc_prob_conditional(input_x, label)

    return {'pys': pys, 'pxs': pxs, 'p_y_given_x': p_y_given_x}


def _calc_prob_conditional(A, B):
    pAs = _calc_prob_flat(A)
    pBs = _calc_prob_flat(B)

    return np.divide(np.multiply(pAs, pBs), pAs)


def _calc_prob_unique(contiguous_array):
    """calculates probabilities from a contiguous array by extracting unique items and their counts -> calculate how often an alements accours in the array"""
    """Careful, this method looses information about the shape and location of the array and the elements"""
    unique_array, unique_counts = np.unique(contiguous_array, return_counts=True)
    pxs = unique_counts / float(np.sum(unique_counts))
    return pxs, unique_array, unique_counts


def _calc_prob_flat(array):
    return np.divide(array, np.sum(array))
