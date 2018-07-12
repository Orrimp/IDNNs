from collections import Counter
from math import log, e
import numpy as np
from scipy import stats


def entropy(x, base=2):
    """Computes the entropy with from e.g. labels = [1, 3, 5, 2, 3]. Its even faster then scipy and numpy impl. """
    # calculate probability for each byte as number of occurrences / array length
    probabilities = [n_x/len(x) for _, n_x in Counter(x).items()]
    # calculate per-character entropy fractions
    e_x = [- p_x * log(p_x, base) for p_x in probabilities]
    # sum fractions to obtain Shannon entropy
    return sum(e_x)


# https://stackoverflow.com/questions/15450192/fastest-way-to-compute-entropy-in-python
def entropy_unique(x, base=e):
    value, counts = np.unique(x, return_counts=True)
    norm_counts = counts / counts.sum()
    return -(norm_counts * np.log(norm_counts)/np.log(base)).sum()


def stats_entropy(x, base=2):
    counter = Counter(x)
    return  stats.entropy([x for x in counter.values()], base=base)

def entropy_prob(x):
    # https: // recast.ai / blog / introduction - information - theory - care /