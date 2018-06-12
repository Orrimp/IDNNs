import timeit
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


if __name__ == "__main__":

    repeat_number = 1000000
    a = timeit.repeat(stmt='''entropy_unique(labels)''',
                      setup='''labels=[1,3,5,2,3,5,3,2,1,3,4,5];from __main__ import entropy_unique''',
                      repeat=3, number=repeat_number)

    c = timeit.repeat(stmt='''entropy(labels)''',
                      setup='''labels=[1,3,5,2,3,5,3,2,1,3,4,5];from __main__ import entropy_shannon''',
                      repeat=3, number=repeat_number)

    d = timeit.repeat(stmt='''stats_entropy(labels)''',
                      setup='''labels=[1,3,5,2,3,5,3,2,1,3,4,5];from __main__ import stats_entropy''',
                      repeat=3, number=repeat_number)

    for approach, timeit_results in zip(['numpy', 'entropy_shannon', 'stats_entropy'], [a, c, d]):
        print('Method: {}, Avg.: {:.6f}'.format(approach, np.array(timeit_results).mean()))

    labels = [1, 3, 5, 2, 3, 5, 3, 2, 1, 3, 4, 5]

    print('entropy_unique base 2: \t\t\t' + str(entropy(labels, 2)))
    print('entropy base 2: \t' + str(entropy(labels, 2)))
    print('stats_entropy base 2: \t\t' + str(stats_entropy(labels, 2)))

    print("")
    print('entropy_unique base e: \t\t\t' + str(entropy(labels, e)))
    print('entropy base e: \t' + str(entropy(labels, e)))
    print('stats_entropy base e: \t\t' + str(stats_entropy(labels, e)))


