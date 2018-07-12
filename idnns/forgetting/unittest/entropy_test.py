from cmath import e

import numpy as np
import timeit
import unittest

from idnns.forgetting.information.entropy import stats_entropy
from idnns.information.entropy_estimators import entropy


class EntropyTest(unittest.TestCase):

    def test(self):
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



if __name__ == "__main__":
    pass