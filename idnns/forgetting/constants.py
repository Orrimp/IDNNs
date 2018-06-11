from enum import Enum
import tensorflow as tf


class Const(Enum):
    WEIGHTS = "weights"
    BIASES = "biases"


if __name__ == '__main__':
    print(Const.WEIGHTS)
    print(Const.BIASES)

