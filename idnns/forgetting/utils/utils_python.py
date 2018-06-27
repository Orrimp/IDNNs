import  numpy as np


def to_contiguous_array(array):
    """put the input array as single row contiguous array with size (itemsize * shape[1]) for faster access"""
    contiguous_input = np.ascontiguousarray(array).view(np.dtype((np.void, array.dtype.itemsize * array.shape[1])))
    return contiguous_input
