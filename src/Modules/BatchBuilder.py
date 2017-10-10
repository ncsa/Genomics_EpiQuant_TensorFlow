""" Batch Builder

Builds batches for training the neural network
"""

import numpy as np
import Modules.Progress as prog

def make_batches(input_array, batches):
    """ Makes batches from a given set.

    Args:
        input_array: A numpy array containing the data to be split into batches.
        batches: The number of batches to create.

    Returns:
        output_array: A numpy array containing the data split into batches.
    """

    output_array = np.asarray(np.array_split(input_array, batches))
    len_out_array = len(output_array)

    for i in range(len_out_array):
        output_array[i] = np.asarray(output_array[i])
        for j in range(len(output_array[i])):
            output_array[i][j] = np.asarray(output_array[i][j])
        prog.progress(i, len_out_array, "Batches Built")

    return output_array
