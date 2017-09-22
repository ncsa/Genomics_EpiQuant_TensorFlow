""" Batch Builder

Builds batches for training the neural network
"""

import numpy as np
import modules.Progress as prog

def make_batches(input_array, batches):
    """ Makes batches from a given set.

    Args:
        input_array: A numpy array containing the data to be split into batches.
        batches: The number of batches to create.

    Returns:
        output_array: A numpy array containing the data split into batches.
    """

    output_array = np.asarray(np.array_split(input_array, batches))

    for i in range(len(output_array)):
        output_array[i] = np.asarray(output_array[i])
        for j in range(len(output_array[i])):
            output_array[i][j] = np.asarray(output_array[i][j])
        prog.progress(i, len(output_array), "Batches Built")

    # print(output_array.shape)
    # print(output_array[0].shape)

    return output_array
