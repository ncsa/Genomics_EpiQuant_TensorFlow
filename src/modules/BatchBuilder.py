import numpy as np
import sys

def makeBatches(inputArray, batches):
    """ Makes batches from a given set.

    Args:
        inputArray: A numpy array containing the data to be split into batches.
        batches: The number of batches to create.

    Returns:
        outputArray: A numpy array containing the data split into batches.
    """
    length = len(inputArray)

    outputArray = np.asarray(np.array_split(inputArray, batches))
    for i in range(len(outputArray)):
        outputArray[i] = np.asarray(outputArray[i])
        for j in range(len(outputArray[i])):
            outputArray[i][j] = np.asarray(outputArray[i][j])
    return outputArray