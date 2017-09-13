import numpy as np
import sys

def makeBatches(inputArray, size):
    """ Makes batches from a given set.

    Args:
        inputArray: An array containing the data to be split into batches.
        size: Size of the batches that should be made.

    Returns:

    """
    print(inputArray)
    length = len(inputArray)
    print(length, size)
    batches = length // size + 1
    print(batches)
    outputArray = np.array_split(inputArray, batches)
    for i in range(len(outputArray)):
        outputArray[i] = np.asarray(outputArray[i])
    print(outputArray)
    print(outputArray.shape)

    sys.exit()
    return inputArray