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
    print(inputArray)
    length = len(inputArray)
    print("Length:", length)
    print("Batches:", batches)

    outputArray = np.asarray(np.array_split(inputArray, batches))
    for i in range(len(outputArray)):
        outputArray[i] = np.asarray(outputArray[i])
        for j in range(len(outputArray[i])):
            outputArray[i][j] = np.asarray(outputArray[i][j])

    print(outputArray)
    print(outputArray.shape)
    print(outputArray[0].shape)

    sys.exit()
    return inputArray