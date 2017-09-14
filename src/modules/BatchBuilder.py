import numpy as np
import sys
import modules.Progress as prog

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
    print(outputArray)
    sys.exit()

    print("Making Batches...")
    for i in range(len(outputArray)):
        prog.progress(i, len(outputArray), "Batches Built")
        outputArray[i] = np.asarray(outputArray[i])
        for j in range(len(outputArray[i])):
            outputArray[i][j] = np.asarray(outputArray[i][j])

    print(outputArray.shape)
    print(outputArray[0].shape)

    return outputArray