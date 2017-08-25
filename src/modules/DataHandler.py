import numpy as np
import _thread
import math

def readData(filePath):
    with open(filePath) as file:
        # Reads in data from a file.
        # Strips white space and newlines from the ends.
        # Splits by newlines then tabs for each split.
        # Slice off name column and delete name column from data array.
        # Convert data array to floating point.            
        fullArray = np.asarray([x.split("\t") for x in file.read().strip().split("\n")])
        nameArray = np.asarray([[x] for x in fullArray[:, 0]])
        dataArray = np.delete(fullArray, 0, 1).astype(np.float)
        return nameArray, dataArray

def nameGeneration(name1, name2, position, outputArray):
    outputArray[position] = ["%s%s%s" % (name1, "::", name2)]
    print(position)

def calculateInteractions(inputArray):
    length = len(inputArray)
    combinations = length + (math.factorial(length) // 2 // math.factorial(length - 2))
    iteration = length
    print(combinations)
    if isinstance(inputArray[0][0], str):
        outputArray = np.empty([combinations, 1], dtype=object)
        for i in range(length):
            outputArray[i] = inputArray[i]
        for i in range(length):
            for j in range(length - (i + 1)):
                nameGeneration(inputArray[i][0], inputArray[length - j - 1][0], iteration, outputArray)
                iteration += 1
        print(outputArray.shape)
    else:
        print(inputArray[0][0])