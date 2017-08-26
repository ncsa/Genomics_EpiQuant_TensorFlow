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

def generateNames(name1, name2, position, outputArray):
    outputArray[position] = ["%s%s%s" % (name1, "::", name2)]

def generateData(array1, array2, position, outputArray):
    outputArray[position] = np.multiply(array1, array2)

def calculateInteractions(inputArray):
    length = len(inputArray)
    combinations = length + (math.factorial(length) // 2 // math.factorial(length - 2))
    iteration = length
    isString = isinstance(inputArray[0][0], str)
    outputArray = None

    if isString:
        outputArray = np.empty([combinations, 1], dtype=object)
    else:
        outputArray = np.empty([combinations, len(inputArray[0])], dtype=object)

    for i in range(length):
        outputArray[i] = inputArray[i]
    for i in range(length):
        for j in range(length - (i + 1)):
            if isString:
                generateNames(inputArray[i][0], inputArray[length - j - 1][0], iteration, outputArray)
            else:
                generateData(inputArray[i], inputArray[length - j - 1], iteration, outputArray)
            iteration += 1
    print(outputArray, outputArray.shape)