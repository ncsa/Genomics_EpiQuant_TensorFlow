import numpy as np
import math

def generateNames(name1, name2, position, outputArray):
    """ Generates all possible name combinations.

    Args:
        name1: The first name to concatenate.
        name2: The second name to concatenate.
        position: The position to place names.
        ouputArray: Contains all name combinations.

    Returns:
        None
    """
    outputArray[position] = ["%s%s%s" % (name1, "::", name2)]

def generateData(array1, array2, position, outputArray):
    """ Generates all possible data combinations.

    Args:
        array1: The first array of data points.
        array2: The second array of data points.
        position: The position to place data.
        outputArray: Contains all data point combinations.

    Returns:
        None
    """
    outputArray[position] = np.multiply(array1, array2)

def calculateInteractions(inputArray):
    """ Calculates all pairwise epistatic interactions.
    
    Args:
        inputArray: Contains all single epistatic contributions.

    Returns:
        outputArray: Contains all single and pairwise interaction contributions.
    """
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

def getData(filePath):
    """ Reads in data from a file and formats it into a name and data array.

    Args:
        filePath: The path to the file to be read in.

    Returns:
        nameArray: Contains the names of each data row.
        dataArray: Contains the data elements of each row.
    """
    with open(filePath) as file:
        # Reads in data from a file.
        # Strips white space and newlines from the ends.
        # Splits by newlines then tabs for each split.   
        fullArray = np.asarray([x.split("\t") for x in file.read().strip().split("\n")])
        # Slice off name column and delete name column from data array.    
        nameArray = np.asarray([[x] for x in fullArray[:, 0]])
        # Convert data array to floating point.
        dataArray = np.delete(fullArray, 0, 1).astype(np.float)
        return calculateInteractions(nameArray), calculateInteractions(dataArray)