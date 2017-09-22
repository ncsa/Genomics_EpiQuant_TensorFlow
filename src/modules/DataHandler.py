""" Data Handler

Imports the data and calculates pairwise combinations.
"""

import math
import numpy as np
import Progress as prog

def generate_names(name_1, name_2, position, output_array):
    """ Generates all possible name combinations.

    Args:
        name_1: The first name to concatenate.
        name_2: The second name to concatenate.
        position: The position to place names.
        ouputArray: Contains all name combinations.

    Returns:
        None
    """
    output_array[position] = ["%s%s%s" % (name_1, "::", name_2)]

def generate_data(array_1, array_2, position, output_array):
    """ Generates all possible data combinations.

    Args:
        array_1: The first array of data points.
        array_2: The second array of data points.
        position: The position to place data.
        output_array: Contains all data point combinations.

    Returns:
        None
    """
    output_array[position] = np.multiply(array_1, array_2)

def calculate_interactions(input_array):
    """ Calculates all pairwise epistatic interactions.

    Args:
        input_array: Contains all single epistatic contributions.

    Returns:
        output_array: Contains all single and pairwise interaction contributions.
    """
    length = len(input_array)
    combinations = length + (math.factorial(length) // 2 // math.factorial(length - 2))
    iteration = length
    is_string = isinstance(input_array[0][0], str)
    output_array = None

    # Pre-allocate output_array
    if is_string:
        output_array = np.empty([combinations, 1], dtype=object)
    else:
        output_array = np.empty([combinations, len(input_array[0])], dtype=object)

    # Assign all single contributions
    for i in range(length):
        output_array[i] = input_array[i]

    # Calculate pairwise interactions
    for i in range(length):
        for j in range(length - (i + 1)):
            if is_string:
                generate_names(input_array[i][0],
                               input_array[length - j - 1][0],
                               iteration, output_array)
            else:
                generate_data(input_array[i], input_array[length - j - 1], iteration, output_array)
            iteration += 1
        prog.progress(i, length, "Interactions Calculated")

    return output_array

def get_data(file_path, get_combinations):
    """ Reads in data from a file and formats it into a name and data array.

    Args:
        file_path: The path to the file to be read in.
        get_combinations: Calculate epistatic interactions.

    Returns:
        name_array: Contains the names of each data row.
        data_array: Contains the data elements of each row.
    """
    with open(file_path) as input_file:
        # Reads in data from a file.
        # Strips white space and newlines from the ends.
        # Splits by newlines then tabs for each split.
        full_array = np.asarray([x.split("\t") for x in input_file.read().strip().split("\n")])
        # Slice off name column and delete name column from data array.
        name_array = np.asarray([[x] for x in full_array[:, 0]])
        # Convert data array to floating point.
        data_array = np.delete(full_array, 0, 1).astype(np.float32)
        if not get_combinations:
            return name_array, data_array
        else:
            return calculate_interactions(name_array), calculate_interactions(data_array)
