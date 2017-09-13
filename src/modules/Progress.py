def progress(i, length, message):
    """ Tracks progress of a loop.
    
    Args:
        i: The current iteration.
        length: The total number of iterations.
        message: The message to be displayed along with the current progress.

    Returns:
        None
    """
    print(" [", "{:6.2f}".format((i + 1) / length * 100) + "%", "]", message, end="\r")
    if i + 1 == length:
        print("\n")