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

def logTraining(pastLoss, currentLoss, alpha, step, appTime):
    print(
        "[", appTime.getTime(), "]",
        "   Step:", "{:8d}".format(step),
        "   Loss:", "{:.2E}".format(currentLoss),
        "   Delta:", "{:.2E}".format(abs(pastLoss-currentLoss)),
        "   Alpha:", "{:.2E}".format(alpha),
        "\n"
    )