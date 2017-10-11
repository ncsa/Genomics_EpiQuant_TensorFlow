""" Progress.py

Tracks progress of loops and neural network training.
"""

import sys

def progress(i, length, message):
    """ Tracks progress of a loop.

    Args:
        i: The current iteration.
        length: The total number of iterations.
        message: The message to be displayed along with the current progress.

    Returns:
        None
    """
    print(" [", "{:6.2f}".format((i + 1) / length * 100) + "%", "]", message, end='\r')
    sys.stdout.flush()
    if i + 1 == length:
        print("\n")

def log_training(accuracy, past_accuracy, alpha, step, app_time):
    """ Logs neural network training.

    Args:
        accuracy: Current model accuracy.
        past_accuracy: Past model accuracy.
        alpha: The loss difference to exit the training loop.
        step: The current training step number.
        app_time: Tracks the current running time of the application.

    Returns:
        None
     """
    print(
        " [", app_time.get_time(), "]",
        " Step:", "{:6d}".format(step),
        " Past:", "{:.2E}".format(past_accuracy),
        " Current:", "{:.2E}".format(accuracy),
        " Delta:", "{:.2E}".format(abs(past_accuracy - accuracy))
        " Alpha:", "{:.2E}".format(alpha),
        "\n"
    )
