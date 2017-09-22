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
    print(" [", "{:6.2f}".format((i + 1) / length * 100) + "%", "]", message, end="\r")
    sys.stdout.flush()
    if i + 1 == length:
        print("\n")

def log_training(past_loss, current_loss, alpha, step, app_time):
    print(
        " [", app_time.get_time(), "]",
        "   Step:", "{:8d}".format(step),
        "   Loss:", "{:.2E}".format(current_loss),
        "   Delta:", "{:.2E}".format(abs(past_loss-current_loss)),
        "   Alpha:", "{:.2E}".format(alpha),
        "\n"
    )