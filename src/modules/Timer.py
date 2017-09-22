""" Timer.py

Starts a timer to track runtime of the application.
"""

import time

class Timer:
    """ Class that defines a Timer. """
    def __init__(self):
        """ Starts the timer. """
        self.start = time.time()

    def restart(self):
        """ Restarts the timer. """
        self.start = time.time()

    def get_time(self):
        """ Gets the current time. """
        end = time.time()
        min, sec = divmod(end - self.start, 60)
        hour, min = divmod(min, 60)
        time_str = "%03d:%02d:%02d" % (hour, min, sec)
        return time_str
