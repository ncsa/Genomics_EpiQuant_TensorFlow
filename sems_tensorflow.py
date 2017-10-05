""" sems_tensorflow.py

Main driver script for stepwise epistatic model selection.
"""

import os

PBS_JOBID = os.environ['PBS_JOBID']
PBS_HOSTNAMES = '/var/spool/torque/aux/' + PBS_JOBID

def main():
    """ Main of sems_tensorflow """
    with open(PBS_HOSTNAMES) as host_file:
        workers = [(x.strip() + ':2222') for x in host_file.readlines()]
    master = [workers.pop()]
    print(master, workers)

if __name__ == "__main__":
    main()
