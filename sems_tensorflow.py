""" sems_tensorflow.py

Main driver script for stepwise epistatic model selection.
"""

import os

PBS_JOBID = os.environ['PBS_JOBID']
PBS_HOSTNAMES = '/var/spool/torque/aux/' + PBS_JOBID

def main():
    """ Main of sems_tensorflow """
    with open(PBS_HOSTNAMES) as host_file:
        host_names = [(x.strip() + ':2222') for x in host_file.readlines()]
    print(host_names)

if __name__ == "__main__":
    main()
