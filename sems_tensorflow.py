""" sems_tensorflow.py

Main driver script for stepwise epistatic model selection.
"""

import os

PBS_JOBID = os.environ['PBS_JOBID']

def main():
    print(PBS_JOBID)
    print('/var/spool/torque/aux/' + PBS_JOBID)

if __name__ == "__main__":
    main()
