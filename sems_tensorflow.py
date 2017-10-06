""" sems_tensorflow.py

Main driver script for stepwise epistatic model selection.
"""

import sys
import os
import argparse
import tensorflow as tf

PBS_JOBID = os.environ['PBS_JOBID']
PBS_HOSTNAMES = '/var/spool/torque/aux/' + PBS_JOBID
FLAGS = None

def main(_):
    """ Main of sems_tensorflow """
    with open(PBS_HOSTNAMES) as host_file:
        workers_hosts = [(x.strip() + ':2222') for x in host_file.readlines()]
    ps_host = [workers_hosts.pop()]
    print(ps_host, workers_hosts)

    num_ps = len(ps_host)
    num_workers = len(workers_hosts)

    print(FLAGS.job_type)

    # tf.train.ClusterSpec({
    #     'master': master_hosts,
    #     'workers': workers_hosts
    # })

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register('type', 'bool', lambda v: v.lower() == 'true')
    parser.add_argument(
        '--job_type',
        type=str,
        default="",
        help="One of 'ps', 'worker'"
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
