#! /usr/bin/python3.8
import sys
import os
import multiprocessing as mp
import time
import rospy
import random
import subprocess
import numpy as np
HOME = os.environ['HOME']
sys.path.append(HOME + '/catkin_ws/src/fl4sr/src')
from main_node import MainWorker

num_instances = 1
if __name__=="__main__":

    workers = [MainWorker(i) for i in range(num_instances)]
    [w.start() for w in workers]
    [w.join() for w in workers]

