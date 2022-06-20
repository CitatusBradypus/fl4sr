#! /home/swacil/anaconda3/envs/robostackenv/bin/python

import sys
import os
HOME = os.environ['HOME']
sys.path.append(HOME + '/catkin_ws/src/fl4sr/src')
from math import sqrt
import time
import numpy as np
import rospy

from worlds import World
from InfoGetter import InfoGetter

from geometry_msgs.msg import Twist
from rospy.service import ServiceException
from sensor_msgs.msg import LaserScan
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from std_srvs.srv import Empty
from tf.transformations import euler_from_quaternion

class ScanTest():

    def __init__(self):
        self.COLLISION_RANGE = 0.2
        self.node = rospy.init_node('scan_test', anonymous=True)
        self.rate = rospy.Rate(1)
        self.laser_count = 24

        self.robot_count = 1

        # lasers info getters, subscribers unused
        self.laser_info_getter = [InfoGetter() for i in range(self.robot_count)]
        self._laser_subscriber = \
            [rospy.Subscriber('/scan', 
                              LaserScan, 
                              self.laser_info_getter[id]) 
            for id in range(self.robot_count)]

    def get_robot_lasers_collisions(self):
        lasers = []
        collisions = [False for _ in range(self.robot_count)]
        for i in range(self.robot_count):
            lasers.append([])
            scan = self.laser_info_getter[i].get_msg()
            for j in range(len(scan.ranges)):
                lasers[i].append(0)
                if scan.ranges[j] == float('Inf') or scan.ranges[j] >= 3.5:
                    lasers[i][j] = 3.5
                elif np.isnan(scan.ranges[j]):
                    lasers[i][j] = 0
                else:
                    lasers[i][j] = scan.ranges[j]
                if self.COLLISION_RANGE > min(lasers[i]) > 0:
                    collisions[i] = True
        lasers = np.array(lasers)
        collisions = np.array(collisions)
        return lasers, collisions

    def laser_check(self):
            lasers, collisions = self.get_robot_lasers_collisions()
            print(f"lasers: {lasers}, collisions: {collisions}")
            self.rate.sleep()


if __name__=="__main__":
    scan_test = ScanTest()
    scan_test.laser_check()





        
