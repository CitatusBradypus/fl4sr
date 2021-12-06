#! /usr/bin/env python3.8
import sys
import os
HOME = os.environ['HOME']
sys.path.append(HOME + '/catkin_ws/src/fl4sr/src')
from IndividualDDPG import IndividualDDPG
from SharedNetworkDDPG import SharedNetworkDDPG
from SharedExperienceDDPG import SharedExperienceDDPG
from FederatedLearningDDPG import FederatedLearningDDPG
from worlds import BASELINE_WORLD
from worlds import TURTLEBOT_WORLD

path_data = '/home/pikmanjan/catkin_ws/src/fl4sr/src/data'
paths_actor = [path_data + '/FederatedLearningDDPG-20211129-230152/weights/actor-final-0.pkl',
               path_data + '/FederatedLearningDDPG-20211129-230152/weights/actor-final-1.pkl',
               path_data + '/FederatedLearningDDPG-20211129-230152/weights/actor-final-2.pkl',
               path_data + '/FederatedLearningDDPG-20211129-230152/weights/actor-final-3.pkl',
               path_data + '/FederatedLearningDDPG-20211129-230152/weights/actor-final-4.pkl',
               path_data + '/FederatedLearningDDPG-20211129-230152/weights/actor-final-5.pkl']
paths_critic = [path_data + '/FederatedLearningDDPG-20211129-230152/weights/critic-final-0.pkl',
                path_data + '/FederatedLearningDDPG-20211129-230152/weights/critic-final-1.pkl',
                path_data + '/FederatedLearningDDPG-20211129-230152/weights/critic-final-2.pkl',
                path_data + '/FederatedLearningDDPG-20211129-230152/weights/critic-final-3.pkl',
                path_data + '/FederatedLearningDDPG-20211129-230152/weights/critic-final-4.pkl',
                path_data + '/FederatedLearningDDPG-20211129-230152/weights/critic-final-5.pkl']
                
test = FederatedLearningDDPG(1000, 256, TURTLEBOT_WORLD)
#test.agents_load(paths_actor, paths_critic)
test.run()

#test = FederatedLearningDDPG(300, 256, TURTLEBOT_WORLD)
#test.agents_load(paths_actor, paths_critic)
#test.EPSILON = 0.0
#test.run()
