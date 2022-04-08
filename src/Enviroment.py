#! /usr/bin/env python3.8
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


class Enviroment():
    """Similar class as openAI gym Env. 
    
    Is able to: reset, step.
    """

    def __init__(self, 
        world: World
        ) -> None:
        """Initializes eviroment.

        Args:
            world (World): Holds enviroment variables
        """
        # params        
        self.COLLISION_RANGE = 0.25
        self.GOAL_RANGE = 0.5
        self.REWARD_GOAL = 100.0
        self.REWARD_COLLISION = -10.0
        self.PROGRESS_REWARD_FACTOR = 40.0
        # simulation services
        # rospy.wait_for_service('/gazebo/reset_simulation')
        # self.reset_simulation = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        rospy.wait_for_service('/gazebo/reset_world')
        self.reset_world = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        rospy.wait_for_service('/gazebo/pause_physics')
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        rospy.wait_for_service('/gazebo/unpause_physics')
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause()
        # world settings
        self.robot_count = world.robot_count
        self.robot_alives = world.robot_alives
        self.robot_indexes = world.robot_indexes

        self.x_starts_all = world.x_starts
        self.y_starts_all = world.y_starts

        self.x_starts = [x[0] for x in world.x_starts]
        self.y_starts = [y[0] for y in world.y_starts]

        self.targets = world.target_positions
        self.x_targets = np.array(self.targets).T[0]
        self.y_targets = np.array(self.targets).T[1]
        # create restart enviroment messages
        self.reset_tb3_messages = \
            [self.create_model_state('tb3_{}'.format(rid), 
                                     self.x_starts[id], 
                                     self.y_starts[id],
                                     -0.2)
            for id, rid in enumerate(self.robot_indexes)]
        self.reset_target_messages = \
            [self.create_model_state('target_{}'.format(rid), 
                                     self.targets[id][0], 
                                     self.targets[id][1], 
                                     0)
            for id, rid in enumerate(self.robot_indexes)]
        self.command_empty = Twist()

        # basic settings
        self.node = rospy.init_node('turtlebot_env', anonymous=True)
        self.rate = rospy.Rate(100)
        self.laser_count = 24
        
        self.observation_dimension = self.laser_count + 4
        self.action_dimension = 2

        # publishers for turtlebots
        self.publisher_turtlebots = \
            [rospy.Publisher('/tb3_{}/cmd_vel'.format(i), 
                             Twist, 
                             queue_size=1) 
            for i in self.robot_indexes]
        # positional info getter
        self.position_info_getter = InfoGetter()
        self._position_subscriber = rospy.Subscriber("/gazebo/model_states", 
                                                     ModelStates, 
                                                     self.position_info_getter)
        # lasers info getters, subscribers unused
        self.laser_info_getter = [InfoGetter() for i in range(self.robot_count)]
        self._laser_subscriber = \
            [rospy.Subscriber('/tb3_{}/scan'.format(rid), 
                              LaserScan, 
                              self.laser_info_getter[id]) 
            for id, rid in enumerate(self.robot_indexes)]

        # various simulation outcomes
        self.robot_finished = np.zeros((self.robot_count), dtype=bool)
        self.robot_succeeded = np.zeros((self.robot_count), dtype=bool)
        # previous and current distances
        self.robot_target_distances_previous = self.get_distance(
            self.x_starts, 
            self.x_targets, 
            self.y_starts, 
            self.y_targets)
        return

    def reset(self,
        robot_id: int=-1
        ) -> None:
        """Resets robots to starting state.
           
           If robot_id is empty all robots will be reseted.

        Args:
            robot_id (int, optional): Id of robot to reset. 
                Defaults to -1.
        """
        # wait for services
        rospy.wait_for_service('/gazebo/reset_simulation')
        rospy.wait_for_service('/gazebo/set_model_state')
        # set model states or reset world
        if robot_id == -1:
            try:
                state_setter = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
                for id, rid in enumerate(self.robot_indexes):
                    # pick new starting position and direction and set them
                    start_index = np.random.randint(len(self.x_starts_all[id]))
                    self.x_starts[id] = self.x_starts_all[id][start_index]
                    self.y_starts[id] = self.y_starts_all[id][start_index]
                    direction = -0.2 # + (np.random.rand() * np.pi / 2) - (np.pi / 4)
                    # generate new message
                    self.reset_tb3_messages[id] = \
                        self.create_model_state('tb3_{}'.format(rid), 
                                             self.x_starts[id], 
                                             self.y_starts[id],
                                             direction)
                    # reset enviroment position
                    state_setter(self.reset_tb3_messages[id])
                    state_setter(self.reset_target_messages[id])
                    self.robot_finished[id] = False
            except rospy.ServiceException as e:
                print('Failed state setter!', e)
            #self.reset_world()
        else:
            try:
                state_setter = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
                state_setter(self.reset_tb3_messages[robot_id])
                state_setter(self.reset_target_messages[robot_id])
                self.robot_finished[robot_id] = False
            except rospy.ServiceException as e:
                print('Failed state setter!', e)
        # set robot move command
        if robot_id == -1:
            for i in range(self.robot_count):
                self.publisher_turtlebots[i].publish(self.command_empty)
                self.robot_target_distances_previous = self.get_distance(
                    self.x_starts, 
                    self.x_targets, 
                    self.y_starts, 
                    self.y_targets)
                self.robot_succeeded = np.zeros((self.robot_count), dtype=bool)
        else:
            self.publisher_turtlebots[robot_id].publish(self.command_empty)            
            self.robot_target_distances_previous[robot_id] = \
                sqrt(
                     (self.x_starts[robot_id] - self.x_targets[robot_id])**2 
                     + (self.y_starts[robot_id] - self.y_targets[robot_id])**2)
        # wait for new scan message, so that laser values are updated
        # kinda cheeky but it works on my machine :D
        self.unpause()
        rospy.wait_for_message('/tb3_{}/scan'.format(self.robot_indexes[0]), LaserScan)
        self.pause()
        return
    
    def step(self,
        actions: np.ndarray,
        time_step: float=0.1
        ) -> tuple:
        """Perform one step of simulations using given actions and lasting for 
        set time.

        Args:
            actions (np.ndarray): Action of each robot.
            time_step (float, optional): Duration of taken step. 
                Defaults to 0.1.

        Returns:
            tuple: states (np.ndarray), 
                   rewards (np.ndarray), 
                   robot_finished (list), 
                   robot_succeeded (list),
                   error (bool)
                   data (dict)
        """
        assert len(actions) == self.robot_count, 'Wrong actions dimension!'
        # generate twists, also get separate values of actions
        twists = [self.action_to_twist(action) for action in actions]
        actions_linear_x = actions.T[1]
        actions_angular_z = actions.T[0]
        # publish twists
        # self.pause()
        for i in range(self.robot_count):
            self.publisher_turtlebots[i].publish(twists[i])
        # start of timing !!! changed to rospy time !!!
        start_time = rospy.get_time()
        running_time = 0
        # move robots with action for time_step        
        self.unpause()
        while(running_time < time_step):
            self.rate.sleep()
            running_time = rospy.get_time() - start_time
        self.pause()
        # send empty commands to robots
        # self.unpause()
        # read current positions of robots
        model_state = self.position_info_getter.get_msg()
        robot_indexes = self.get_robot_indexes_from_model_state(model_state)
        x, y, theta, correct = self.get_positions_from_model_state(model_state, 
                                                                   robot_indexes)
        # check for damaged robots
        if np.any(np.isnan(correct)):
            print('ERROR: Enviroment: nan robot twist detected!')
            return None, None, None, None, True, None

        theta = theta % (2 * np.pi)
        # get current distance to goal
        robot_target_distances = self.get_distance(x, self.x_targets, 
                                                   y, self.y_targets)
        # get current robot angles to targets
        robot_target_angle = self.get_angle(self.x_targets, x, 
                                            self.y_targets, y)
        robot_target_angle = robot_target_angle % (2 * np.pi)
        robot_target_angle_difference = (robot_target_angle - theta - np.pi) % (2 * np.pi) - np.pi
        # get current laser measurements
        robot_lasers, robot_collisions = self.get_robot_lasers_collisions()

        # create state array 
        # = lasers (24), 
        #   action linear x (1), action angular z (1), 
        #   distance to target (1), angle to target (1)
        # = dimension (6, 28)
        s_actions_linear = actions_linear_x.reshape((self.robot_count, 1))
        s_actions_angular = actions_angular_z.reshape((self.robot_count, 1))
        s_robot_target_distances = robot_target_distances.reshape((self.robot_count, 1))
        s_robot_target_angle_difference = robot_target_angle_difference.reshape((self.robot_count, 1))
        assert robot_lasers.shape == (self.robot_count, 24), 'Wrong lasers dimension!'
        assert s_actions_linear.shape == (self.robot_count, 1), 'Wrong action linear dimension!'
        assert s_actions_angular.shape == (self.robot_count, 1), 'Wrong action angular dimension!'
        assert s_robot_target_distances.shape == (self.robot_count, 1), 'Wrong distance to target!'
        assert s_robot_target_angle_difference.shape == (self.robot_count, 1), 'Wrong angle to target!'
        states = np.hstack((robot_lasers, 
                            s_actions_linear, s_actions_angular, 
                            s_robot_target_distances, s_robot_target_angle_difference))
        assert states.shape == (self.robot_count, self.observation_dimension), 'Wrong states dimension!'
        
        # rewards
        # distance rewards
        # CHECK for possible huge value after reset
        reward_distance = self.PROGRESS_REWARD_FACTOR * (self.robot_target_distances_previous - robot_target_distances)
        # reward_distance = - np.e ** (0.25 * robot_target_distances)
        # goal reward
        reward_goal = np.zeros(self.robot_count)
        reward_goal[robot_target_distances < self.GOAL_RANGE] = self.REWARD_GOAL
        self.robot_finished[robot_target_distances < self.GOAL_RANGE] = True
        self.robot_succeeded[robot_target_distances < self.GOAL_RANGE] = True
        # collision reward
        reward_collision = np.zeros(self.robot_count)
        reward_collision[np.where(robot_collisions)] = self.REWARD_COLLISION
        self.robot_finished[np.where(robot_collisions)] = True
        # total reward
        rewards = reward_distance + reward_goal + reward_collision
        
        # set current target distance as previous
        distances_help = self.robot_target_distances_previous.copy()
        self.robot_target_distances_previous = robot_target_distances.copy()
        # restart robots
        robot_finished = self.robot_finished.copy()
        for i in range(self.robot_count):
            if self.robot_finished[i]:
                self.reset(i)
        # additional data to send
        data = {}
        data['x'] = x
        data['y'] = y
        data['theta'] = theta

        return states, rewards, robot_finished, self.robot_succeeded, False, data

    '''
    def init_subscribers(self
        ) -> None:
        # basic settings
        self.node = rospy.init_node('turtlebot_env', anonymous=True)
        self.rate = rospy.Rate(100)
        # ...
        rospy.wait_for_service('/gazebo/pause_physics')
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        rospy.wait_for_service('/gazebo/unpause_physics')
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        # positional info getter
        self.position_info_getter = InfoGetter()
        self._position_subscriber = rospy.Subscriber("/gazebo/model_states",
                                                     ModelStates,
                                                     self.position_info_getter)
        # lasers info getters, subscribers unused
        self.laser_info_getter = [InfoGetter() for i in range(self.robot_count)]
        self._laser_subscriber = \
            [rospy.Subscriber('/tb3_{}/scan'.format(i),
                              LaserScan,
                              self.laser_info_getter[i])
            for i in range(self.robot_count)]
        rospy.wait_for_message('/tb3_0/scan', LaserScan)
        return
    '''

    def create_model_state(self, 
        name: str,
        pose_x: float,
        pose_y: float,
        orientation_z: float,
        ) -> ModelState:
        """Creates basic ModelState with specified values. 
        Other values are set to zero.

        Args:
            name (str): Name.
            pose_x (float): Value of ModelState.pose.x
            pose_y (float): Value of ModelState.pose.y
            orientation_z (float): Value of ModelState.oritentation.z

        Returns:
            ModelState: Initialized model state.
        """
        model_state = ModelState()
        model_state.model_name = name
        model_state.pose.position.x = pose_x
        model_state.pose.position.y = pose_y
        model_state.pose.position.z = 0
        model_state.pose.orientation.x = 0.0
        model_state.pose.orientation.y = 0.0
        model_state.pose.orientation.z = orientation_z
        model_state.pose.orientation.w = 0.0
        return model_state

    def get_robot_indexes_from_model_state(self,
        model_state: ModelStates=None
        ) -> list:
        """Creates list with indexes of robots in model state.

        Args:
            model_state (ModelStates, optional): Source of robot indexes. 
                Defaults to None.

        Returns:
            list: Robot indexes. ('tb_2' index is list[2])
        """
        robots = [None for i in range(len(self.robot_alives))]
        if model_state is None:
            model_state = self.position_info_getter.get_msg()
        for i in range(len(model_state.name)):
            if 'tb3' in model_state.name[i]:
                robots[int(model_state.name[i][-1])] = i
        return robots

    def action_to_twist(self,
        action: np.ndarray,
        ) -> Twist:
        """Transforms action 2d ndarray to Twist message.

        Args:
            action (np.ndarray): Ndarray 2d.

        Returns:
            Twist: Transformed message from ndarray.
        """
        assert len(action) == 2, 'Wrong action dimension!'
        twist = Twist()
        twist.linear.x = action[1] * 0.25
        twist.angular.z = action[0] # * (np.pi / 2)
        return twist

    def get_distance(self, 
        x_0: np.ndarray,
        x_1: np.ndarray,
        y_0: np.ndarray,
        y_1: np.ndarray
        ) -> np.ndarray:
        """Returns distance between two arrays of positions.

        Args:
            x_0 (np.ndarray): X positions of first points
            x_1 (np.ndarray): X positions of second points
            y_0 (np.ndarray): Y positions of first points
            y_1 (np.ndarray): Y positions of second points

        Returns:
            np.ndarray: Distances between points.
        """
        return np.sqrt((np.square(x_0 - x_1) + np.square(y_0 - y_1))) 

    def get_positions_from_model_state(self,
        model_state: ModelStates,
        robot_indexes: list
        ) -> tuple:
        """Get positional information from model_state

        Args:
            model_state (ModelStates): Information source.
            robot_indexes (list): List of robot indexes.

        Returns:
            tuple: x, y, theta ndarrays of robots
        """
        x, y, theta, correct = [], [], [], []
        for rid in self.robot_indexes:
            index = robot_indexes[rid]
            pose = model_state.pose[index]
            twist = model_state.twist[index]
            x.append(pose.position.x)
            y.append(pose.position.y)
            theta.append(euler_from_quaternion((pose.orientation.x, 
                                                pose.orientation.y, 
                                                pose.orientation.z, 
                                                pose.orientation.w,))[2])
            correct.append(twist.angular.x)
        x = np.array(x)
        y = np.array(y)
        theta = np.array(theta)
        correct = np.array(correct)
        return x, y, theta, correct

    def get_angle(self,
        x_0: np.ndarray,
        x_1: np.ndarray,
        y_0: np.ndarray,
        y_1: np.ndarray
        ) -> np.ndarray:
        """Returns base angle value between array of two points.

        Args:
            x_0 (np.ndarray): Source x values.
            x_1 (np.ndarray): Positional x values.
            y_0 (np.ndarray): Source y values.
            y_1 (np.ndarray): Positional y values.

        Returns:
            np.ndarray: Angles between array of two points.
        """
        x_diff = x_0 - x_1
        y_diff = y_0 - y_1
        return np.arctan2(y_diff, x_diff)

    def get_robot_lasers_collisions(self,
        ) -> tuple:
        """Returns values of all robots lasers and if robots collided.

        Returns:
            tuple: lasers, collisions
        """
        lasers = []
        collisions = [False for i in range(self.robot_count)]
        # each robot
        for i in range(self.robot_count):
            lasers.append([])
            scan = self.laser_info_getter[i].get_msg()
            # each laser in scan
            for j in range(len(scan.ranges)):
                lasers[i].append(0)
                if scan.ranges[j] == float('Inf'):
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

    def get_starting_states(self
        ) -> np.ndarray:
        """Returns starting states.

        Returns:
            np.ndarray: Starting states.
        """
        model_state = self.position_info_getter.get_msg()
        robot_indexes = self.get_robot_indexes_from_model_state(model_state)
        x, y, theta, _ = self.get_positions_from_model_state(model_state, 
                                                             robot_indexes)
        # get current distance to goal
        robot_target_distances = self.get_distance(x, self.x_targets, 
                                                   y, self.y_targets)
        # get current robot angles to targets
        robot_target_angle = self.get_angle(self.x_targets, x, 
                                            self.y_targets, y)
        robot_target_angle_difference = (robot_target_angle - theta - np.pi) % (2 * np.pi) - np.pi
        # get current laser measurements
        robot_lasers, robot_collisions = self.get_robot_lasers_collisions()
        
        # create state array 
        # = lasers (24), 
        #   action linear x (1), action angular z (1), 
        #   distance to target (1), angle to target (1)
        # = dimension (6, 28)
        s_actions_linear = np.zeros((self.robot_count, 1))
        s_actions_angular = np.zeros((self.robot_count, 1))
        s_robot_target_distances = robot_target_distances.reshape((self.robot_count, 1))
        s_robot_target_angle_difference = robot_target_angle_difference.reshape((self.robot_count, 1))
        assert robot_lasers.shape == (self.robot_count, 24), 'Wrong lasers dimension!'
        assert s_actions_linear.shape == (self.robot_count, 1), 'Wrong action linear dimension!'
        assert s_actions_angular.shape == (self.robot_count, 1), 'Wrong action angular dimension!'
        assert s_robot_target_distances.shape == (self.robot_count, 1), 'Wrong distance to target!'
        assert s_robot_target_angle_difference.shape == (self.robot_count, 1), 'Wrong angle to target!'
        states = np.hstack((robot_lasers, 
                           s_actions_linear, s_actions_angular, 
                           s_robot_target_distances, s_robot_target_angle_difference))
        assert states.shape == (self.robot_count, self.observation_dimension), 'Wrong states dimension!'
        return states
