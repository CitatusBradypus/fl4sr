#! /usr/bin/env python3.8
import sys
import os
HOME = os.environ['HOME']
sys.path.append(HOME + '/catkin_ws/src/fl4sr/src')
import numpy as np

class World():
    '''
    Data container for basic world variables.
    '''
    def __init__(self, 
        robot_alives: list,
        x_starts: list,
        y_starts: list,
        target_positions: list
        ) -> None:

        assert len(x_starts) == len(y_starts) == len(target_positions) == len(robot_alives), 'World creation len error!'
        
        self.robot_count = sum(robot_alives)
        self.robot_alives = robot_alives
        self.robot_indexes = np.where(robot_alives)[0]

        self.x_starts = [x_starts[i] for i in self.robot_indexes] # x_starts
        self.y_starts = [y_starts[i] for i in self.robot_indexes] # y_starts
        self.target_positions = [target_positions[i] for i in self.robot_indexes] # target_positions


BASELINE_WORLD = World(
    robot_alives=[True],
    x_starts=[6.0],
    y_starts=[6.0],
    target_positions=[[2.0, 2.0]]
)

TURTLEBOT_WORLD_6 = World(
    robot_alives=[True, 
                  True, 
                  True, 
                  True, 
                  True, 
                  True],
    x_starts=[[10.0], 
             [1.7], 
             [-6.0], 
             [12.0], 
             [0.0], 
             [-12.0]],
    y_starts=[[9.0], 
             [9.0], 
             [9.0], 
             [-9.0], 
             [-9.0], 
             [-9.0]],
    target_positions=[[6.0, 5.0], 
                      [-2.0, 5.0], 
                      [-10.0, 5.0], 
                      [8.0, -5.0], 
                      [-4.0, -5.0], 
                      [-16.0, -5.0]]
)

TURTLEBOT_WORLD_5 = World(
    robot_alives=[True, 
                  False, 
                  True, 
                  True, 
                  True, 
                  True],
    x_starts=[[10.0], 
             [1.7], 
             [-6.0], 
             [12.0], 
             [0.0], 
             [-12.0]],
    y_starts=[[9.0], 
             [9.0], 
             [9.0], 
             [-9.0], 
             [-9.0], 
             [-9.0]],
    target_positions=[[6.0, 5.0], 
                      [-2.0, 5.0], 
                      [-10.0, 5.0], 
                      [8.0, -5.0], 
                      [-4.0, -5.0], 
                      [-16.0, -5.0]]
)

EVAL_WORLD_0 = World(
    robot_alives=[True],
    x_starts=[[0.0]],
    y_starts=[[0.0]],
    target_positions=[[5.0, 5.0]]
)

EVAL_WORLD_1 = World(
    robot_alives=[True],
    x_starts=[[0.0]],
    y_starts=[[0.0]],
    target_positions=[[-5.0, 5.0]]
)

EVAL_WORLD_2 = World(
    robot_alives=[True],
    x_starts=[[0.0]],
    y_starts=[[0.0]],
    target_positions=[[5.0, -5.0]]
)

EVAL_WORLD_3 = World(
    robot_alives=[True],
    x_starts=[[0.0]],
    y_starts=[[0.0]],
    target_positions=[[-5.0, -5.0]]
)

TURTLEBOT_WORLD_5_STARTS = World(
    robot_alives=[True,
                  False,
                  True,
                  True,
                  True,
                  True],
    x_starts=[[10.0, 8.0, 10.0],
             [1.7],
             [-6.0, -8.0, -6.0],
             [12.0, 12.0, 15.5, 15.5],
             [0.0, 3.0, -3.0, 3.0],
             [-12.0, -8.0, -12.0, -9.0]],
    y_starts=[[9.0, 9.0, 7.0],
             [9.0],
             [9.0, 9.0, 7.0],
             [-9.0, -12.5, -9.0, -12.5],
             [-9.0, -6.0, -12.0, -12.0],
             [-9.0, -9.0, -13.0, -12.0]],
    target_positions=[[6.0, 5.0],
                      [-2.0, 5.0],
                      [-10.0, 5.0],
                      [8.0, -5.0],
                      [-4.0, -5.0],
                      [-16.0, -5.0]]
)

# TODO needs to be modified after setting up WhyCODE
REAL_WORLD = World(
    robot_alives=[True],
    x_starts=[[0.7]],
    y_starts=[[0.3]],
    target_positions=[[3.4, 2.8]]
)
REAL_SIM_WORLD = World(
    robot_alives=[True],
    x_starts=[[0.5]],
    y_starts=[[0.5]],
    target_positions=[[3.6, 2.8]]
)


REAL_WORLD_8 = World(
    robot_alives=[True, 
                  True, 
                  True, 
                  True, 
                  True, 
                  True,
                  True,
                  True],
    x_starts=[[0.6], 
             [0.6], 
             [0.6], 
             [0.6], 
             [-4.5], 
             [-4.5],
             [-4.5],
             [-4.5]],
    y_starts=[[5.5], 
             [0.5],
             [-4.5],
             [-9.5], 
             [7.8], 
             [2.8],
             [-2.2], 
             [-7.2]],
    target_positions=[[3.8, 7.8], 
                      [3.8, 2.8], 
                      [3.8, -2.2], 
                      [3.8, -7.2], 
                      [-1.2, 5.2],
                      [-1.2, 0.2],
                      [-1.2, -4.8], 
                      [-1.2, -9.8]]
)

REAL_WORLD_4_diff_reward = World(
    robot_alives=[True, 
                  True, 
                  True, 
                  True],
    x_starts=[[0.6], 
             [0.6],
             [-4.5],
             [-4.5]],
    y_starts=[[5.5], 
             [0.5],
             [-2.2], 
             [-7.2]],
    target_positions=[[3.8, 7.8], 
                      [3.8, 2.8], 
                      [-1.2, -4.8], 
                      [-1.2, -9.8]]
)