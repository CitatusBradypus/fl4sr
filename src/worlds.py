#! /usr/bin/env python3.8
import sys
import os
HOME = os.environ['HOME']
sys.path.append(HOME + '/catkin_ws/src/fl4sr/src')

class World():
    '''
    Data container for basic world variables.
    '''
    def __init__(self, 
        robot_count: int,
        x_starts: list,
        y_starts: list,
        target_positions: list
        ) -> None:

        assert len(x_starts) == len(y_starts) == len(target_positions) == robot_count, 'World creation len error!'
        
        self.robot_count = robot_count
        self.x_starts = x_starts
        self.y_starts = y_starts
        self.target_positions = target_positions


BASELINE_WORLD = World(
    robot_count=1,
    x_starts=[6.0],
    y_starts=[6.0],
    target_positions=[[2.0, 2.0]]
)

TURTLEBOT_WORLD = World(
    robot_count=6,
    x_starts=[10.0, 
             1.7, 
             -6.0, 
             12.0, 
             0.0, 
             -12.0],
    y_starts=[9.0, 
             9.0, 
             9.0, 
             -9.0, 
             -9.0, 
             -9.0],
    target_positions=[[6.0, 5.0], 
                      [-2.0, 5.0], 
                      [-10.0, 5.0], 
                      [8.0, -5.0], 
                      [-4.0, -5.0], 
                      [-16.0, -5.0]]
)

