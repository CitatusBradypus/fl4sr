#! /usr/bin/env python3.8
import sys
import os
from matplotlib.pyplot import step
HOME = os.environ['HOME']
sys.path.append(HOME + '/catkin_ws/src/fl4sr/src')
import torch
import torch.nn as nn


class Actor(nn.Module):
    """Actor neural network.
    """

    def __init__(self,
        input_dimension: int,
        hidden_layers: list
        ) -> None:
        """Creates actor neural network. Output dimension = 2, with first output
            value is in [-1, 1], second value is in [0, 1].

        Args:
            input_dimension (int): 
            hidden_layers (list): List with sizes of hidden layers.
        """
        super(Actor, self).__init__()
        # save parameters
        self.input_dimension = input_dimension
        self.hidden_layers = hidden_layers
        self.hidden_dimension = len(hidden_layers)
        self.output_dimension = 2
        # create layers
        self.layers = nn.ModuleList()
        for i in range(self.hidden_dimension):
            if i == 0:
                self.layers.append(nn.Linear(input_dimension, hidden_layers[i]))
                self.layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            elif i == self.hidden_dimension - 1:
                self.layers.append(nn.Linear(hidden_layers[i], 1))
                nn.init.xavier_normal_(self.layers[-1].weight)
                self.layers.append(nn.Linear(hidden_layers[i], 1))
                nn.init.xavier_normal_(self.layers[-1].weight)
            else:
                self.layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
        self.layers_len = len(self.layers)
        # prepare activation functions
        self.ReLU = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        return
    
    def forward(self, 
        x: torch.tensor
        ) -> torch.tensor:
        """Run actor neural network with x as input.

        Args:
            x (torch.tensor): Input tensor.

        Returns:
            torch.tensor: 2d output tensor.
        """
        # run computation
        for i in range(0, self.layers_len - 1, 1):
            if i == self.layers_len - 2:
                output0 = self.tanh(self.layers[i](x))
                output1 = self.sigmoid(self.layers[i + 1](x))
            else:
                x = self.ReLU(self.layers[i](x))
        return torch.cat((output0, output1), dim=1)


class Critic(nn.Module):
    """Critic neural network.
    """
    def __init__(self,
        state_dimension: int,
        action_dimension: int,
        hidden_layers: list
        ) -> None:
        """Creates actor neural network. Output dimension = 1, value is 
            unbounded.

        Args:
            state_dimension (int): 
            action_dimension (int): 
            hidden_layers (list): List with sizes of hidden layers.
        """
        super(Critic, self).__init__()
        # save parameters
        self.state_dimension = state_dimension
        self.action_dimension = action_dimension
        self.hidden_layers = hidden_layers
        self.hidden_dimension = len(hidden_layers)
        self.output_dimension = 1
        # create layers
        self.layers = nn.ModuleList()
        for i in range(self.hidden_dimension):
            if i == 0:
                self.layers.append(nn.Linear(state_dimension + action_dimension, hidden_layers[i]))
                self.layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            elif i == self.hidden_dimension - 1:
                self.layers.append(nn.Linear(hidden_layers[i], 1))
            else:
               self.layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
        self.layers_len = len(self.layers)
        # prepare activation functions
        self.ReLU = nn.ReLU()
        return
    
    def forward(self,
        y: tuple
        ) -> torch.tensor:
        """Run critic neural network with x as input.

        Args:
            y (tuple): Tuple of state tensor and action tensor.

        Returns:
            torch.tensor: 1d output tensor.
        """
        x, a = y
        x = torch.cat((x, a), 1)
        # run computation
        for i in range(self.layers_len):
            if i == self.layers_len - 1:
                output = self.layers[i](x)
            else:
                x = self.ReLU(self.layers[i](x))
        return output


if __name__ == '__main__':
    #actor = Actor(3, [512, 512, 512])
    #actor.cuda()
    state_t = torch.tensor([[-0.5, 0.1, -1]]).type(torch.cuda.FloatTensor)
    action_t = torch.tensor([[2, -2]]).type(torch.cuda.FloatTensor)
    #print(input_t.size())
    #action = actor.forward(input_t)
    #print(action.size())
    #print(action)

    critic = Critic(3, 2, [64, 64, 64])
    critic.cuda()
    value = critic.forward((state_t, action_t))
    print(value.size())
    print(value)
