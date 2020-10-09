import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Actor(nn.Module):
    """
    Defines a neural network for the actor (that derives the actions)
    """
    def __init__(self, input_dims, n_actions, learning_rate, checkpoint_dir, name):
        super(Actor, self).__init__()

        self.input = input_dims
        self.fc1 = nn.Linear(2*input_dims, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, n_actions)

        self.optimizer = optim.RMSprop(self.parameters(), lr=learning_rate)
        self.loss = nn.MSELoss()

        self.device_type = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(self.device_type)
        self.to(self.device)

        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_name = os.path.join(checkpoint_dir, name)

    def forward(self, data):
        fc_layer1 = F.relu(self.fc1(data))
        fc_layer2 = F.relu(self.fc2(fc_layer1))
        actions = self.fc3(fc_layer2)

        return actions

    def save_checkpoint(self):
        print('Saving checkpoint ...')
        torch.save(self.state_dict(), self.checkpoint_name)

    def load_checkpoint(self):
        print('Loading checkpoint ...')
        self.load_state_dict(torch.load(self.checkpoint_name))


class Critic(nn.Module):
    """
    Defines a neural network for the critic (that derives the value)
    """
    def __init__(self, input_dims, n_actions, learning_rate, checkpoint_dir, name):
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(2*input_dims + n_actions, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)

        self.optimizer = optim.RMSprop(self.parameters(), lr=learning_rate)
        self.loss = nn.MSELoss()

        self.device_type = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(self.device_type)
        self.to(self.device)

        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_name = os.path.join(checkpoint_dir, name)

    def forward(self, data1, data2):
        fc_layer1 = F.relu(self.fc1(torch.cat((data1, data2), 1)))
        fc_layer2 = F.relu(self.fc2(fc_layer1))
        value = self.fc3(fc_layer2)

        return value

    def save_checkpoint(self):
        print('Saving checkpoint ...')
        torch.save(self.state_dict(), self.checkpoint_name)

    def load_checkpoint(self):
        print('Loading checkpoint ...')
        self.load_state_dict(torch.load(self.checkpoint_name))
