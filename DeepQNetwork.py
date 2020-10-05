import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class DeepQNetwork(nn.Module):
    def __init__(self, learning_rate, n_actions, input_dims, checkpoint_dir):
        super(DeepQNetwork, self).__init__()

        self.conv1 = nn.Conv2d(*input_dims, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=2, stride=1)

        self.output_conv_layer_dims = self.get_conv_output_dims(input_dims)

        self.fc1 = nn.Linear(self.output_conv_layer_dims, 512)
        self.fc2 = nn.Linear(512, n_actions)

        self.optimizer = optim.RMSprop(self.parameters(), lr=learning_rate)
        self.loss = nn.MSELoss()

        self.device_type = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(self.device_type)
        self.to(self.device)

        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_name = os.path.join(checkpoint_dir, 'dqn')

    def get_conv_output_dims(self, input_dims):
        temp = torch.zeros(*input_dims, 1)
        dim = self.conv1(temp)
        dim = self.conv2(dim)
        dim = self.conv3(dim)

        return int(np.prod(dim.size()))

    def forward(self, data):
        conv_layer1 = F.relu(self.conv1(data))
        conv_layer2 = F.relu(self.conv2(conv_layer1))
        conv_layer3 = F.relu(self.conv3(conv_layer2))

        flattened_layer = conv_layer3.view(conv_layer3.size()[0], -1)

        fc_layer1 = F.relu(self.fc1(flattened_layer))
        actions = self.fc2(fc_layer1)

        return actions

    def save_checkpoint(self):
        print('Saving checkpoint ...')
        torch.save(self.state_dict(), self.checkpoint_name)

    def load_checkpoint(self):
        print('Loading checkpoint ...')
        self.load_state_dict(torch.load(self.checkpoint_name))


