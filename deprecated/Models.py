import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import vizdoom as vzd
from collections import deque
import random
import torch.nn.functional as F

import main

# Hyperparameters
learning_rate = 0.001
gamma = 0.99
num_episodes = 1000
replay_memory_size = 10000
minibatch_size = 64
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 0.995

class DQNLin(nn.Module):
    def __init__(self, num_actions):
        super(DQNLin, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(30*45, 128), # Replace 3 * 30 * 45 with 76800
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

    def forward(self, x):
        #x = x.view(-1, x.size(0))
        x = x.reshape(x.size(0),-1)
        return self.layers(x)


class DQNConv(nn.Module):
    def __init__(self, num_actions, c1=8, c2=16, c3=24):
        super(DQNConv, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU()
        )

        self.img_size = c2*26*41

        self.layers = nn.Sequential(
            nn.Linear(self.img_size, 128), # Replace 3 * 30 * 45 with 76800
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

    def forward(self, x):
        #x = x.view(-1, x.size(0))
        x = self.conv1(x)
        x = self.conv2(x)

        x = x.view(-1, self.img_size)

        x = self.layers(x)

        return x


class ActorCriticModel(nn.Module):
    def __init__(self, num_actions, c1=32, c2=32, c3=32):
        super(ActorCriticModel, self).__init__()
        print("Running A2C")

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(c2, c3, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.ReLU()
        )

        self.img_size = c3*24*39

        self.actor = nn.Sequential(
            nn.Linear(self.img_size, 100),
            nn.ReLU(),
            nn.Linear(100, num_actions)
        )

        self.critic = nn.Sequential(
            nn.Linear(self.img_size, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(-1, self.img_size)

        actor_policy = self.actor(x)
        actor_policy = F.softmax(actor_policy, dim=1)

        critic_value = self.critic(x)

        actor_policy = torch.squeeze(actor_policy)
        critic_value = torch.squeeze(critic_value)

        return actor_policy, critic_value

    def predict(self, x):
        x, _ = self.forward(x)
        x = torch.squeeze(x)
        return torch.argmax(x)


class ActorCriticModelLSTM(nn.Module):
    def __init__(self, num_actions, c1=16, c2=24, c3=32, hidden_size=128):
        super(ActorCriticModelLSTM, self).__init__()
        print("Running A2C")

        self.hidden_size = hidden_size

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(c2, c3, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.ReLU()
        )

        self.img_size = c3*24*39

        self.lstm = nn.LSTM(self.img_size, self.hidden_size, batch_first=True, num_layers=1)

        self.actor = nn.Sequential(
            nn.Linear(self.hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )

        self.critic = nn.Sequential(
            nn.Linear(self.hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        # x shape: (batch_size, timesteps, height, width)
        batch_size, timesteps, height, width = x.size()

        input_channels = 1

        # Reshape the input tensor to feed it to the convolutional layers
        x = x.view(batch_size * timesteps, input_channels, height, width)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(batch_size, timesteps, -1)

        h0 = torch.zeros(1, batch_size, self.hidden_size).to(main.device)
        c0 = torch.zeros(1, batch_size, self.hidden_size).to(main.device)

        # Pass x through the LSTM layer
        lstm_out, (h_n, c_n) = self.lstm(x, (h0, c0))

        lstm_last_out = lstm_out[:, -1, :]

        # Use the last LSTM output for actor and critic layers
        #lstm_last_out = lstm_out[:, -1, :]

        actor_policy = self.actor(lstm_last_out)
        actor_policy = F.softmax(actor_policy, dim=1)

        critic_value = self.critic(lstm_last_out)

        actor_policy = torch.squeeze(actor_policy)
        critic_value = torch.squeeze(critic_value)

        return actor_policy, critic_value

    def predict(self, x):
        x, _ = self.forward(x)
        x = torch.squeeze(x)
        return torch.argmax(x)