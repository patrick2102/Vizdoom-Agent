import torch
import torch.nn as nn
import torch.nn.functional as F

class DQNLinear(nn.Module):
    def __init__(self, num_actions):
        super(DQNLinear, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(30*45, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        return self.layers(x)

class A2CModel(nn.Module):
    def __init__(self, num_actions, c1=8, c2=32, c3=32, unroll_num=1):
        super(A2CModel, self).__init__()
        print("Running A2C")
        self.unroll_num = unroll_num

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