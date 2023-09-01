import os
import time

import torch

import models
import torch.optim as optim
import torch.nn as nn
import cv2
import numpy as np

import vizdoom as vzd

class Agent:
    def __init__(self, model, model_name, hyper_parameters, device):
        self.model = model
        self.model_name = model_name
        self.hyper_parameters = hyper_parameters
        self.device = device

    def init_model(self):
        pass

    def get_action(self, state):
        pass

    def train(self, state, action, reward, next_state, done):
        pass

    def save(self):
        pass

    def load(self):
        pass

class A2C:
    def __init__(self, name, device, actions):
        self.name = name
        self.device = device
        self.actions = actions
        self.num_actions = len(self.actions)
        self.learning_rate = 1e-4
        self.discount_factor = 0.9
        self.tics_per_action = 12
        self.downscale = (45, 30)
        self.replay_bool = False
        self.memory_max_size = 10000
        self.memory_min_size = 64
        self.memory_size = 0
        self.memory_idx = 0
        self.actor_losses = []
        self.critic_losses = []
        self.model = models.A2CModel(self.num_actions).to(self.device)
        self.init_model()
        self.memory = self.init_memory()

    def init_memory(self):
        self.memory_idx = 0
        self.memory_size = 0
        self.replay_bool = False
        memory = {
            'states': torch.zeros((self.memory_max_size, 1, 45, 30), dtype=torch.float32),
            'actions': torch.zeros((self.memory_max_size,), dtype=torch.int64),
            'rewards': torch.zeros((self.memory_max_size,), dtype=torch.float32),
            'next_states': torch.zeros((self.memory_max_size, 1, 45, 30), dtype=torch.float32),
            'dones': torch.zeros((self.memory_max_size,), dtype=torch.int)
        }
        return memory

    def init_model(self):
        # if model exists, load it
        if os.path.exists("models/" + self.name + ".pth"):
            self.model.load_state_dict(torch.load("models/" + self.name + ".pth"), strict=False)
            print("Loaded model")

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.criterion = nn.MSELoss().to(self.device)

    def preprocess(self, state):
        state = cv2.resize(state, self.downscale, interpolation=cv2.INTER_AREA)

        # Display the resized state
        state = np.moveaxis(state, 1, 0)
        state = np.expand_dims(state, axis=0)
        state = np.array(state, dtype=float) / 255

        state = torch.tensor(state, dtype=torch.float32)

        return state

    def make_action(self, game: vzd.DoomGame, train=False):

        if train:
            action = self.add_to_memory(game)
            if self.replay_bool:
                self.train()
        else:
            with torch.no_grad():
                state = self.preprocess(game.get_state().screen_buffer).to(self.device)
                action = self.model.forward(state).argmax().item()

        return action

    def add_to_memory(self, game):
        state = self.preprocess(game.get_state().screen_buffer).to(self.device)
        state = torch.unsqueeze(state, 0)
        action_idx, _ = self.model.forward(state)
        action_idx = action_idx.argmax().item()
        action = self.actions[action_idx]
        reward = torch.tensor(game.make_action(action, self.tics_per_action), dtype=torch.float32)
        if not game.is_episode_finished():
            next_state = self.preprocess(game.get_state().screen_buffer)
        else:
            next_state = torch.tensor(np.zeros((1, 45, 30)), dtype=torch.float32)
        next_state = torch.unsqueeze(next_state, 0)
        done = torch.tensor(int(game.is_episode_finished()), dtype=torch.int)

        self.memory['states'][self.memory_idx] = state
        self.memory['actions'][self.memory_idx] = action_idx
        self.memory['rewards'][self.memory_idx] = reward
        self.memory['next_states'][self.memory_idx] = next_state
        self.memory['dones'][self.memory_idx] = done

        self.memory_size += 1
        if self.memory_max_size < self.memory_size:
            self.memory_size = self.memory_max_size

        self.memory_idx = (self.memory_idx + 1) % self.memory_max_size
        if self.memory_idx > self.memory_min_size:
            self.replay_bool = True
        return action

    def train(self, iter=10):
        loss = 0

        for i in range(1):
            minibatch = torch.randperm(self.memory_size)[:self.memory_min_size]
            states = self.memory['states'][minibatch].to(self.device)
            actions = self.memory['actions'][minibatch].to(self.device)
            rewards = self.memory['rewards'][minibatch].to(self.device)
            next_states = self.memory['next_states'][minibatch].to(self.device)
            dones = self.memory['dones'][minibatch].to(self.device)

            #Invert dones
            not_dones = dones * -1 + 1

            prob_dists, state_values = self.model.forward(states)
            prob_dists = prob_dists.gather(torch.tensor(1, dtype=torch.int64), actions.unsqueeze(1)).squeeze()

            with torch.no_grad():
                _, next_state_values = self.model(next_states)

            target_q_values = rewards + self.discount_factor * next_state_values * not_dones

            advantage = target_q_values - state_values

            with torch.no_grad():
                prob_dists += 1e-10  # To avoid exploding gradients. If we take the log of something with 0 or very close to 0, then

            log_probs = torch.log(prob_dists)

            actor_loss = -(log_probs * advantage.detach())
            critic_loss = advantage.pow(2)
            loss = (actor_loss + critic_loss).mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.actor_losses.append(actor_loss.mean().item())
            self.critic_losses.append(critic_loss.mean().item())

        self.model.unroll_num += 1

        return loss

    def save(self):
        torch.save(self.model.state_dict(), "models/" + self.name + ".pth")






