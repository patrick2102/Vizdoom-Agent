import os
import itertools as it
import cv2
import torch
import time
import torch.nn as nn
import torch.optim as optim
import numpy as np
import vizdoom as vzd
from agents import A2C
from deprecated import Models
from collections import deque
import random
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# Hyperparameters
num_episodes = 100
num_epochs = 100
rollout_size = 100
downscale = (45, 30)

device = "cpu"
if torch.cuda.is_available():
    device = "cuda:0"

def preprocess(state):
    state = cv2.resize(state, downscale, interpolation=cv2.INTER_AREA)

    # Display the resized state
    #cv2.imshow("Preprocessed State", state)
    #cv2.waitKey(1)  # Wait for 1 millisecond to update the window

    state = np.moveaxis(state, 1, 0)
    state = np.expand_dims(state, axis=0)
    state = np.array(state, dtype=float) / 255

    state = torch.tensor(state, dtype=torch.float32).to(device)

    return state

def create_game(config_file):
    game = vzd.DoomGame()
    game.load_config(config_file)
    game.init()
    num_actions = game.get_available_buttons_size()
    actions = [list(a) for a in it.product([0, 1], repeat=num_actions)]
    return game, actions

def train(agent, game):
    writer = SummaryWriter(log_dir=f"runs/{agent.name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    for epoch in range(num_epochs):
        total_reward = 0
        for episode in range(num_episodes):
            game.new_episode()
            average_time_episode = 0
            episode_count = 0

            while not game.is_episode_finished():
                episode_count += 1
                before = time.time()
                agent.make_action(game, train=True)
                average_time_episode += (time.time() - before)

            total_reward += game.get_total_reward()
            print('episode:', episode, 'epoch:', epoch, 'average action time: ', average_time_episode/episode_count)

        avg_reward = total_reward/num_episodes

        agent.train()

        avg_actor_loss = sum(agent.actor_losses)/len(agent.actor_losses)
        avg_critic_loss = sum(agent.critic_losses)/len(agent.critic_losses)
        agent.actor_losses = []
        agent.critic_losses = []
        agent.save()

        writer.add_scalar("Actor loss", avg_actor_loss, epoch)
        writer.add_scalar("Critic loss", avg_critic_loss, epoch)
        writer.add_scalar("Average Reward", avg_reward, epoch)

    game.close()

def main():
    config_file = "scenarios/basic.cfg"
    game, actions = create_game(config_file)
    agent = A2C('A2C3', device, actions)
    train(agent, game)


#Dictionary to handle command line arguments:

args = {
    "train": False,
    "use_gpu": False,
}

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
    cv2.destroyAllWindows()
