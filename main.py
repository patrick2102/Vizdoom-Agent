import os
import sys
import itertools as it
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import vizdoom as vzd
import Agent
import Models
from collections import deque
import random
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from multiprocessing import Process, Queue, Lock, Manager
import copy
from Agent import HyperParameters, DQNAgent, ActorCriticAgent, ActorCriticAgentPPO, ActorCriticAgentPPOLSTM

# Hyperparameters
gamma = 0.99
num_episodes = 10000
replay_memory_size = 5000
rollout_size = 1000
minibatch_size = 32
#rollout_size = minibatch_size*5
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 0.9995
tics_per_action = 12
train_after_episodes = 100
training_epochs = 10
replay_memory = deque(maxlen=replay_memory_size)
downscale = (45, 30)

save_model_freq = 100

shutdown = False

def preprocess(state, device):
    state = cv2.resize(state, downscale, interpolation=cv2.INTER_AREA)

    # Display the resized state
    #cv2.imshow("Preprocessed State", state)
    #cv2.waitKey(1)  # Wait for 1 millisecond to update the window

    state = np.moveaxis(state, 1, 0)
    state = np.expand_dims(state, axis=0)
    state = np.array(state, dtype=float) / 255

    state = torch.tensor(state, dtype=torch.float32).to(device)

    return state

def get_DQNAgent(num_actions):
    game = vzd.DoomGame()
    # Initialize DQN and target networks

    model_name = "DQNConv"

    policy_net = Models.DQNConv(num_actions).float().to(device)
    target_net = Models.DQNConv(num_actions).float().to(device)

    # if model exists, load it
    if os.path.exists("models/"+model_name + ".pth"):
        policy_net.load_state_dict(torch.load("models/"+model_name + ".pth"), strict=False)
        target_net.load_state_dict(torch.load("models/"+model_name + ".pth"), strict=False)
        print("Loaded model")

    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)
    criterion = nn.MSELoss().to(device)
    # Initialize hyperparameters
    hyperParameters = HyperParameters(optimizer=optimizer, criterion=criterion)
    # Initialize agent
    agent = DQNAgent(target_net, policy_net, model_name, hyperParameters)
    return agent

def get_ActorCritic(num_actions):
    game = vzd.DoomGame()

    model_name = "ActorCritic"

    actorCriticModel = Models.ActorCriticModel(num_actions).float().to(device)

    # if model exists, load it
    if os.path.exists("models/"+model_name + ".pth"):
        actorCriticModel.load_state_dict(torch.load("models/"+model_name + ".pth"), strict=False)
        print("Loaded model")

    learning_rates = [{'params': actorCriticModel.conv1.parameters(), 'lr': 1e-4},
                      {'params': actorCriticModel.conv2.parameters(), 'lr': 1e-4},
                    {'params': actorCriticModel.conv3.parameters(), 'lr': 1e-4},
                      {'params': actorCriticModel.actor.parameters(), 'lr': 1e-4},
                      {'params': actorCriticModel.critic.parameters(), 'lr': 1e-4}]
    optimizer = optim.Adam(learning_rates)
    #optimizer = optim.Adam(lr=1e-3, params=actorCriticModel.parameters())

    criterion = nn.MSELoss().to(device)
    # Initialize hyperparameters
    hyperParameters = HyperParameters(optimizer=optimizer, criterion=criterion)
    # Initialize agent
    agent = ActorCriticAgent(actorCriticModel, "ActorCritic", hyperParameters)
    return agent

def get_ActorCriticPPO(num_actions):
    game = vzd.DoomGame()

    model_name = "ActorCriticPPO"

    actorCriticModel = Models.ActorCriticModel(num_actions).float().to(device)

    # if model exists, load it
    if os.path.exists("models/"+model_name + ".pth"):
        actorCriticModel.load_state_dict(torch.load("models/"+model_name + ".pth"), strict=False)
        print("Loaded model")

    learning_rates = [{'params': actorCriticModel.conv1.parameters(), 'lr': 1e-4},
                      {'params': actorCriticModel.conv2.parameters(), 'lr': 1e-4},
                      {'params': actorCriticModel.conv3.parameters(), 'lr': 1e-4},
                      {'params': actorCriticModel.actor.parameters(), 'lr': 1e-4},
                      {'params': actorCriticModel.critic.parameters(), 'lr': 1e-4}]
    optimizer = optim.Adam(learning_rates)
    #optimizer = optim.Adam(lr=1e-3, params=actorCriticModel.parameters())

    criterion = nn.MSELoss().to(device)
    # Initialize hyperparameters
    hyperParameters = HyperParameters(optimizer=optimizer, criterion=criterion)
    # Initialize agent
    agent = ActorCriticAgentPPO(actorCriticModel, model_name, hyperParameters)
    return agent

def get_ActorCriticPPOLSTM(num_actions, device):
    game = vzd.DoomGame()

    model_name = "ActorCriticPPOLSTM"

    actorCriticModel = Models.ActorCriticModelLSTM(num_actions, device).float().to(device)

    # if model exists, load it
    if os.path.exists("models/"+model_name + ".pth"):
        actorCriticModel.load_state_dict(torch.load("models/"+model_name + ".pth"), strict=False)
        print("Loaded model")

    learning_rates = [{'params': actorCriticModel.conv1.parameters(), 'lr': 1e-4},
                      {'params': actorCriticModel.conv2.parameters(), 'lr': 1e-4},
                      {'params': actorCriticModel.conv3.parameters(), 'lr': 1e-4},
                      {'params': actorCriticModel.lstm.parameters(), 'lr': 1e-4},
                      {'params': actorCriticModel.actor.parameters(), 'lr': 1e-4},
                      {'params': actorCriticModel.critic.parameters(), 'lr': 1e-4}]
    optimizer = optim.Adam(learning_rates)

    criterion = nn.MSELoss().to(device)
    # Initialize hyperparameters
    hyperParameters = HyperParameters(optimizer=optimizer, criterion=criterion)
    # Initialize agent
    agent = ActorCriticAgentPPOLSTM(actorCriticModel, model_name, hyperParameters)
    return agent



def TrainActorCritic():
    # Create VizDoom environment
    game = vzd.DoomGame()
    game.load_config("scenarios/basic.cfg")
    game.init()

    global_step = 0

    num_actions = game.get_available_buttons_size()
    actions = [list(a) for a in it.product([0, 1], repeat=num_actions)]

    epsilon = epsilon_start
    agent = get_ActorCritic(len(actions))
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f"runs/{agent.model_name}_{now}"
    writer = SummaryWriter(log_dir=log_dir)


    for episode in range(num_episodes):
        game.new_episode()
        state = preprocess(game.get_state().screen_buffer)
        done = False

        while not done:

            # Select action using epsilon-greedy strategy
            if np.random.rand() < epsilon:
                action_idx = random.randrange(len(actions))
            else:
                with torch.no_grad():
                    action_idx = agent.get_action(state)

            state = preprocess(game.get_state().screen_buffer)

            # Perform action and get next state
            #action = [0] * num_actions
            #action[action_idx] = 1
            action = actions[action_idx]


            reward = game.make_action(action, tics_per_action)
            done = game.is_episode_finished()

            if not done:
                next_state = preprocess(game.get_state().screen_buffer)
            else:
                next_state = torch.zeros(downscale, dtype=torch.float32).to(device)
                next_state = torch.unsqueeze(next_state, 0)

            # Save experience to replay memory
            replay_memory.append((state, action_idx, reward, next_state, done))
            #state = next_state

            # Train DQN using experience replay
        if len(replay_memory) >= minibatch_size:
            minibatch = random.sample(replay_memory, minibatch_size)
            actor_loss, critic_loss = agent.train(minibatch, device)
            writer.add_scalar("Actor Loss", actor_loss, global_step)
            writer.add_scalar("Critic Loss", critic_loss, global_step)
            global_step += 1

        if episode % save_model_freq == 0:
            torch.save(agent.model.state_dict(), f"models/{agent.model_name}.pth")

        # Decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        # Print episode results
        print(f"Episode: {episode}, Total Reward: {game.get_total_reward()}, Epsilon: {epsilon}")

        # Log rewards and epsilon to TensorBoard
        writer.add_scalar("Total Reward", game.get_total_reward(), episode)
        writer.add_scalar("Epsilon", epsilon, episode)

    game.close()
    writer.close()

def TrainActorCriticPPO():
    # Create VizDoom environment
    game = vzd.DoomGame()
    game.load_config("scenarios/basic.cfg")
    game.init()

    global_step = 0

    num_actions = game.get_available_buttons_size()
    actions = [list(a) for a in it.product([0, 1], repeat=num_actions)]

    epsilon = epsilon_start

    agent = get_ActorCriticPPO(len(actions))

    #writer = SummaryWriter()
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f"runs/{agent.model_name}_{now}"
    writer = SummaryWriter(log_dir=log_dir)

    rollout = []
    rollout_idx = 0
    #rollout = deque(maxlen=memory_size)
    steps = 0
    avg_reward = 0

    for episode in range(num_episodes):
        game.new_episode()
        state = preprocess(game.get_state().screen_buffer)
        done = False

        while not done:
            # Perform action and get next state
            action_idx, prop_dist = agent.get_action(state)
            action = actions[action_idx]
            reward = game.make_action(action, tics_per_action)
            done = game.is_episode_finished()

            if not done:
                next_state = preprocess(game.get_state().screen_buffer)
            else:
                next_state = torch.zeros(downscale, dtype=torch.float32).to(device)
                next_state = torch.unsqueeze(next_state, 0)

            # Save experience to replay memory
            prop_dist = prop_dist[action_idx]

            if rollout_idx < rollout_size:
                rollout.append((state, action_idx, reward, next_state, done, prop_dist))
            else:
                rollout[rollout_idx % rollout_size] = (state, action_idx, reward, next_state, done, prop_dist)

            rollout_idx += 1
            state = next_state

        if len(rollout) >= minibatch_size:
            al, cl = 0.0, 0.0
            for epoch in range(training_epochs):
                random.shuffle(rollout)
                for i in range(0, len(rollout), minibatch_size):
                    minibatch = rollout[i:i + minibatch_size]
                    actor_loss, critic_loss = agent.train(minibatch, device)
                    al += actor_loss
                    cl += critic_loss

            al /= (len(rollout)/minibatch_size) * training_epochs
            cl /= (len(rollout)/minibatch_size) * training_epochs
            global_step += 1
            writer.add_scalar("Actor Loss", al, global_step)
            writer.add_scalar("Critic Loss", cl, global_step)

            # Log rewards to TensorBoard
            avg_reward = 0
            run_steps = 0

            # Clear the rollout after training
            #rollout = []

        if episode % save_model_freq == 0:
            torch.save(agent.model.state_dict(), f"models/{agent.model_name}.pth")
            print("Saved model!")


        # Print episode results
        print(f"Episode: {episode}, Total Reward: {game.get_total_reward()}, Epsilon: {epsilon}")
        writer.add_scalar("Total Reward", game.get_total_reward(), episode)


    game.close()
    writer.close()

def update_local_model(local_model, shared_state_dict, lock):
    with lock:
        local_model.load_state_dict(shared_state_dict.copy())

def TrainActorCriticPPOLSTM():
    # Create VizDoom environment
    game = vzd.DoomGame()
    game.load_config("scenarios/basic.cfg")
    #game.init()

    num_actions = game.get_available_buttons_size()
    actions = [list(a) for a in it.product([0, 1], repeat=num_actions)]

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    agent = get_ActorCriticPPOLSTM(len(actions), device)

    #writer = SummaryWriter()
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f"runs/{agent.model_name}_{now}"
    writer = SummaryWriter(log_dir=log_dir)

    num_agents = 1
    shared_queue = Queue(maxsize=64)
    processes = []

    manager = Manager()
    shared_state_dict = manager.dict(agent.model.state_dict())
    update_lock = Lock()

    for i in range(num_agents):
        p = Process(target=RunAgent, args=(agent.model, shared_queue, shared_state_dict, update_lock, "cpu"))
        p.start()
        processes.append(p)

    rollout = []

    for episode in range(num_episodes):
        # Collect episodes until there is enough for a minibatch
        while len(rollout) < minibatch_size:
            rollout.append(shared_queue.get())

        # Train the model using the rollout
        al, cl = 0.0, 0.0
        for epoch in range(training_epochs):
            random.shuffle(rollout)
            for i in range(0, len(rollout), minibatch_size):
                minibatch = rollout[i:i + minibatch_size]
                actor_loss, critic_loss = agent.train(minibatch, device)
                al += actor_loss
                cl += critic_loss


        al /= training_epochs
        cl /= training_epochs

        average_reward = sum([element[2] for element in rollout])

        writer.add_scalar("Actor Loss", al, episode)
        writer.add_scalar("Critic Loss", cl, episode)
        writer.add_scalar("Average Reward", average_reward, episode)

        rollout = []

        if episode % save_model_freq == 0:
            torch.save(agent.model.state_dict(), f"models/{agent.model_name}.pth")
            print("Saved model!")

        #with update_lock:
        #    shared_state_dict.update(agent.model.state_dict())

    for p in processes:
        p.terminate()

    game.close()
    writer.close()

def main():
    # Create VizDoom environment
    game = vzd.DoomGame()
    game.load_config("scenarios/basic.cfg")
    game.init()

    global_step = 0

    num_actions = game.get_available_buttons_size()

    epsilon = epsilon_start

    agent = get_DQNAgent(num_actions)

    #writer = SummaryWriter()
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f"runs/{agent.model_name}_{now}"
    writer = SummaryWriter(log_dir=log_dir)

    for episode in range(num_episodes):
        game.new_episode()
        state = preprocess(game.get_state().screen_buffer)
        done = False

        while not done:

            # Select action using epsilon-greedy strategy
            if np.random.rand() < epsilon:
                action_idx = random.randrange(num_actions)
            else:
                with torch.no_grad():
                    action_idx = agent.get_action(state)

            # Perform action and get next state
            action = [0] * num_actions
            action[action_idx] = 1
            reward = game.make_action(action, tics_per_action)
            done = game.is_episode_finished()

            if not done:
                next_state = preprocess(game.get_state().screen_buffer)
            else:
                next_state = torch.zeros(downscale, dtype=torch.float32).to(device)
                next_state = torch.unsqueeze(next_state, 0)

            # Save experience to replay memory
            replay_memory.append((state, action_idx, reward, next_state, done))
            state = next_state

            # Train DQN using experience replay
        if len(replay_memory) >= minibatch_size:
            minibatch = random.sample(replay_memory, minibatch_size)
            loss = agent.train(minibatch, device)
            writer.add_scalar("Loss", loss, global_step)
            global_step += 1

        # Update the target network
        if episode % 10 == 0:
            agent.target.load_state_dict(agent.policy.state_dict())

        if episode % save_model_freq == 0:
            torch.save(agent.policy.state_dict(), f"models/{agent.model_name}.pth")

        # Decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        # Print episode results
        print(f"Episode: {episode}, Total Reward: {game.get_total_reward()}, Epsilon: {epsilon}")

        # Log rewards and epsilon to TensorBoard
        writer.add_scalar("Total Reward", game.get_total_reward(), episode)
        writer.add_scalar("Epsilon", epsilon, episode)

    game.close()
    writer.close()

def RunAgent(global_model, shared_queue, shared_state_dict, update_lock, device):
    game = vzd.DoomGame()
    game.load_config("scenarios/basic.cfg")
    game.init()
    num_actions = game.get_available_buttons_size()
    actions = [list(a) for a in it.product([0, 1], repeat=num_actions)]
    max_time_step = 4

    local_model = copy.deepcopy(global_model).to(device)

    #local_model = global_model.deepcopy().to(device)
    #local_model.to(device)
    local_model.device = device
    #update_local_model(local_model, shared_state_dict, update_lock)
    update_freq = 10
    episode = 0

    while not shutdown:
        game.new_episode()
        prev_frames = torch.zeros(1, max_time_step, 45, 30, dtype=torch.float32).cpu()
        frame = preprocess(game.get_state().screen_buffer, device)
        prev_frames[:, -1, :, :] = frame
        state = prev_frames.detach()
        done = False
        while not done:
            # Perform action and get next state
            with torch.no_grad():
                prop_dist, state_value = local_model.forward(state)
            action_idx = torch.argmax(prop_dist).item()
            action = actions[action_idx]
            reward = game.make_action(action, tics_per_action)
            done = game.is_episode_finished()

            if not done:
                frame = preprocess(game.get_state().screen_buffer, device)
            else:
                frame = torch.zeros(downscale, dtype=torch.float32).to(device)
                frame = torch.unsqueeze(frame, 0)

            # Move frames forward
            prev_frames = torch.roll(prev_frames, -1, 1)
            prev_frames[:, -1, :, :] = frame

            # add frame to prev_frames
            next_state = prev_frames.detach()

            # Save experience to replay memory
            prop_dist = prop_dist[action_idx]

            shared_queue.put((state, action_idx, reward, next_state, done, prop_dist))

            state = next_state
        episode += 1
        #update_local_model(local_model, shared_state_dict, update_lock)
        local_model = copy.deepcopy(global_model).to(device)
        local_model.device = device
        print("Model updated!")




#Dictionary to handle command line arguments:

args = {
    "train": False,
    "use_gpu": False,
}

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #TrainActorCritic()
    #main()
    #TrainActorCriticPPO()
    TrainActorCriticPPOLSTM()
    cv2.destroyAllWindows()
