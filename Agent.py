import numpy as np
import torch


class HyperParameters:
    def __init__(self, criterion, optimizer, gamma=0.99):
        self.gamma = gamma
        self.criterion = criterion
        self.optimizer = optimizer
        self.entropy_weight = 0.01

class DQNAgent:
    def __init__(self, target_model, policy_model, model_name, hyper_parameters):
        self.frame_stack_size = 10
        self.memory_idx = 0
        self.exploration_decay = 0.9995
        self.target = target_model
        self.policy = policy_model

        self.model_name = model_name
        self.hp = hyper_parameters
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda:0"

    def train(self, minibatch, device):
        # Sample minibatch from the memory
        minibatch_size = len(minibatch)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        hp = self.hp

        states = torch.cat(states).to(device)
        states = states.unsqueeze(1)

        actions = torch.tensor(actions).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        next_states = torch.cat(next_states).to(device)
        #next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
        next_states = next_states.unsqueeze(1)

        current_q_values = self.policy(states).gather(1, actions.unsqueeze(1)).squeeze()
        next_q_values = self.target(next_states).max(1)[0].detach()
        target_q_values = rewards + (hp.gamma * next_q_values)

        loss = hp.criterion(current_q_values, target_q_values)
        hp.optimizer.zero_grad()
        loss.backward()
        hp.optimizer.step()

        return loss.item()

    def get_action(self, state):
        state = torch.unsqueeze(state, 0)
        return self.policy(state).argmax().item()

class ActorCriticAgent:
    def __init__(self, model, model_name, hyper_parameters):
        self.frame_stack_size = 10
        self.memory_idx = 0
        self.exploration_decay = 0.9995
        self.model = model
        self.model_name = model_name
        self.hp = hyper_parameters
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda:0"

    def train(self, minibatch, device):
        states, actions, rewards, next_states, dones = zip(*minibatch)
        dones = torch.tensor(dones, dtype=torch.float32).to(device)

        minibatch_size = len(minibatch)

        hp = self.hp
        states = torch.cat(states).to(device)
        states = states.unsqueeze(1)

        actions = torch.tensor(actions).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        next_states = torch.cat(next_states).to(device)
        next_states = next_states.unsqueeze(1)

        row = np.arange(minibatch_size)

        a = row, actions

        prob_dists, state_values = self.model.forward(states)
        with torch.no_grad():
            _, next_state_values = self.model.forward(next_states)

        target_q_values = rewards + (hp.gamma * next_state_values * (1 - dones))
        advantage = target_q_values - state_values

        prob_dists = prob_dists[a]

        #action_dists = prob_dists[torch.arange(minibatch_size), actions.long()]

        #action_dist = torch.distributions.Categorical(action_dists)
        #with torch.no_grad():

        with torch.no_grad():
            prob_dists += 1e-10

        log_probs = torch.log(prob_dists)


        actor_loss = -(log_probs * advantage).mean()
        critic_loss = hp.criterion(state_values, target_q_values).mean()

        loss = actor_loss + critic_loss

        hp.optimizer.zero_grad()
        loss.backward()
        hp.optimizer.step()

        return actor_loss.item(), critic_loss.item()

    def get_action(self, state):
        state = torch.unsqueeze(state, 0)
        return self.model.predict(state)


class ActorCriticAgentPPO:
    def __init__(self, model, model_name, hyper_parameters):
        self.frame_stack_size = 10
        self.memory_idx = 0
        self.exploration_decay = 0.9995
        self.model = model
        self.model_name = model_name
        self.hp = hyper_parameters
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda:0"

    def train(self, minibatch, device):
        states, actions, rewards, next_states, dones, old_prop_dist = zip(*minibatch)
        dones = torch.tensor(dones, dtype=torch.float32).to(device)

        minibatch_size = len(minibatch)

        hp = self.hp
        states = torch.cat(states).to(device)
        states = states.unsqueeze(1)

        actions = torch.tensor(actions).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        next_states = torch.cat(next_states).to(device)
        next_states = next_states.unsqueeze(1)
        #old_prop_dist = torch.tensor(old_prop_dist, dtype=torch.float32).to(device)

        row = np.arange(minibatch_size)

        a = row, actions

        prob_dists, state_values = self.model.forward(states)
        #
        if len(prob_dists.shape) == 1:
            prob_dists = prob_dists.unsqueeze(0)
        #print("prob_dists shape:", prob_dists.shape)

        with torch.no_grad():
            _, next_state_values = self.model.forward(next_states)

            target_q_values = rewards + (hp.gamma * next_state_values * (1 - dones))
            target_q_values = target_q_values.squeeze(0)
            advantage = target_q_values - state_values

            old_prop_dist = torch.tensor(old_prop_dist, dtype=torch.float32).to(device)
            old_prob_dist_logs = torch.log(old_prop_dist + 1e-10)

        log_probs = torch.log(prob_dists)
        entropy = -(prob_dists * log_probs).sum(dim=-1)
        prob_dists = prob_dists[a]
        log_probs = log_probs[a]

        eps = 0.2
        ratios = torch.exp(log_probs - old_prob_dist_logs)
        clipped_ratios = ratios.clamp(1 - eps, 1 + eps)
        actor_loss = -(torch.min(ratios, clipped_ratios) * advantage).mean()
        critic_loss = hp.criterion(state_values, target_q_values).mean()

        entropy = entropy.mean()

        loss = actor_loss + critic_loss - self.hp.entropy_weight * entropy

        hp.optimizer.zero_grad()
        loss.backward()
        hp.optimizer.step()

        old_prop_dist[row] = prob_dists.detach()
        return actor_loss.item(), critic_loss.item()

    def get_action(self, state):
        with torch.no_grad():
            prop_dist, state_value = self.model.forward(state)

        return prop_dist.argmax().item(), prop_dist.detach()


class ActorCriticAgentPPOLSTM:
    def __init__(self, model, model_name, hyper_parameters):
        self.frame_stack_size = 10
        self.memory_idx = 0
        self.exploration_decay = 0.9995
        self.model = model
        self.model_name = model_name
        self.hp = hyper_parameters
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda:0"

    def train(self, minibatch, device):
        states, actions, rewards, next_states, dones, old_prop_dist = zip(*minibatch)
        dones = torch.tensor(dones, dtype=torch.float32).to(device)

        minibatch_size = len(minibatch)

        hp = self.hp
        states = torch.cat(states).to(device)
        #states = states.unsqueeze(1)

        if len(states.shape) == 1:
            states = states.unsqueeze(0)

        actions = torch.tensor(actions).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        next_states = torch.cat(next_states).to(device)

        if len(next_states.shape) == 1:
            next_states = next_states.unsqueeze(0)

        #next_states = next_states.unsqueeze(1)
        #old_prop_dist = torch.tensor(old_prop_dist, dtype=torch.float32).to(device)

        row = np.arange(minibatch_size)

        a = row, actions

        prob_dists, state_values = self.model.forward(states)

        if len(prob_dists.shape) == 1:
            prob_dists = prob_dists.unsqueeze(0)



        #print("prob_dists shape:", prob_dists.shape)

        with torch.no_grad():
            _, next_state_values = self.model.forward(next_states)

            target_q_values = rewards + (hp.gamma * next_state_values * (1 - dones))
            target_q_values = target_q_values.squeeze(0)
            advantage = target_q_values - state_values

            old_prop_dist = torch.tensor(old_prop_dist, dtype=torch.float32).to(device)
            old_prob_dist_logs = torch.log(old_prop_dist + 1e-10)

        log_probs = torch.log(prob_dists)
        entropy = -(prob_dists * log_probs).sum(dim=-1)
        prob_dists = prob_dists[a]
        log_probs = log_probs[a]

        eps = 0.2
        ratios = torch.exp(log_probs - old_prob_dist_logs)
        clipped_ratios = ratios.clamp(1 - eps, 1 + eps)
        actor_loss = -(torch.min(ratios, clipped_ratios) * advantage).mean()
        critic_loss = hp.criterion(state_values, target_q_values).mean()

        entropy = entropy.mean()

        loss = actor_loss + critic_loss - self.hp.entropy_weight * entropy

        hp.optimizer.zero_grad()
        loss.backward()
        hp.optimizer.step()

        old_prop_dist[row] = prob_dists.detach()
        return actor_loss.item(), critic_loss.item()

    def get_action(self, state):
        with torch.no_grad():
            prop_dist, state_value = self.model.forward(state)

        return prop_dist.argmax().item(), prop_dist.detach()