import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gym

class ActorCritic(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, n_actions=2):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
            nn.Softmax(dim=1)
        )

        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        probs = self.actor(state)
        value = self.critic(state)

        probs = torch.squeeze(probs)
        value = torch.squeeze(value)

        return probs, value

def train(actor_critic, env, optimizer, gamma=0.99, max_episodes=1000):
    total_reward = 0
    for episode in range(max_episodes):
        state, _ = env.reset()
        done = False

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            probs, value = actor_critic(state_tensor)
            action_dist = torch.distributions.Categorical(probs)
            action = action_dist.sample()
            #action = env.action_space.sample()

            next_state, reward, done, _, _ = env.step(action.item())
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)

            if done:
                next_value = torch.zeros_like(value)
            else:
                _, next_value = actor_critic(next_state_tensor)

            advantage = reward + gamma * next_value - value
            actor_loss = -action_dist.log_prob(action) * advantage
            critic_loss = advantage.pow(2)
            loss = actor_loss + critic_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state = next_state
            total_reward += reward

        if (episode + 1) % 10 == 0:
            print(f'Episode {episode + 1}, Total Reward: {total_reward/10.0}')
            total_reward = 0

if __name__ == '__main__':
    env = gym.make('CartPole-v0')

    actor_critic = ActorCritic(env.observation_space.shape[0], 128, 2)
    optimizer = optim.Adam(actor_critic.parameters(), lr=1e-3)

    train(actor_critic, env, optimizer)
    env.close()
