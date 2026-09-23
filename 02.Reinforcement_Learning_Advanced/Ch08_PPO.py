import torch, torch.nn as nn, torch.nn.init as init, torch.optim as optim
import numpy as np, matplotlib.pyplot as plt, time, warnings, os, io, base64
import gymnasium as gym
import contextlib, torch.nn.functional as F
from torch.distributions import Normal
from torch.utils.data import DataLoader, TensorDataset, dataloader
from glob import glob
from PIL import Image
from IPython.display import HTML
warnings.filterwarnings("ignore")

np.bool8 = np.bool_

env = gym.make("MountainCarContinuous-v0")
obs_dim, act_dim = 2, 1
gamma, lamda = 0.99, 0.95 # 할인율(gamma), GAE용 람다(lamda)
entropy_coef, clip_epsilon = 0.003, 0.2
rollout_len, total_episodes, num_epochs, batch_size = 999, 2000, 32, 100
solved_reward, actor_lr, critic_lr = 90, 0.001, 0.001

def plot_train_history(scores):
    with plt.style.context('seaborn-v0_8-darkgrid'):
        plt.figure(figsize=(6, 4))
        plt.plot(scores, c="crimson")
        plt.xlabel('Episode')
        plt.title(f"Score {np.mean(scores[-10:])}")
        plt.tight_layout()
        plt.show()

class Memory:
    def __init__(self):
        self.states, self.actions, self.rewards, self.log_probs, self.values, self.dones = [], [], [], [], [], []

    def clear(self): #매 에피소드 끝날 때마다 메모리를 초기화
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.dones.clear()

class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 32)
        self.fc2 = nn.Linear(32, 32)
        self.mu_layer = nn.Linear(32, act_dim)
        self.sigma_layer = nn.Linear(32, act_dim)

        nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc2.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.mu_layer.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.sigma_layer.weight, gain=np.sqrt(2))
        nn.init.constant_(self.sigma_layer.bias, 0.0)
        nn.init.constant_(self.mu_layer.bias, 0.0)

    def forward(self, state):
        out = F.tanh(self.fc1(state))
        out = F.tanh(self.fc2(out))
        mu = torch.tanh(self.mu_layer(out))
        sigma = torch.exp(torch.tanh(self.sigma_layer(out))) # torch.tanh
        dist = Normal(mu, sigma) #정규분포 객체
        return dist

class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 32)
        self.fc2 = nn.Linear(32, 32)
        self.value_layer = nn.Linear(32, 1)

        nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc2.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.value_layer.weight, gain=np.sqrt(2))
        nn.init.constant_(self.value_layer.bias, 0.0)

    def forward(self, state_batch):
        out = torch.tanh(self.fc1(state_batch))
        out = torch.tanh(self.fc2(out))
        value = self.value_layer(out)
        return value

class PPO:
    def __init__(self):
        self.actor = Actor()
        self.critic = Critic()
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

    def select_action(self, state):

        dist = self.actor(state)
        action = dist.sample()
        return action, dist.log_prob(action)

    def update(self, memory, next_value):
        q_values, advantages = self._compute_qvalues_gae(memory, next_value)
        states = torch.cat(memory.states).view(-1, obs_dim)
        actions = torch.cat(memory.actions)
        old_log_probs = torch.cat(memory.log_probs).detach()

        for _ in range(num_epochs): #32
            dataset = TensorDataset(states, actions, q_values, old_log_probs, advantages)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            for s_batch, a_batch, q_batch, old_lp_batch, adv_batch in dataloader: # 10번 정도 돔
                dist = self.actor(s_batch)
                cur_log_probs = dist.log_prob(a_batch)

                #확률비 계산
                ratio = torch.exp(cur_log_probs - old_lp_batch)
                surr1 = ratio * adv_batch
                surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) *adv_batch

                #최소를 취함으로써 정책 업데이트 폭을 제한 (클리핑)
                actor_loss = -torch.mean(torch.min(surr1, surr2))
                critic_loss = F.mse_loss(self.critic(s_batch), q_batch)
                entropy = dist.entropy().mean()

                total_loss = actor_loss + 0.5 * critic_loss - entropy_coef * entropy

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.step()

    def _compute_qvalues_gae(self, memory, next_value):
        q_values, advantages, gae = [], [], 0

        memory.values.append(next_value)

        for i in reversed(range(len(memory.rewards))):
            delta = memory.rewards[i] + (1 - memory.dones[i].float()) * gamma * memory.values[i + 1] - memory.values[i]
            gae = delta + gamma * lamda * (1 - memory.dones[i].float()) * gae
            advantages.insert(0, gae)
            q_values.insert(0, gae + memory.values[i])

        memory.values.pop()  # 마지막에 추가한 0.0 제거
        return torch.cat(q_values).detach(), torch.cat(advantages).detach()


agent = PPO()
memory = Memory()
scores = []
t0 = time.time()

# main 학습 루프

for episode in range(total_episodes):
    state = env.reset()[0].reshape(1, -1)
    score = 0.0

    for _ in range(rollout_len):
        state_tensor = torch.FloatTensor(state)
        action, log_prob = agent.select_action(state_tensor)
        value = agent.critic(state_tensor)
        action_to_numpy = action.squeeze().detach().cpu().numpy().flatten()
        next_state, reward, terminated, truncated, info = env.step(action_to_numpy)
        done = terminated or truncated

        next_state = next_state.reshape(1, -1)

        memory.states.append(state_tensor)
        memory.actions.append(action)
        memory.values.append(value)
        memory.log_probs.append(log_prob)
        memory.rewards.append(torch.FloatTensor([[reward]]))
        memory.dones.append(torch.BoolTensor([[terminated]]))

        state = next_state
        score += reward
        if done:
            break

    if terminated:
        next_value = torch.zeros(1, 1)
    else:
        with torch.no_grad():
            next_value = agent.critic(torch.FloatTensor(state))

    agent.update(memory, next_value)
    memory.clear()
    scores.append(score)

    if episode % 5 == 0:
        print(f"Episode {episode} ended with Score: {score}")

    if len(scores) >= 20 and np.mean(scores[-20:]) >= solved_reward:
        plot_train_history(scores)
        print("#Solved_reward_done")
        break

env.close()
t1 = time.time()
print(f"Training Time: {t1 - t0} seconds")


#평가 (비디오 생략)

def evaluate(agent, num_episodes: int = 10, record: bool = False):

    env_eval = gym.make("MountainCarContinuous-v0")

    for episode in range(1, num_episodes + 1):
        state = env_eval.reset()
        state = np.reshape(state[0], (1, -1))
        done = False
        episode_reward = 0

        while not done:
            state_tensor = torch.FloatTensor(state)
            dist = agent.actor(state_tensor)
            mean = dist.mean.detach().cpu().numpy().flatten()
            action = mean
            # action = dist.sample().detach().cpu().numpy().flatten()

            next_state, reward, terminated, truncated, info = env_eval.step(action)
            done = terminated or truncated
            state = next_state
            episode_reward += reward

        print(f"Evaluation Episode {episode} ended with Reward: {episode_reward}")

    env_eval.close()

print("Evaluation completed.")
evaluate(agent)

    

        