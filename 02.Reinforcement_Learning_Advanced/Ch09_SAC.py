import time, os, io, base64, warnings, numpy as np, torch, torch.nn as nn
import torch.nn.functional as F, torch.optim as optim, matplotlib.pyplot as plt
from glob import glob
from IPython.display import clear_output, HTML
from collections import deque
import gymnasium as gym
warnings.filterwarnings("ignore")
np.bool8 = np.bool_

env = gym.make("Pendulum-v1", render_mode = "rgb_array")

num_frames = 200_100
memory_size = 100_100
batch_size = 256
initial_random_steps = 10_000
target_reward = -150
policy_update_freq = 2
gamma = 0.99
tau = 0.01

def plot_train_history(scores):
    plt.figure(figsize=(6, 4))
    plt.plot(scores, c = "crimson")
    plt.title(f"Score (last 10 avg:{np.mean(scores[-10:]):.2f})")
    plt.xlabel
    plt.tight_layout()
    plt.show()

class ReplayBuffer:
    def __init__(self):
        self.buffur = deque([], maxlen=memory_size) #선입선출

    def store(self, state, act, reward, next_state, done):
        transition = (state, act, reward, next_state, done)
        self.buffur.append(transition)

    def sample_batch(self):
        indices = np.random.choice(len(self.buffur), batch_size, replace=False)
        samples = [self.buffur[idx] for idx in indices]
        batch = {
            'state':np.array([s[0] for s in samples]),
            'action':np.array([s[1] for s in samples]),
            'reward':np.array([s[2] for s in samples]),
            'next_state':np.array([s[3] for s in samples]),
            'done':np.array([s[4] for s in samples])
        }
        return batch
    def __len__(self):
        return len(self.buffur)

def orthogonal_init(layer):
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, gain = np.sqrt(2))
        nn.init.constant_(layer.bias, 0.)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, log_std_min=-20, log_std_max=2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh() )
        self.mu_layer = nn.Linear(32, action_dim)
        self.log_std_layer = nn.Linear(32, action_dim)

        self.log_std_min, self.log_std_max = log_std_min, log_std_max
        self.apply(orthogonal_init)

    def forward(self, state, deterministic = False):
        x = self.net(state)
        mu = self.mu_layer(x)
        log_std = self.log_std_layer(x).clamp(self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mu, std) #정규분포 생성
        z = mu if deterministic else dist.rsample()

        action = 2.0 * torch.tanh(z)
        log_prob = None
        if not deterministic:
            log_prob = dist.log_prob(z)
            log_prob += np.log(2.0) - torch.log(4.0 - action.pow(2) + 1e-7)
            log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob

class criticQ(nn.Module):
    def __init__(self, state_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim + act_dim, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 1))
        self.apply(orthogonal_init)

    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=-1))

class SACAgent:
    def __init__(self, state_dim, act_dim):
        self.actor = Actor(state_dim, act_dim)
        self.qf1 = criticQ(state_dim, act_dim)
        self.qf2 = criticQ(state_dim, act_dim)
        self.qf1_targ = criticQ(state_dim, act_dim)
        self.qf2_targ = criticQ(state_dim, act_dim)

        self.qf1_targ.load_state_dict(self.qf1.state_dict())
        self.qf2_targ.load_state_dict(self.qf2.state_dict())

        self.target_entropy = -float(act_dim)
        self.log_alpha = torch.zeros(1, requires_grad=True)

        self.actor_optim = optim.Adam(self.actor.parameters(), lr=1e-3)
        self.qf1_optim = optim.Adam(self.qf1.parameters(), lr=1e-3)
        self.qf2_optim = optim.Adam(self.qf2.parameters(), lr=1e-3)
        self.alpha_optim = optim.Adam([self.log_alpha], lr=1e-3)

        self.memory = ReplayBuffer()
        self.total_step = 0

    def select_action(self, state):
        state = torch.FloatTensor(state)
        action, _ = self.actor(state)
        return action.detach().cpu().numpy()

    def update(self):

        batch = self.memory.sample_batch()

        state = torch.FloatTensor(batch['state'])
        action = torch.FloatTensor(batch['action'])
        reward = torch.FloatTensor(batch['reward']).unsqueeze(-1)
        next_state = torch.FloatTensor(batch['next_state'])
        done = torch.FloatTensor(batch['done']).unsqueeze(-1)

        alpha = self.log_alpha.exp()

        with torch.no_grad():
            next_action, next_log_prob = self.actor(next_state)
            qf1_targ = self.qf1_targ(next_state, next_action)
            qf2_targ = self.qf2_targ(next_state, next_action)
            q_targ = torch.min(qf1_targ, qf2_targ) - alpha * next_log_prob
            bellman_backup = reward + (1 - done) * gamma * q_targ

        q1_pred = self.qf1(state, action)
        q2_pred = self.qf2(state, action)

        qf1_loss = F.mse_loss(q1_pred, bellman_backup)
        qf2_loss = F.mse_loss(q2_pred, bellman_backup)

        self.qf1_optim.zero_grad()
        qf1_loss.backward()
        self.qf1_optim.step()

        self.qf2_optim.zero_grad()
        qf2_loss.backward()
        self.qf2_optim.step()

        if self.total_step % policy_update_freq == 0:
            new_action, log_prob = self.actor(state)
            q1_new = self.qf1(state, new_action)
            q2_new = self.qf2(state, new_action)
            q_new = torch.min(q1_new, q2_new)
            actor_loss = (alpha.detach() * log_prob - q_new).mean()

            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self._target_soft_update(self.qf1, self.qf1_targ)
            self._target_soft_update(self.qf2, self.qf2_targ)

    def _target_soft_update(self, net, net_target):
        for p, p_target in zip(net.parameters(), net_target.parameters()):
            p_target.data.copy_(tau * p.data + (1 - tau) * p_target.data)

    def train(self, num_frames, target_reward = None):
        scores, score, episode = [], 0, 0
        state,_ = env.reset()

        for self.total_step in range(1, num_frames + 1):
            if self.total_step < initial_random_steps:
                    action = env.action_space.sample()
            else:
                    action = self.select_action(state)

            next_state, reward, terminated, truncated, _= env.step(action)
            done = terminated or truncated
            self.memory.store(state, action, reward, next_state, done)
            score += reward
            state = next_state

            if len(self.memory) > batch_size and self.total_step > initial_random_steps:
                    self.update()

            if done:
                scores.append(score)
                episode += 1
                score = 0
                state,_ = env.reset()

                if self.total_step % 1000 == 0 and len(scores) >= 50:
                    print(f"[Episode: {episode}]"
                            f"avg. of last 50 rewards = {np.mean(scores[-50:]):.2f}")
                if target_reward is not None and len(scores[-50:]) >= 50:
                    if np.mean(scores[-50:]) >= target_reward:
                        plot_train_history(scores)
                        print(f"Target reward {target_reward} reached")
                        break

#main

state_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

agent = SACAgent(state_dim, act_dim)

t0 = time.time()
agent.train(num_frames, target_reward = target_reward)
t1 = time.time()
print(f"Training time: {t1 - t0:.2f} seconds")

#evalute

def evaluate(agent, num_episodes: int = 10, record: bool = False):
    env_eval = gym.make("Pendulum-v1", render_mode = "rgb_array")

    for ep in range(1, num_episodes + 1):
        state,_ = env_eval.reset()
        done = False
        score = 0

        while not done:
            with torch.no_grad():
                action,_ = agent.actor(torch.FloatTensor(state), deterministic = True)
                next_state, reward, terminated, truncated, _ = env_eval.step(action.numpy())
                done = terminated or truncated
                score += reward
                state = next_state

            print(f"[Episode: {ep}] score = {score}")
        env_eval.close()

print('After training, evaluating agent')
evaluate(agent, num_episodes = 10)



