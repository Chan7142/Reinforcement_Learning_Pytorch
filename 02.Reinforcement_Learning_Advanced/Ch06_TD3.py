import time, gym, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, torch.nn.init as init, matplotlib.pyplot as plt, random, os, io, base64, warnings
from collections import deque
from glob import glob
from IPython.display import HTML
from IPython.display import clear_output, HTML; from collections import deque

warnings.filterwarnings('ignore') #불필요한 경고를 무시
np.bool8 = np.bool_

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lr = 0.001
tau = 0.005
gamma = 0.99
policy_noise = 0.2
noise_clip = 0.5
policy_delay = 2
max_t = 1000
memory_size = 2000
memory_warmup = 500
batch_size = 32
exploration_noise = 2.0 #초기 탐험 노이즈 크기
max_episodes = 2000
solved_reward = 90 #해결 판정 기준 보상

env = gym.make('MountainCarContinuous-v0')
state_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
min_action = float(env.action_space.low[0])
max_action = float(env.action_space.high[0])

def plot_train_history(scores, actor_loss_history, critic_loss_history):
    data = [scores, actor_loss_history, critic_loss_history]
    labels = [
        f"Score {np.mean(scores[-10:])}",
        f"Actor Loss {np.mean(actor_loss_history[-10:])}",
        f"Critic Loss {np.mean(critic_loss_history[-10:])}"
    ]
    clear_output(True)
    with plt.style.context("seaborn-v0_8-dark-palette"):
        fig, axes = plt.subplots(3, 1, figsize=(6, 8))
        for i, ax in enumerate(axes):
            ax.plot(data[i], c = "crimson")
            ax.set_title(labels[i])
        axes[0].set_xlabel("Episode")
        plt.tight_layout()
        plt.show()

class ReplayMemory:
    def __init__(self, memory_size):
        self.memory = deque(maxlen=memory_size)

    def __len__(self):
        return len(self.memory)

    def append(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc_out= nn.Linear(32, act_dim, bias = False)

        init.xavier_normal_(self.fc1.weight)
        self.fc1.weight.data.clamp_(-0.1, 0.1)
        init.xavier_normal_(self.fc2.weight)
        self.fc2.weight.data.clamp_(-0.1, 0.1)
        init.xavier_normal_(self.fc_out.weight)
        self.fc_out.weight.data.clamp_(-0.1, 0.1)

    def forward(self, state):
        x = F.elu(self.fc1(state))
        x = F.elu(self.fc2(x))
        x = self.fc_out(x)
        return torch.tanh(x) * max_action

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc_state = nn.Linear(state_dim, 32)
        self.fc_action = nn.Linear(act_dim, 32)
        self.fc = nn.Linear(32 + 32, 32)
        self.fc_out = nn.Linear(32, act_dim, bias=False)

        init.xavier_normal_(self.fc_state.weight)
        self.fc_state.weight.data.clamp_(-0.1, 0.1)
        init.xavier_normal_(self.fc_action.weight)
        self.fc_action.weight.data.clamp_(-0.1, 0.1)
        init.xavier_normal_(self.fc.weight)
        self.fc.weight.data.clamp_(-0.1, 0.1)
        init.xavier_normal_(self.fc_out.weight)
        self.fc_out.weight.data.clamp_(-0.1, 0.1)

    def forward(self, state, action):
        out_s = F.elu(self.fc_state(state))
        out_a = F.elu(self.fc_action(action))
        out_sa = torch.cat([out_s, out_a], dim=1)
        out = F.elu(self.fc(out_sa))
        out = self.fc_out(out)
        return out

class TD3:

    def __init__(self):
        self.actor = Actor().to(device)
        self.target_actor = Actor().to(device)
        self.target_actor.load_state_dict(self.actor.state_dict())

        self.critic1 = Critic().to(device)
        self.target_critic1 = Critic().to(device)
        self.target_critic1.load_state_dict(self.critic1.state_dict())

        self.critic2 = Critic().to(device)
        self.target_critic2 = Critic().to(device)
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        self.memory = ReplayMemory(memory_size)

        self.total_it = 0 #지연 업데이트를 위한 전체 스텝 카운트

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=lr)

        #학습 과정 모니터링용 리스트
        self.scores = []
        self.actor_loss_history = []
        self.critic1_loss_history = []
        self.critic2_loss_history = []

    def select_action(self, state, exploration_noise):

        state = torch.FloatTensor(state).to(device)
        with torch.no_grad():
            action = self.actor(state).cpu().data.numpy()


        #탐험 노이즈 추가
        
        action += np.random.normal(0, exploration_noise, size=action.shape)
        return np.clip(action, min_action, max_action)

    def update_critic(self, states, actions, target_Q):

        current_Q1 = self.critic1(states, actions)
        current_Q2 = self.critic2(states, actions)

        critic1_loss = F.mse_loss(current_Q1, target_Q.detach())
        critic2_loss = F.mse_loss(current_Q2, target_Q.detach())

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        return critic1_loss.item()

    def update_actor(self, states):

        actor_loss = -self.critic1(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return actor_loss.item()

    def train(self, batch_size):
        batch = self.memory.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

        with torch.no_grad():
            noise = torch.normal(0, policy_noise, size=actions.shape).to(device)
            noise = torch.clamp(noise, -noise_clip, noise_clip)
            next_actions = self.target_actor(next_states) + noise
            next_actions = torch.clamp(next_actions, min_action, max_action)

            target_Q1 = self.target_critic1(next_states, next_actions)
            target_Q2 = self.target_critic2(next_states, next_actions)
            target_Q = torch.min(target_Q1, target_Q2)                  #더 작은 Q선택
            target_Q = rewards + (1 - dones) * gamma * target_Q       #벨만 최적방정식 적용

        critic1_loss = self.update_critic(states, actions, target_Q)

        if self.total_it % policy_delay == 0:
            actor_loss = self.update_actor(states)
            self.soft_update(self.critic1, self.target_critic1)
            self.soft_update(self.critic2, self.target_critic2)
            self.soft_update(self.actor, self.target_actor)

            self.actor_loss_history.append(actor_loss)
            self.critic1_loss_history.append(critic1_loss)

        self.total_it += 1

    def soft_update(self, source, target):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


# 메인 학습 루프

td3 = TD3()
t0 = time.time()
scores = []

for episode in range(max_episodes):
    state = env.reset()
    episode_score = 0.0

    for t_step in range(max_t):
        action = td3.select_action(state, exploration_noise)

        next_state, reward, done, _ = env.step(action)

        td3.memory.append([state, action, reward, next_state, done])

        state = next_state
        episode_score += reward

        if len(td3.memory) > memory_warmup:
            td3.train(batch_size)

        if done:
            break

    exploration_noise = max(0.1, exploration_noise * 0.995) #노이즈 점진적 감소
    scores.append(episode_score)
    td3.scores = scores
    print(f"Episode {episode}, ended with score {episode_score:.2f}")

    if len(scores) >= 100 and np.mean(scores[-10:]) > solved_reward:
        plot_train_history(td3.scores, td3.actor_loss_history, td3.critic1_loss_history)
        print('# 최근 10개 에피소드 평균 보상이 solved_reward 초과로 학습 종료')
        break


#학습이 끝난후 평가
env.close()
t1 = time.time()
print(f"Training time: {t1 - t0:.2f} seconds")

#비디오 생략

def evaluate(agent, num_episodes: int = 5, record: bool = False):

    env = gym.make("MountainCarContinuous-v0")

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False

        while not done:
            with torch.no_grad():
                action = agent.select_action(state, exploration_noise=0.0) #학습된 Actor 로 행동 선택 노이즈 없음
            action = np.clip(action, min_action, max_action)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            episode_reward += reward

        print(f"Evaluation Episode {episode}, Reward: {episode_reward:.2f}")

    env.close()

print('## After training, evaluate the agent')
evaluate(td3, num_episodes = 10, record = False)
            