# !pip install --upgrade gym # 0.26.2

import torch, torch.nn as nn, torch.nn.init as init, torch.optim as optim
import numpy as np, gym, matplotlib.pyplot as plt, time, warnings, os, io, base64
import contextlib, torch.nn.functional as F
from torch.distributions import Normal
from torch.utils.data import TensorDataset, DataLoader
from glob import glob
from PIL import Image
from IPython.display import HTML
warnings.filterwarnings('ignore')

# gym(0.25.2)는 np.bool8을 사용. numpy(2.0.2)는 np.bool8 대신 np.bool_를 사용
np.bool8 = np.bool_


# --------------------------------Configuration--------------------------------
env = gym.make("MountainCarContinuous-v0")
obs_dim, act_dim = 2, 1
gamma, lamda = 0.99, 0.95 # 할인율(gamma), GAE용 람다(lamda)
entropy_coef, clip_epsilon = 0.003, 0.2
rollout_len, total_episodes, num_epochs, batch_size = 999, 2000, 8, 100
solved_reward, actor_lr, critic_lr = 90, 1e-3, 5e-3


# ====================
# Plotting Function
# ====================
def plot_train_history(scores):
    with plt.style.context("seaborn-v0_8-dark-palette"):
        plt.figure(figsize=(6, 4))
        plt.plot(scores, c="crimson")
        plt.title(f"Score {np.mean(scores[-10:])}")
        plt.xlabel('Episode')
        plt.tight_layout()
        plt.show()


# ====================
# Memory
# ====================
class Memory:
    def __init__(self):
        self.states, self.actions, self.rewards, self.log_probs, self.values = [], [], [], [], []

    def clear(self): # 매 에피소드가 끝날 때마다 메모리를 초기화
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()


# =========================================
# Actor and Critic(V 함수) 신경망
# =========================================
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
        nn.init.constant_(self.mu_layer.bias, 0.0)
        nn.init.constant_(self.sigma_layer.bias, 0.0)

    def forward(self, state):
        out = F.tanh(self.fc1(state))
        out = F.tanh(self.fc2(out))
        mu = torch.tanh(self.mu_layer(out))
        sigma = torch.exp(torch.tanh(self.sigma_layer(out)))
        dist = Normal(mu, sigma) # Normal(정규) 분포 객체
        return dist


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3_value = nn.Linear(32, 1)

        nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc2.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc3_value.weight, gain=np.sqrt(2))
        nn.init.constant_(self.fc3_value.bias, 0.0)

    def forward(self, states_batch):
        out = torch.tanh(self.fc1(states_batch))
        out = torch.tanh(self.fc2(out))
        states_values = self.fc3_value(out)
        return states_values


# ====================
# PPO Agent
# ====================
class PPO:
    def __init__(self):
        self.actor = Actor()
        self.critic = Critic()
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

    def select_action(self, state):
        """ 주어진 단일 상태(1 x obs_dim)에서 액션과 로그 확률(log_prob)을 반환 """
        dist = self.actor(state)
        action = dist.sample() # 확률적 정책이므로 분포에서 샘플링
        return action, dist.log_prob(action)

    def update(self, memory):
        q_values, advantages = self._compute_qvalues_gae(memory)
        states = torch.cat(memory.states).view(-1, obs_dim)
        actions = torch.cat(memory.actions)
        old_log_probs = torch.cat(memory.log_probs).detach()

        for _ in range(num_epochs):
            # dataset을 구성하여 DataLoader로 미니배치 학습 진행
            dataset = TensorDataset(states, actions, q_values, old_log_probs, advantages)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            for s_batch, a_batch, q_batch, old_lp_batch, adv_batch in dataloader:
                dist = self.actor(s_batch)
                cur_log_prob = dist.log_prob(a_batch)

                # PPO objective의 ratio(확률비) 계산
                ratio = torch.exp(cur_log_prob - old_lp_batch)
                surr1 = ratio * adv_batch
                surr2 = torch.clamp(
                    ratio,
                    1 - clip_epsilon,
                    1 + clip_epsilon
                ) * adv_batch

                # 최소를 취함으로써 정책 업데이트 폭을 제한(클리핑)
                actor_loss = -torch.mean(torch.min(surr1, surr2))
                critic_loss = F.mse_loss(self.critic(s_batch), q_batch)
                entropy = dist.entropy().mean()

                total_loss = actor_loss + 0.5 * critic_loss - entropy_coef * entropy

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.step()

    def _compute_qvalues_gae(self, memory):
        q_values, advantages, gae = [], [], 0

        # 점화식 정의를 위하여 마지막 스텝 이후(T+1)의 상태 가치 0을 append
        memory.values.append(torch.tensor([[0.0]]))

        for i in reversed(range(len(memory.rewards))):
            delta = memory.rewards[i] + gamma * memory.values[i + 1] - memory.values[i]
            gae = delta + gamma * lamda * gae
            advantages.insert(0, gae)
            q_values.insert(0, gae + memory.values[i])

        # 추가했던 마지막 스텝 이후(T+1)의 상태 가치 0을 제거
        memory.values.pop()
        return torch.cat(q_values).detach(), torch.cat(advantages).detach()


# ====================
# Main
# ====================
agent = PPO()
memory = Memory()
scores = []
t0 = time.time()

for episode in range(total_episodes):
    state = env.reset()[0].reshape(1, -1)
    score = 0.0

    for _ in range(rollout_len):
        state_tensor = torch.FloatTensor(state)
        action, log_prob = agent.select_action(state_tensor)
        value = agent.critic(state_tensor)
        action_to_numpy = action.squeeze(0).detach().cpu().numpy()
        next_state, reward, terminated, truncated, info = env.step(action_to_numpy)
        done = terminated or truncated

        next_state = next_state.reshape(1, -1)

        memory.states.append(state_tensor)
        memory.actions.append(action)
        memory.log_probs.append(log_prob)
        memory.values.append(value)
        memory.rewards.append(torch.FloatTensor([[reward]]))

        state = next_state
        score += reward
        if done:
            break

    agent.update(memory)
    memory.clear()

    scores.append(score)

    if episode % 5 == 0:
        print(f"Episode {episode} ended with score {score:.2f}")

    # If average score over last 20 episodes is solved_reward, finish
    if len(scores) >= 20 and np.mean(scores[-20:]) > solved_reward:
        plot_train_history(scores)
        print("#Solved_reward done")
        break

env.close()
t1 = time.time()
print(f"Training time: {round((t1 - t0)/60, 2)} min")


# ============================
# Evaluate & video
# ============================
def show_video(path: str):
    video_path = sorted(glob(os.path.join(path, "*.mp4")))[-1]
    video = io.open(video_path, 'rb').read()
    encoded = base64.b64encode(video)
    return HTML(data=f"""
<video controls>
<source src="data:video/mp4;base64,{encoded.decode()}" type="video/mp4" />
</video>
""")


def evaluate(agent, num_episodes: int = 10, record: bool = False, video_dir: str = "videos"):
    env_eval = gym.make("MountainCarContinuous-v0", render_mode="rgb_array")

    if record:
        env_eval = gym.wrappers.RecordVideo(
            env_eval,
            video_dir,
            episode_trigger=lambda episode_id: True
        )

    for episode in range(1, num_episodes + 1):
        state = env_eval.reset()
        state = np.reshape(state[0], (1, -1))
        done = False
        episode_reward = 0

        while not done:
            state_tensor = torch.FloatTensor(state)
            dist = agent.actor(state_tensor)
            action = dist.sample().detach().cpu().numpy().flatten()[0]

            # Redirect output during the step where video writing happens(Moviepy 메시지 출력안함)
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                next_state, reward, terminated, truncated, info = env_eval.step([action])

            done = terminated or truncated
            state = np.reshape(next_state, (1, -1))
            episode_reward += reward

        print(f"Episode {episode}: Reward = {episode_reward:.2f}")

    env_eval.close()

    if record:
        time.sleep(1)


print("## After training, evaluate the agent")
dir = "MountainCarContinuous-v0/videos"
evaluate(agent, record=False, video_dir=dir)
# show_video(dir)