import time, gym, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, torch.nn.init as init, matplotlib.pyplot as plt, random, os, io, base64, warnings
from collections import deque
from glob import glob
from IPython.display import HTML
from IPython.display import clear_output

warnings.filterwarnings('ignore') #불필요한 경고를 무시
np.bool8 = np.bool_

#GPU 사용 가능시,  GPU 를 사용하도록 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

gamma = 0.99
lr = 1e-4
tau = 0.001 #소프트 업데이트 비율
max_episodes = 1000 #최대 학습 에피소드 수
max_t = 1000 #최대 학습 스텝 수
batch_size = 64
memory_size = 10000
memory_warmup = batch_size * 3 # 메모리가 batch_size 의 3배이면 학습시작
solved_reward = 90

#환경
env = gym.make('MountainCarContinuous-v0')
state_dim = env.observation_space.shape[0]   #상태차원
act_dim = env.action_space.shape[0]             #액션차원
min_action = float(env.action_space.low[0])     #연속 액션의 최솟값
max_action = float(env.action_space.high[0])     #연속 액션의 최댓값

def plot_train_history(scores, actor_loss_history, critic_loss_history):
    # 학습 진행상황 (에피소드 별 보상, Actor/Critic Loss) 을 시각화

    data = [scores, actor_loss_history, critic_loss_history]

    labels = [
        f"Score {np.mean(scores[-10:])}",
        f"Actor Loss {np.mean(actor_loss_history[-10:])}",
        f"Critic Loss {np.mean(critic_loss_history[-10:])}",
    ]

    #주피터 노트북 상에서 이전 그래프를 지우고 새 그래프를 표시하기 위해 사용
    clear_output(True)

    with plt.style.context("seaborn-v0_8-dark-palette"):
        fig, axes = plt.subplots(3, 1, figsize=(6, 8))
        for i, ax in enumerate(axes):
            ax.plot(data[i], c = "crimson")
            ax.set_title(labels[i])
        plt.tight_layout()
        plt.show()

class Memory(object):

    def __init__(self, memory_size):
        self.memory = deque(maxlen=memory_size)

    def __len__(self):
        return len(self.memory)

    def append(self, item):
        self.memory.append(item)
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

def OUNoise(theta = 0.15, sigma = 0.3, mu = 0.0):
    # OU: 노이즈를 1차원으로 생성하기 위한 제너레이터 함수
    # theta: 평균 복원 항의 계수
    # sigma: 노이즈의 표준편차
    # mu: 노이즈 평균

    state = 0
    while True:
        dx = theta * (mu - state) + sigma * np.random.randn()
        state += dx
        yield state # step마다 노이즈를 하나씩 생성해 반환

# 정책 신경망 (Actor)
class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc_1 = nn.Linear(state_dim, 64)
        self.fc_2 = nn.Linear(64, 32)
        self.fc_out = nn.Linear(32, act_dim, bias = False)

        # 가중치 초기화
        init.xavier_normal_(self.fc_1.weight)
        init.xavier_normal_(self.fc_2.weight)
        init.xavier_normal_(self.fc_out.weight)

    def forward(self, state):
        x = F.elu(self.fc_1(state))
        x = F.elu(self.fc_2(x))
        x = self.fc_out(x)
        return torch.tanh(x) * max_action

# Q 신경망 (Critic)
class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc_state = nn.Linear(state_dim, 32)
        self.fc_action = nn.Linear(act_dim, 32)
        self.fc = nn.Linear(32 + 32, 32)
        self.fc_out = nn.Linear(32, act_dim, bias = False)

        # 가중치 초기화
        init.xavier_normal_(self.fc_state.weight)
        init.xavier_normal_(self.fc_action.weight)
        init.xavier_normal_(self.fc.weight)
        init.xavier_normal_(self.fc_out.weight)

    def forward(self, state, action):
        out_s = F.elu(self.fc_state(state))
        out_a = F.elu(self.fc_action(action))
        out_sa = torch.cat([out_s, out_a], dim=1)
        out = F.elu(self.fc(out_sa))
        return self.fc_out(out)

# DDPG Agent

class DDPG:
    def __init__(self):
        self.actor = Actor().to(device)
        self.target_actor = Actor().to(device)
        self.target_actor.load_state_dict(self.actor.state_dict())

        self.critic = Critic().to(device)
        self.target_critic = Critic().to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.memory = Memory(memory_size)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        #학습 과정 모니터링용 리스트
        self.scores = []
        self.actor_loss_history = []
        self.critic_loss_history = []

    def select_action(self, state, OU_noise):
        state = torch.FloatTensor(state).to(device) #텐서로 변환
        with torch.no_grad():
            action = self.actor(state).cpu().data.numpy()
        action += OU_noise # 탐험 노이즈 추가
        return np.clip(action, min_action, max_action) # 액션 범위 제한

    def update_critic(self, state, action, target):

        q_value = self.critic(state, action)
        loss = F.mse_loss(q_value, target)
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()
        return loss.item()

    def update_actor(self, state):
        action = self.actor(state)
        action = torch.clamp(action, min_action, max_action) 
        loss = -torch.mean(self.critic(state, action))
        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()
        return loss.item()

    def train(self, batch_size):
        batch = self.memory.sample(batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

        #1단계: Critic 업데이트
        with torch.no_grad():
            next_actions = self.target_actor(next_states)
            next_actions = torch.clamp(next_actions, min_action, max_action)
            target_Q = self.target_critic(next_states, next_actions)
            target_Q = rewards + (1 - dones) * gamma * target_Q

        critic_loss_value = self.update_critic(states, actions, target_Q)

        #2단계: Actor 업데이트
        actor_loss_value = self.update_actor(states)

        #3단계: 타겟 네트워크 소프트 업데이트
        self.soft_update(self.target_actor, self.actor, tau)
        self.soft_update(self.target_critic, self.critic, tau)

        #손실저장
        self.critic_loss_history.append(critic_loss_value)
        self.actor_loss_history.append(actor_loss_value)

    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

# Training Loop
ddpg = DDPG()
t0 = time.time() # 학습시간 측정 시작
scores = []

#학습 루프
for episode in range(max_episodes):
    state = env.reset()
    episode_score = 0.0
    noise = OUNoise()

    for t_step in range(max_t):
        action = ddpg.select_action(state, next(noise))
        next_state, reward, done, _ = env.step(action)
        ddpg.memory.append([state, action, reward, next_state, done])
        state = next_state
        episode_score += reward
        if len(ddpg.memory) >= memory_warmup:
            ddpg.train(batch_size)
        if done:
            break
    scores.append(episode_score)
    ddpg.scores = scores

    #콘솔 출력
    print(f"Episode {episode} ended with score: {episode_score:.2f}")

    #최근 10개 에피소드 평균이 solved_reward 초과 시 중단
    if len(scores) >= 10 and np.mean(scores[-10:]) >= solved_reward:
        plot_train_history(ddpg.scores, ddpg.actor_loss_history, ddpg.critic_loss_history)
        print('#최근 10개 에피소드의 평균 보상이 solved_reward를 초과하여 학습을 종료합니다.0')
        break

env.close()
t1 = time.time() # 학습시간 측정 종료
print("Training Time: {}min". format(round((t1-t0)/60, 2)))

#학습 완료후 에이전트 평가 및 비디오 녹화

def show_video(path: str):
    video_path = sorted(glob(os.path.join(path, "*.mp4")))[-1]
    video = io.open(video_path, 'rb').read()
    encoded = base64.b64encode(video)
    return HTML(data=f"""
    <video width="640" height="480" controls>
        <source src="data:video/mp4;base64,{encoded.decode()}" type="video/mp4">
    </video>
    """)

def evaluate(actor, num_episodes: int = 10, record: bool = False,
             video_dir: str = "MountainCarVideos/videos"):
    env = gym.make("MountainCarContinuous-v0")
    if record:
        env = gym.wrappers.RecordVideo(env, video_dir)

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0.0
        done = False
        while not done:
            with torch.no_grad():
                action = ddpg.select_action(state, OU_noise = 0.0)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            episode_reward += reward
        print(f"Episode {episode} ended with reward: {episode_reward:.2f}")

    env.close()
    if record:
        time.sleep(1)  # Ensure the video file is written before displaying

print('##After training, evaluating the agent')
dir = "MountainCarVideos/videos"
evaluate(ddpg.actor, num_episodes=10, record=True, video_dir=dir) # 동영상 재생 여부 확인 True or False

show_video(dir)