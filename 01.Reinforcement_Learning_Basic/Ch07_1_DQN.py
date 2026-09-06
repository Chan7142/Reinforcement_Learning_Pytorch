import gym, math, random, matplotlib, numpy as np
import matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F
from collections import namedtuple, deque
from itertools import count
from itertools import groupby
from IPython import display

is_ipython = 'inline' in matplotlib.get_backend()

def plot_durations(show_result = False, save = False):
    fig = plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype= torch.float)
    if show_result:
        plt.title('dqn_training')
    else:
        plt.clf()
        plt.title('Training...')

    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())

    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())
    plt.pause(0.001) #
    if save:
        fig.savefig('dqn_training_plot.png', dpi=300, bbox_inches = 'tight')
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait = True)
        else:
            display.display(plt.gcf())

#상수
batch_size = 64
gamma = 0.99 #할인율
eps_start, eps_end, eps_decay = 0.9, 0.05, 1000 # 앱실론 정책
tau = 0.005 # 신경망 복사율
LR = 1e-4 #학습률

env = gym.make("CartPole-v0")
n_actions = env.action_space.n
state = env.reset()
n_observations = len(state)
#신경망

class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

q_net = DQN()
target_net = DQN()
target_net.load_state_dict(q_net.state_dict())

# 앱실론 정책
steps_done =0
def select_action(state: np.ndarray)->int:
    global steps_done
    state = torch.FloatTensor(state).unsqueeze(0)
    eps_threshold = eps_end + (eps_start - eps_end) * math.exp(-1. *steps_done / eps_decay)

    steps_done += 1
    random01 = random.random()

    if random01 <= eps_threshold:
        return env.action_space.sample()
    else:
        with torch.no_grad():
            return q_net(state).max(1)[1].item()

state = env.reset()
torch.FloatTensor(state)
torch.FloatTensor(state).unsqueeze(0)

state = env.reset()
state = torch.FloatTensor(state).unsqueeze(0)
q_net(state)
q_net(state).shape
q_net(state).max(1)

q_net(state).max(1)[1]
q_net(state).max(1)[1].item()

class Memory():
    def __init__(self, capacity):
        self.deque = deque([], maxlen=capacity)
    def add(self,
            state:np.ndarray,
            action:int,
            reward:float,
            next_state:np.ndarray,
            done:bool):
        self.deque.append((state, action, reward, next_state, done))
    def sampling(self, batch_size)->list:
        return random.sample(self.deque, batch_size)

    def __len__(self):
        return len(self.deque)
memory = Memory(1000)

#최적화

optimizer = optim.Adam(q_net.parameters(), lr = LR)
Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'done'))
def optimize_model():
    transitions = memory.sampling(batch_size)
    batch = Transition(*zip(*transitions))

    batch_state = torch.FloatTensor(np.vstack(batch.state))
    batch_action = torch.LongTensor(batch.action).unsqueeze(1)
    batch_reward = torch.FloatTensor(batch.reward)
    batch_next_state = torch.FloatTensor(np.vstack(batch.next_state))
    batch_done = torch.LongTensor(batch.done)

    state_action_values = q_net(batch_state).gather(1, batch_action)
    with torch.no_grad():
        next_state_action_values = target_net(batch_next_state).max(1)[0]

    expected_state_action_values = batch_reward + gamma * (1 - batch_done)*next_state_action_values
    loss = (state_action_values - expected_state_action_values.unsqueeze(1)).pow(2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 10번 연속해서 만점을 받으면 학습 종료할 때 사용할 함수
def consecutive_length(lst):
    return [len(list(group)) for _, group in groupby(enumerate(lst),
                                                     lambda i_x:i_x[0]-i_x[1])]

#학습
episode_durations = []
movement199 = []
num_episodes = 5000
for i_episode in range(num_episodes):
    state = env.reset()
    for num_movement in count():
        action = select_action(state)
        next_state, reward, done, _=env.step(action)

        memory.add(state, action, reward, next_state, done)
        state = next_state
        if len(memory) >= batch_size:
            optimize_model()

        target_net_state_dict = target_net.state_dict()
        q_net_state_dict = q_net.state_dict()
        for key in q_net_state_dict:
            target_net_state_dict[key] = q_net_state_dict[key]*tau + target_net_state_dict[key] * (1-tau)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_durations.append(num_movement + 1)
            plot_durations()
            movement_end = num_movement
            break
    if movement_end == 199:
        movement199.append(i_episode)
        if max(consecutive_length(movement199)) >= 30:
            break

print(f'Complete: movement_end = {movement_end}')
plot_durations(show_result=True, save=True)
plt.show()

from tqdm import tqdm
q_net.eval()
with torch.no_grad():
    for i_episode in tqdm(range(5000)):
        state = env.reset()
        state = torch.FloatTensor(state).unsqueeze(0)

        done=False
        t = 0
        while not done:
            t+=1
            action = q_net(state).max(1)[1].item()
            next_state, _, done, _=env.step(action)
            next_state = torch.FloatTensor(next_state).unsqueeze(0)
            state = next_state
            if done:
                if t!=200:
                    print(t)