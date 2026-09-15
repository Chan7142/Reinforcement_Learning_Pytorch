import statistics, torch, torch.nn as nn, torch.optim as optim, warnings, numpy as np
import torch.nn.functional as F, torch.distributions as distributions, matplotlib.pyplot as plt
import gym, time; from itertools import groupby; from tqdm import tqdm
warnings.filterwarnings('ignore')

env = gym.make("CartPole-v0")
np.bool8 = np.bool_
gamma = 0.99

# 정책 신경망
class Policy_net(nn.Module):
    def __init__(self):
        super(). __init__()
        self.input_layer = nn.Linear(4, 32)
        self.output_layer = nn.Linear(32, 2)
    def forward(self, state):
        x = F.selu(self.input_layer(state))
        action_logits = self.output_layer(x)
        action_probs = F.softmax(action_logits, dim = -1)
        return action_probs

policy_net = Policy_net()

# 상태가치 신경망
class Value_net(nn.Module):
    def __init__(self):
        super(). __init__()
        self.input_layer = nn.Linear(4, 32)
        self.output_layer = nn.Linear(32, 1)
    def forward(self, x):
        x = F.selu(self.input_layer(x))
        state_value = self.output_layer(x)
        return state_value
value_net = Value_net()

#정책함수

def select_action(state):
    state = torch.FloatTensor(state).unsqueeze(0)
    action_probs = policy_net(state)
    dist = distributions.Categorical(action_probs)
    action = dist.sample()
    log_prob_action = dist.log_prob(action)
    return action.item(), log_prob_action

optimizer_policy = optim.Adam(policy_net.parameters(), lr = 0.005)
optimizer_value = optim.Adam(value_net.parameters(), lr = 0.005)


def update(reward_list, logPi_list, state_list):
    logPi_traj = torch.cat(logPi_list)

    gammas = []
    Rts = []
    t = 0
    R = 0
    for reward in reversed(reward_list):
        gammas.append(gamma ** t)
        R = reward + gamma * R
        Rts.insert(0, R)
        t += 1
    
    gamma_traj = torch.tensor(gammas)
    Rt_traj = torch.tensor(Rts)

    states = torch.tensor(np.stack(state_list, axis = 0))
    value_traj = value_net(states).squeeze()

    # 가치 손실과 업데이트
    loss_value = F.mse_loss(Rt_traj, value_traj)
    optimizer_value.zero_grad()
    loss_value.backward()
    optimizer_value.step()

    #정책 손실과 업데이트
    delta_traj = (Rt_traj - value_traj).detach()

    loss_policy = -(logPi_traj*gamma_traj*delta_traj).sum()
    optimizer_policy.zero_grad()
    loss_policy.backward()
    optimizer_policy.step()

#30번 연속해서 만점을 받으면 학습 종료할 때 사용할 함수
def consecutive_length(lst):
    return [len(list(group)) for _, group in groupby(enumerate(lst), lambda i_x: i_x[0] - i_x[1])]

# main
episode_durations = []
movement200 = []
num_episodes = 5000
t0 = time.time()

for episode in range(num_episodes):
    logPi_list = []
    reward_list = []
    state_list = []

    state = env.reset()
    for i_movement in range(200):
        action, log_prob_action = select_action(state)
        new_state, reward, done, _ = env.step(action)
        state_list.append(state)
        logPi_list.append(log_prob_action)
        reward_list.append(reward)
        if done:
            episode_durations.append(i_movement+1)
            movement_end = i_movement +1
            break
        state = new_state
    #파라미터 업데이트
    update(reward_list, logPi_list, state_list)
    #episode 종료조건 
    if movement_end == 200:
        movement200.append(episode)
        print(episode, ',',max(consecutive_length(movement200)))
        if max(consecutive_length(movement200)) >= 30:
         break

t1 = time.time()
print("Trainging time: {}min".format(round((t1-t0)/60,2)))

plt.title('REINFORCE_Baseline training')
plt.ylabel('Duration')
plt.xlabel('Episode')
plt.plot(episode_durations)
plt.savefig('REINFORCE_Baseline_training.png', dpi = 300, bbox_inches = 'tight')
plt.show()

# 검증

policy_net.eval()
scores = []
with torch.no_grad():
    for i_episode in tqdm(range(50)):
        state = env.reset()
        done=False
        t = 0
        while not done:
            t += 1
            with torch.no_grad():
                action_prob = policy_net(torch.tensor(state, dtype = torch.float32))
            action = torch.argmax(action_prob, dim = -1)
            next_state, reward, done, _ = env.step(action.item())
            state = next_state
            if done:
                scores.append(t)
                if t != 200:
                    print(t)
print('검증', statistics.mean(scores))
print('Standard Devitation', statistics.stdev(scores))