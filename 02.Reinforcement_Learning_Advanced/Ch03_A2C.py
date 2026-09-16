import os, time, statistics, matplotlib.pyplot as plt, torch, torch.nn as nn
import torch.optim as optim, torch.nn.functional as F, gym, warnings, numpy as np
from torch.distributions import Categorical; from itertools import groupby
from tqdm import tqdm
warnings.filterwarnings('ignore')

np.bool8 = np.bool_

env = gym.make("CartPole-v0")
gamma = 0.99

# 정책 신경망 (actor)
class Actor(nn.Module):
    def __init__(self, input_size = 4, num_actions = 2):
        super(Actor, self).__init__()
        self.input_layer = nn.Linear(input_size, 128)
        self.output_layer = nn.Linear(128, num_actions)
    def forward(self, state):
        x = F.relu(self.input_layer(state))
        action_logits = self.output_layer(x)
        action_prob = F.softmax(action_logits, dim=-1)
        return action_prob

actor = Actor()

# 상태가치 신경망 (critic)
class Critic(nn.Module):
    def __init__(self, input_size = 4):
        super(Critic, self).__init__()
        self.input_layer = nn.Linear(input_size, 128)
        self.output_layer = nn.Linear(128, 1)
    def forward(self, state):
        x = F.relu(self.input_layer(state))
        state_value = self.output_layer(x)
        return state_value

critic = Critic()

#정책 함수: 상태를 입력받아 샘플링한 행동과 로그 확률을 반환
def select_action(state):
    action_probs = actor(state)
    dist = Categorical(action_probs)
    action = dist.sample()
    log_prob_action = dist.log_prob(action)
    return action.item(), log_prob_action

def consecutive_length(lst):
    return [len(list(group)) for _, group in groupby(enumerate(lst), lambda i_x: i_x[0] - i_x[1])]

#파라미터 업데이트를 위한 Optimizer 설정
optimizer_actor = optim.AdamW(actor.parameters(), lr=1e-3)
optimizer_critic = optim.AdamW(critic.parameters(), lr=1e-3)

# main 학습 루프

episode_durations = []
movement200 = []
num_episodes = 5000 # 최대 에피소드 수
t0 = time.time()

for episode in range(num_episodes):
    state = env.reset()
    done = False
    step_count = 0

    while not done:
        state = torch.FloatTensor(state)
        action, log_prob_action = select_action(state)
        next_state, reward, done, _ = env.step(action)

        value = critic(state) # 신경망을 통한 현재상태의 가치 추정
        next_value = critic(torch.FloatTensor(next_state)) # 신경망을 통한 다음 상태의 가치 추정

        #Td target 과 Advantage 계산
        td_target = reward + gamma * next_value * (1-done)
        advantage = td_target - value

        #critic 업데이트
        critic_loss = F.mse_loss(value, td_target.detach())
        optimizer_critic.zero_grad()
        critic_loss.backward()
        optimizer_critic.step()

        # Actor 업데이트
        actor_loss = -(log_prob_action * advantage.detach())
        optimizer_actor.zero_grad()
        actor_loss.backward()
        optimizer_actor.step()

        state = next_state
        step_count += 1

    episode_durations.append(step_count)
    if step_count >= 200:
        movement200.append(episode)
        print(episode, ',',max(consecutive_length(movement200)))
        if max(consecutive_length(movement200)) >= 30:
            break

t1 = time.time() #학습 종료 시간
print("Training time: {}".format(round((t1-t0)/60, 2)))

#학습 결과 시각화
plt.title('A2C_one_step_training')
plt.ylabel('Duration')
plt.xlabel('Episode')
plt.plot(episode_durations)
plt.show()
plt.savefig('A2C_one_step_training.png', dpi = 300, bbox_inches='tight')

# 검증
actor.eval() #평가모드 전환
scores = []

with torch.no_grad():
    #50회의 에피소드 테스트
    for i_episode in tqdm(range(50)):  # 50회의 에피소드 테스트
        state = env.reset()
        done = False
        t = 0
        while not done:
            t += 1
            with torch.no_grad():
                action_prob = actor(torch.tensor(state, dtype = torch.float32))
            action = torch.argmax(action_prob, dim = -1) #가장 확률이 높은 액션 선택
            next_state, _, done, _ = env.step(action.item())
            state = next_state
            if done:
                scores.append(t)
                #200스텝을 채우지 못했다면 점수 출력
                if t!= 200:
                    print(t)

print("평균:", statistics.mean(scores))
print("표준편차: ", statistics.stdev(scores))