from Ch03_basicCode import GridEnv, stochastic_policy

env = GridEnv()

num_episodes = 2
episode_data = []

for episode in range(num_episodes):
    state = env.reset()

    while state not in env.terminalStates:
        action = stochastic_policy(state)
        next_state, reward = env.step(action)
        episode_data.append((episode, state, action, reward, next_state))
        state = next_state

episode_data
# print("에피소드 데이터:")
# for data in episode_data:
#     print(f"에피소드: {data[0]}, 상태: {data[1]}, 행동: {data[2]}, 보상: {data[3]}, 다음 상태: {data[4]}")

import random
gamma = 0.5
class GridEnv:
    def __init__(self):
        self.gridSize = 4
        self.terminalStates = [(0, 0), (3, 3)]
        self.actions = [(1, 0), (-1, 0), (0, 1), (0, -1)]  # down, up, right, left
        self.states = [(i, j) for i in range(self.gridSize) for j in range(self.gridSize)]

    def reset(self):
        self.current_state = random.choice(self.states)
        return self.current_state
    def step(self, action):
        next_state = (self.current_state[0] + action[0], self.current_state[1] + action[1])
        self.current_state = next_state
        return next_state, reward

env = GridEnv()

def stochastic_policy(state):
    while True:
        action = random.choice(env.actions)
        next_state = (state[0] + action[0], state[1] + action[1])
        if 0 <= next_state[0] < env.gridSize and 0 <= next_state[1] < env.gridSize:
            return action

import collections

def mc_every_visit_stateValue(num_episodes):
    returns_sum = collections.defaultdict(float)
    returns_count = collections.defaultdict(float)
    V = collections.defaultdict(float)

    for episode in range(num_episodes):
        state = env.reset()
        episode_data = []

        while state not in env.terminalStates:
            action = stochastic_policy(state)
            next_state, reward = env.step(action)
            episode_data.append((state, action, reward))
            state = next_state

        G = 0
        # visited_states = set()

        for t in range(len(episode_data)-1,-1,-1):
            state, action, reward = episode_data[t]
            G = gamma * G + reward

            # if state not in visited_states:
                # visited_states.add(state)
            returns_sum[state] += G
            returns_count[state] += 1
            V[state] = returns_sum[state] / returns_count[state]
    return V
V = mc_every_visit_stateValue(num_episodes = 1000)

print("상태 가치 함수 V(s):")
for state in sorted(V.keys()):
    print(f"상태: {state}, 가치: {V[state]:.2f}")

#처음방문 상태행동가치

def mc_every_visit_stateActionValue(num_episodes, gamma = 0.5):
    returns_sum = collections.defaultdict(float)
    returns_count = collections.defaultdict(float)
    Q = collections.defaultdict(float)

    for episode in range(num_episodes):
        state = env.reset()
        episode_data = []
        while state not in env.terminalStates:
            action = stochastic_policy(state)
            next_state, reward = env.step(action)
            episode_data.append((state, action, reward))
            state = next_state

        # states_actions_visited = set() ## 상태가치와 다름
        G = 0
        for t in range(len(episode_data)-1,-1,-1):
            state, action, reward = episode_data[t]
            G = gamma * G + reward
            sa = (state, action)
            # if sa not in states_actions_visited:##
                # states_actions_visited.add(sa)
            returns_sum[sa] += G
            returns_count[sa] += 1
            Q[sa] = returns_sum[sa] / returns_count[sa]
    return Q

num_episodes = 10000
Q = mc_every_visit_stateActionValue(num_episodes)

# for state_action, value in sorted(Q.items()):
#     print(f"state-Action : {state_action}, Q-Value: {value:.2f}")

# 최적 정책 구하기

def valid_actions(state, env):
    valid_actions = []
    for action in env.actions:
        next_state = (state[0] + action[0], state[1] + action[1])
        if 0 <= next_state[0] < env.gridSize and 0 <= next_state[1] < env.gridSize:
            valid_actions.append(action)

    return valid_actions

def epsilon_greedy_policy(state, Q, epsilon = 0.1): # 행동 정책
    actions = valid_actions(state, env)
    if random.random() < epsilon:
        return random.choice(actions)
    else:
        return max(actions, key = lambda action: Q[(state, action)])

def mc_every_visit_control(num_episodes, epsilon = 0.1, gamma = 0.5):
    Q = collections.defaultdict(float)
    returns_sum = collections.defaultdict(float)
    returns_count = collections.defaultdict(int)

    policy = {state: random.choice(valid_actions(state, env))
              for state in env.states}

    for episode in range(num_episodes):
        state = env.reset()
        episode_data = []
        while state not in env.terminalStates:
            action = epsilon_greedy_policy(state, Q, epsilon)
            next_state, reward = env.step(action)
            episode_data.append((state, action, reward))
            state = next_state
        G = 0
        # states_actions_visited = set()
        for t in range(len(episode_data)-1, -1, -1):
            state, action, reward = episode_data[t]
            G = gamma * G + reward
            sa = (state, action)
            # if sa not in states_actions_visited:
                # states_actions_visited.add(sa)
            returns_sum[sa] += G
            returns_count[sa] += 1
            Q[sa] = returns_sum[sa]/returns_count[sa]

    for state in policy.keys():
        policy[state] = max(valid_actions(state, env),
                            key = lambda action: Q[(state, action)])
    return policy, Q

num_episodes = 10000
policy, Q = mc_every_visit_control(num_episodes)

for state, action in sorted(policy.items()):
    print(f"State: {state}, Optimal Action {action}")
for state_action, value in sorted(Q.items()):
    print(f"state-Action: {state_action}, Q-value: {value:.2f}")