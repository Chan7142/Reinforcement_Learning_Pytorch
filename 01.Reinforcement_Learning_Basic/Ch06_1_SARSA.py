import collections, random, numpy as np

alpha = 0.1 # 학습률
gamma = 0.9 # 할인율
epsilon = 0.1 # 탐험확률

class GridEnv:
    def __init__(self):
        self.gridSize = 4
        self.terminalStates = [(0,0), (self.gridSize-1, self.gridSize-1)]
        self.actions = [(1,0), (0,1), (-1, 0), (0, -1)]
        self.states = [(i, j) for i in range(self.gridSize)
                       for j in range(self.gridSize)
                       if (i, j) not in self.terminalStates]

    def reset(self):
        self.current_state = random.choice(self.states)
        return self.current_state

    def step(self, action):
        next_state = (self.current_state[0] + action[0], self.current_state[1] + action[1])
        self.current_state = next_state
        return next_state, -1

env = GridEnv()

def valid_actions(state):
    valid_actions = []
    for action in env.actions:
        next_state = (state[0] + action[0], state[1] + action[1])
        if 0 <= next_state[0] < env.gridSize and 0 <= next_state[1] < env.gridSize:
            valid_actions.append(action)
    return valid_actions

def epsilon_greedy_policy(state, Q):
    if state in env.terminalStates:
        return None

    actions = valid_actions(state)
    if random.random() < epsilon:
        return random.choice(actions)
    else:
        return max(actions, key=lambda action: Q[(state, action)]) ## 현재상태에서 최고의 액션을 반환

# SARSA 알고리즘

def sarsa(episodes):
    Q = collections.defaultdict(float)
    for i in range(episodes):
        state = env.reset()
        action = epsilon_greedy_policy(state, Q)

        while state not in env.terminalStates:
            next_state, reward = env.step(action)

            if next_state not in env.terminalStates:
                next_action = epsilon_greedy_policy(next_state, Q)
                Q[(state, action)] += alpha * (reward + gamma * Q[(next_state, next_action)] - Q[(state, action)])
            state, action = next_state, next_action

    return Q

# SARSA 학습
Q = sarsa(episodes = 1000)

#출력
for state in env.states:
    for action in valid_actions(state):
        print(f"Q[{state},{action}] = {Q[(state, action)]:.2f}")











