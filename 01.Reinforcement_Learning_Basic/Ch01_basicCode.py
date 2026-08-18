import random

#환경코드
class SimpleEnvironment:
    def __init__(self):
        self.state = ["s1", "s2"]
        self.actions = ["a1", "a2", "a3"]

        self.trans = {
            #(상태, 행동) : {(다음상태, 보상): 확률}
            ("s1", "a1") : {("s1", +1): 1},
            ("s1", "a2") : {("s1", +4): 0.5, ("s2", +5): 0.5},

            ("s2", "a1") : {("s1", +1): 1},
            ("s2", "a2") : {("s1", -3): 0.8, ("s2", +2): 0.2},
            ("s2", "a3") : {("s1", 0): 1}
        }

    def reset(self):
        self.current_state = random.choice(self.state)
        return self.current_state

    def step(self, action):
        trans_values = self.trans[(self.current_state, action)]
        next_state, reward = random.choice(list(trans_values.items()))[0]
        self.current_state = next_state
        return next_state, reward

class GridEnv:
    def __init__(self):
        self.gridSize = 4
        self.terminalStates = [(0,0), (self.gridSize-1, self.gridSize-1)]
        self.actions = [(1, 0), (-1, 0), (0, 1), (0, -1)] # down, up, right, left
        self.states = [(i, j) for i in range(self.gridSize) for j in range(self.gridSize)
                       if (i, j) not in self.terminalStates]

    def reset(self):
            self.current_state = random.choice(self.states)
            return self.current_state

    def step(self, action):
            next_state = (self.current_state[0] + action[0], self.current_state[1] + action[1])
            reward = -1
            self.current_state = next_state
            return next_state, reward

env = GridEnv()
#정책코드
def deterministic_policy(state):
    if state[0] == 0 or state[1] == 3:
        return (1, 0) # down
    else:
         return (0, 1) # right

def stochastic_policy(state):

    while True:
         action = random.choice(env.actions)
         next_state = (state[0] + action[0], state[1] + action[1])

         if 0 <= next_state[0] < env.gridSize \
            and 0 <= next_state[1] < env.gridSize:
             return action

#상호작용 코드

print("env-policy 상호작용 5회이하")
print("현재상태, 행동, 다음상태, 보상")

state = env.reset()
for _ in range(5):
     action = deterministic_policy(state)
     next_state, reward = env.step(action)

     print(f"{state}, {action}, {next_state}, {reward}")
     if next_state in env.terminalStates:
            print("도착상태에 도달했습니다.")
            break
     state = next_state

print("env-policy 상호작용 5회이하")
print("현재상태, 행동, 다음상태, 보상")

state = env.reset()
for _ in range(5):
     action = stochastic_policy(state)
     next_state, reward = env.step(action)

     print(f"{state}, {action}, {next_state}, {reward}")
     if next_state in env.terminalStates:
            print("도착상태에 도달했습니다.")
            break
     state = next_state