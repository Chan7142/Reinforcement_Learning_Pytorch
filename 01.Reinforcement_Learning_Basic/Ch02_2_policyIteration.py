import random, numpy as np
from ray import state

gamma = 0.5 #할인율
theta = 1e-10 #업데이트 멈춤 상수

class SimpleEnvironment:
    def __init__(self):
        self.state = ["s1", "s2"]
        self.actions = ["a1", "a2", "a3"]

        self.trans = {
            #(상태, 행동) : {(다음상태, 보상): 확률}
            ("s1", "a1") : {("s1", +100): 1},
            ("s1", "a2") : {("s1", +4): 0.7, ("s2", +5): 0.3},

            ("s2", "a1") : {("s1", +1): 1},
            ("s2", "a2") : {("s1", -3): 0.8, ("s2", -2): 0.2},
            ("s2", "a3") : {("s1", +100): 1}
        }

    def reset(self):
        self.current_state = random.choice(self.state)
        return self.current_state

    def step(self, action):
        trans_values = self.trans[(self.current_state, action)]
        next_state, reward = random.choice(list(trans_values.items()))[0]
        self.current_state = next_state
        return next_state, reward

env = SimpleEnvironment()

#초기 정책 (무작위)

policy = {state: random.choice(
           [action  for action in env.actions
            if not (state == "s1" and action == "a3")]
            ) for state in env.state}

def policy_evaluation(env, policy, gamma=0.5, theta=1e-10):
    #정책 평가 알고리즘
    v = {state: 0 for state in env.state}

    while True:
        delta = 0
    
        for state in env.state:
            old_v = v[state]
            sum_s1 = 0
            for (next_state, reward), prob in env.trans[(state, policy[state])].items():
                sum_s1 += prob * (reward + gamma * v[next_state])

            v[state] = sum_s1

            delta = max(delta, abs(old_v - v[state]))

        if delta < theta:
            break
    return v

def policy_iteration(env, policy):

    while True:

        # 1. 정책 평가
        v = policy_evaluation(env, policy, gamma, theta)

        # 2. 정책 개선
        policy_stable = True

        for state in env.state:

            old_action = policy[state]

            q_values = []

            relevant_actions = [
                action
                for (s, action) in env.trans.keys()
                if s == state
            ]

            # 모든 행동의 Q값 계산
            for action in relevant_actions:

                q_value = 0

                for (next_state, reward), trans_prob in env.trans[(state, action)].items():

                    q_value += trans_prob * (
                        reward + gamma * v[next_state]
                    )

                q_values.append(q_value)

            # 가장 좋은 행동 선택
            best_action = relevant_actions[np.argmax(q_values)]

            policy[state] = best_action

            # 정책이 바뀌었나?
            if old_action != best_action:
                policy_stable = False

        # 모든 상태에서 정책이 안 바뀌었다면 종료
        if policy_stable:
            break

    return policy

policy_u = policy_iteration(env, policy)
print("최적 정책:", policy_u)