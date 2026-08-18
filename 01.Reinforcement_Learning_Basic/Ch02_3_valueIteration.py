import random, numpy as np
from ray import state
from Ch02_2_policyIteration import SimpleEnvironment, policy, policy_evaluation
env = SimpleEnvironment()


def value_iteration():
    v = {state: 0 for state in env.state}
    gamma = 0.5
    theta = 1e-10
    #step1 최적의 상태 가치 함수 v(s)를 찾는다.
    while True:
        delta = 0
        for state in env.state:
            old_v = v[state]
            q_values = []

            relevant_actions = [
                action for (s, action) in env.trans.keys()
                if s == state]
            for action in relevant_actions:
                q_value = 0
                for (next_state, reward), trans_prob in env.trans[(state, action)].items():
                    q_value += trans_prob * (reward + gamma * v[next_state])
                q_values.append(q_value)
            v[state] = max(q_values)
            delta = max(delta, abs(old_v - v[state]))
        if delta < theta:
            break
    
    #step2 최적 정책을 찾는다.
    for state in env.state:
        q_values = []
        relevant_actions = [
            action for (s, action) in env.trans.keys()
            if s == state]
        for action in relevant_actions:
            q_value = 0
            for (next_state, reward), trans_prob in env.trans[(state, action)].items():
                q_value += trans_prob * (reward + gamma * v[next_state])
            q_values.append(q_value)
        policy[state] = relevant_actions[np.argmax(q_values)]
    return policy

optimal_policy = value_iteration()
print("최적 정책:", optimal_policy)