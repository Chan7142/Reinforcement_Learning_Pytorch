from Ch01_basicCode import GridEnv, SimpleEnvironment


#상수
gamma = 0.5 #할인율
theta = 1e-10 #업데이트 멈춤 상수
env = SimpleEnvironment()
def policy(state):
    if state == "s1":
        action = "a1"
    else:
        action = "a2"
    return action

def policy_evaluation():
    #정책 평가 알고리즘
    v = {state: 0 for state in env.state}

    while True:
        delta = 0
    
        for state in env.state:
            old_v = v[state]
            sum_s1 = 0
            for (next_state, reward), prob in env.trans[(state, policy(state))].items():
                sum_s1 += prob * (reward + gamma * v[next_state])

            v[state] = sum_s1

            delta = max(delta, abs(old_v - v[state]))

        if delta < theta:
            break
    return v

print(policy_evaluation())
