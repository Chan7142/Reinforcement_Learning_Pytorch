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
print("에피소드 데이터:")
for data in episode_data:
    print(f"에피소드: {data[0]}, 상태: {data[1]}, 행동: {data[2]}, 보상: {data[3]}, 다음 상태: {data[4]}")
