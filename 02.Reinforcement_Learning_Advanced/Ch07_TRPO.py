import gym, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import copy, matplotlib.pyplot as plt, warnings, os, io, base64
from glob import glob
from IPython.display import display, HTML
warnings.filterwarnings("ignore")
np.bool8 = np.bool_

# ----------------------------------------------------------------------------
# 1. 환경 및 하이퍼파라미터 설정
# ----------------------------------------------------------------------------
env = gym.make("MountainCarContinuous-v0")
gamma = 0.995
gae_lambda = 0.97       # GAE (Generalized Advantage Estimation) 람다
max_kl = 0.005           # KL 발산 허용 최댓값 (TRPO 업데이트 시 사용)
damping = 0.1            # TRPO 업데이트 시 사용하는 damping 계수
max_step = 999
batch_size = 5000
max_episodes = 500
solved_reward = 90       # 문제 해결로 간주할 보상 기준

#MountainCarContinuous-v0: observation_space.shape = (2,), action_space = Box([-1, 1], [1, 1])
state_dim = env.observation_space.shape[0]  # 2
action_dim = env.action_space.shape[0]      # 1


def plot_train_history(scores):
    with plt.style.context("seaborn-v0_9-dark-palette"):
        plt.figure(figsize=(10, 5))
        plt.plot(scores, c="crimson")
        plt.title(f"Score {np.mean(scores[-10:])}")
        plt.xlabel("Episode")
        plt.tight_layout()
        plt.show()


# ----------------------------------------------------------------------------
# 2. 모델 정의 (Actor & Critic)
# ----------------------------------------------------------------------------
class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3_value = nn.Linear(32, 1)
        
        nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc2.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc3_value.weight, gain=np.sqrt(2))
        nn.init.constant_(self.fc3_value.bias, 0.0)
        
    def forward(self, states_batch):
        # states_batch: (batch, state_dim)
        out = torch.tanh(self.fc1(states_batch))
        out = torch.tanh(self.fc2(out))
        states_values = self.fc3_value(out)
        return states_values.squeeze()  # (batch,)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3_mean = nn.Linear(32, action_dim)
        # 초기 시그마: exp(0.5) ≈ 1.65
        self.log_sigma = nn.Parameter(torch.full((action_dim,), 0.5))
        
        nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc2.weight, gain=np.sqrt(2))
        # 마지막 레이어는 초기 출력을 0 근처로 만들기 위해 gain을 작게 설정
        nn.init.orthogonal_(self.fc3_mean.weight, gain=0.01)
        nn.init.constant_(self.fc3_mean.bias, 0.0)
        
    def forward(self, states_batch):
        # states_batch: (batch, state_dim)
        out = torch.tanh(self.fc1(states_batch))
        out = torch.tanh(self.fc2(out))
        action_mean = self.fc3_mean(out)       # (batch, action_dim)
        action_sigma = torch.exp(self.log_sigma) # (action_dim,)
        
        # 배치 크기에 맞춰 시그마 확장
        action_sigma = action_sigma.expand_as(action_mean)
        
        return action_mean, action_sigma

    def get_action(self, states):
        # states: (batch, state_dim)
        with torch.no_grad():
            mu, sigma = self(states)
            dist = torch.distributions.Normal(mu, sigma)
            action = dist.sample()
            return action

    def get_log_prob(self, states, actions):
        # states: (batch, state_dim), actions: (batch, action_dim)
        mu, sigma = self(states)
        dist = torch.distributions.Normal(mu, sigma)
        # 연산(Continuous Action)의 경우, 각 차원의 log_prob를 합산
        log_prob = dist.log_prob(actions).sum(dim=-1)
        return log_prob


# ----------------------------------------------------------------------------
# 3. 유틸리티 클래스
# ----------------------------------------------------------------------------
class Online_mean_std:
    """
    상태의 평균과 표준편차를 온라인으로 업데이트하는 클래스.
    Welford's Algorithm 사용.
    """
    def __init__(self, state_dim):
        self.len = 0
        self.online_mean = np.zeros(state_dim)
        self.online_var = np.zeros(state_dim)
        
    def update(self, new_state):
        self.len += 1
        old_mean = self.online_mean.copy()
        self.online_mean += (new_state - self.online_mean) / self.len
        self.online_var += (new_state - old_mean) * (new_state - self.online_mean)
        
    def mean(self):
        return self.online_mean
        
    def std(self):
        if self.len > 1:
            return np.sqrt(self.online_var / (self.len - 1))
        else:
            return np.zeros_like(self.online_mean)


# ----------------------------------------------------------------------------
# 4. TRPO 핵심 함수들
# ----------------------------------------------------------------------------
def update_critic(critic, states_batch, targets):
    """
    LBFGS 옵티마이저를 사용하여 Critic을 업데이트.
    """
    critic_optimizer = torch.optim.LBFGS(
        critic.parameters(),
        lr = 0.001,
        max_iter = 30,
        history_size = 10,
        line_search_fn = 'strong_wolfe',
    )
    
    def closure():
        critic_optimizer.zero_grad()
        states_values = critic(states_batch)
        # targets shape: (batch,), states_values shape: (batch,)
        critic_loss = F.mse_loss(states_values, targets)
        critic_loss.backward()
        return critic_loss
    
    critic_optimizer.step(closure)


def get_kl_divergence(old_actor, actor, states):
    """
    가우시안 분포에 기반한 KL 발산 계산.
    KL(Old || New)
    """
    with torch.no_grad():
        old_mus, old_sigmas = old_actor(states)
    new_mus, new_sigmas = actor(states)
    
    # 각 차원에 대한 KL 발산 계산 후 평균
    # Formula: log(sigma_new/sigma_old) + (sigma_old^2 + (mu_old - mu_new)^2) / (2 * sigma_new^2) - 0.5
    kl_div = (
        torch.log(new_sigmas / old_sigmas)
        + (old_sigmas**2 + (old_mus - new_mus)**2) / (2 * new_sigmas**2)
        - 0.5
    )
    
    # (batch, action_dim) -> (batch,) -> scalar
    return torch.mean(kl_div)


def Hessian_vector_product(old_actor, actor, states, vector):
    """
    Hessian 행렬과 벡터의 내적 (Hv) 계산.
    create_graph=True를 이용해 2차 미분 가능하게 함.
    """
    kl_div = get_kl_divergence(old_actor, actor, states)
    
    # 1차 미분 (Gradient)
    kl_grads = torch.autograd.grad(kl_div, actor.parameters(), create_graph=True)
    kl_grad_flat = torch.cat([grad.view(-1) for grad in kl_grads])
    
    # Gradient와 Vector의 내적 (스칼라)
    kl_grad_vec_product = kl_grad_flat.dot(vector)
    
    # 2차 미분 (Hessian-Vector Product)
    hess_vec_prod = torch.autograd.grad(kl_grad_vec_product, actor.parameters())
    hess_vec_prod_flat = torch.cat([grad.view(-1) for grad in hess_vec_prod]).detach()
    
    # Damping (정규화)
    return hess_vec_prod_flat + vector * damping


def Conjugate_gradient(old_actor, actor, states, actor_loss_grad_flat, nsteps, max_error=1e-6):
    """
    Conjugate Gradient(연직 그라디언트) 알고리즘.
    Hessian 행렬을 직접 만들지 않고, Hessian-Vector Product만 사용하여 최적 방향을 찾음.
    """
    x = torch.zeros_like(actor_loss_grad_flat)
    r = actor_loss_grad_flat.clone()
    p = actor_loss_grad_flat.clone()
    rr = r.dot(r)
    
    if rr < max_error:
        return x
        
    for i in range(nsteps):
        Ap = Hessian_vector_product(old_actor, actor, states, p)
        
        # p와 Ap가 모두 0에 가깝거나 Ap가 음수를 가지는 등 수치적 문제가 생길 수 있음
        # 일반적으로 p.dot(Ap) > 0 인지 확인
        if p.dot(Ap) < 1e-8:
            break
            
        alpha = rr / p.dot(Ap)
        x += alpha * p
        r -= alpha * Ap
        
        new_rr = r.dot(r)
        if new_rr < max_error:
            break
            
        beta = new_rr / rr
        p = r + beta * p
        rr = new_rr
        
    return x


def apply_flat_params(model, flat_params):
    """
    평탄화(flat)된 파라미터를 모델에 다시 적용.
    """
    prev_ind = 0
    for param in model.parameters():
        flat_size = param.numel()
        param.data.copy_(flat_params[prev_ind:prev_ind + flat_size].view_as(param))
        prev_ind += flat_size


def actor_loss_fn_factory(actor, states, actions, old_log_probs, advantages):
    """
    Actor Loss 함수 팩토리.
    """
    def loss_fn(flat_params=None):
        if flat_params is not None:
            apply_flat_params(actor, flat_params)
            
        # new log prob 계산
        # 주의: actor가 flat_params로 업데이트되었으므로, forward는 그 파라미터를 사용
        new_log_probs = actor.get_log_prob(states, actions)
        
        # Policy Ratio
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        # TRPO 정책 기울기 손실 (Surrogate Objective)
        # J(θ) = E[ (π_θ(a|s) / π_old(a|s)) * A(s,a) ]
        # Loss는 -J(θ)
        loss = -torch.mean(ratio * advantages)
        
        return loss
    return loss_fn


def linesearch(loss_fn, flat_params, full_step, actor, old_actor, states, max_kl):
    """
    Line Search.
    계산된 step이 KL 제약과 Loss 개선 조건을 만족하는지 확인하고,
    필요 시 step 크기를 축소.
    """
    loss_current = loss_fn().item()
    
    # step size 축소 시도
    for step in [1.0, 0.5, 0.25, 0.125, 0.0625]:
        flat_params_new = flat_params + step * full_step
        
        # 새로운 파라미터 적용 (Loss 계산용)
        # loss_fn이 flat_params를 받으면 내부에서 적용
        loss_new = loss_fn(flat_params_new).item()
        
        actual_improvement = loss_current - loss_new
        kl_new = get_kl_divergence(old_actor, actor, states).item()
        
        # Loss가 감소하고(개선), KL이 허용 범위 이내일 경우 성공
        if actual_improvement > 0 and kl_new <= max_kl:
            return flat_params_new
        
    # 성공하지 못하면 원래 파라미터 반환
    return flat_params


def update_actor(old_actor, actor, states, actions, old_log_probs, advantages, max_kl):
    """
    Actor 업데이트 메인 로직.
    1. Loss Gradient 계산
    2. Conjugate Gradient로 최적 방향 탐색
    3. KL 제약을 만족하는 최대 Step Size 계산
    4. Line Search로 최종 파라미터 결정
    """
    loss_fn = actor_loss_fn_factory(actor, states, actions, old_log_probs, advantages)
    
    # 1. 현재 Loss의 Gradient 계산
    loss = loss_fn()
    grads = torch.autograd.grad(loss, actor.parameters())
    loss_grad_flat = torch.cat([grad.view(-1) for grad in grads]).detach()
    
    # 2. Conjugate Gradient로 Hessian과 Gradient의 내적 방향(step_direction) 계산
    # 이 방향은 Hessian의 역행렬에 Gradient를 곱한 것과 유사한 방향
    step_direction = Conjugate_gradient(old_actor, actor, states, loss_grad_flat, nsteps=200)
    
    # 3. 최대 Step Size 계산 (KL Constraint 기반)
    # xHx = (step_direction)^T * Hessian * (step_direction)
    # Hessian을 직접 계산하지 않고 HVP를 이용해 계산
    hess_step = Hessian_vector_product(old_actor, actor, states, step_direction)
    xHx = step_direction.dot(hess_step)
    
    if xHx <= 1e-8:
        # Hessian이 양의 정부호가 아니면(평탄하거나 음수 곡률) 업데이트 중단
        return
        
    # max_step_size = sqrt(2 * max_kl / xHx)
    max_step_size = torch.sqrt(2 * max_kl / xHx).item()
    
    # 4. Full Step 결정
    # Gradient 하강 방향이므로 마이너스
    full_step = - step_direction * max_step_size
    
    # 5. Line Search
    flat_params = torch.cat([param.data.view(-1) for param in actor.parameters()])
    flat_params_new = linesearch(loss_fn, flat_params, full_step, actor, old_actor, states, max_kl)
    
    # 6. Actor 파라미터 업데이트
    apply_flat_params(actor, flat_params_new)


# ----------------------------------------------------------------------------
# 5. TRPO 클래스 (학습 루프 정리)
# ----------------------------------------------------------------------------
class TRPO:
    def __init__(self, actor, max_kl):
        self.actor = actor
        self.max_kl = max_kl
        self.scores = []

    def train(self, critic, actor, old_actor, memory, batch_size):
        states, actions, rewards, _, dones = zip(*memory)
        
        # numpy -> torch 변환
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.float32)
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)
        
        states_actor = torch.FloatTensor(states)
        actions_actor = torch.FloatTensor(actions)
        
        # 1. Critic으로 Value 계산
        with torch.no_grad():
            values = critic(states_actor).numpy()
        
        # 2. GAE (Generalized Advantage Estimation) 및 Returns 계산
        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)
        next_return = 0
        next_value = 0
        next_advantage = 0
        
        for t in reversed(range(len(memory))):
            # Note: MountainCarContinuous에서 episode 종료 시 done=True
            # 보상이 마지막 step에 더해지므로, done 처리는 주의해야 함
            # 일반적인 GAE 구현
            delta = rewards[t] + gamma * next_value - values[t]
            advantages[t] = delta + gamma * gae_lambda * next_advantage
            next_return = rewards[t] + gamma * next_return + (1 - dones[t]) * 0 
            # Returns 계산은 일반적으로: R_t = r_t + gamma * R_{t+1}
            # 여기서 done이 True이면 보상이 끝남
            # 단순화를 위해 다음과 같이 반환 계산
            # 더 정확한 Returns:
            # returns[t] = rewards[t] + gamma * next_return
            # next_return = returns[t] if not done[t] else 0
            # 하지만 GAE는 advantages만 주로 사용하며, Critic 업데이트에는 returns를 씀
            # 여기서는 GAE 계산에 집중하고 returns는 표준 방식 사용
            pass
        
        # Returns를 올바르게 계산하기 위해 다시 계산 (Critic 업데이트용)
        next_value = 0
        for t in reversed(range(len(memory))):
            next_value = rewards[t] + gamma * next_value * (1 - dones[t])
            returns[t] = next_value
            
        # Advantage 정규화
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        returns = torch.FloatTensor(returns)
        advantages = torch.FloatTensor(advantages)
        
        # 3. Old Log Probs 계산 (Line Search 및 Ratio 계산용)
        # 주의: old_actor는 현재 actor와 동일한 파라미터를 가짐 (load_state_dict됨)
        with torch.no_grad():
            old_log_probs = old_actor.get_log_prob(states_actor, actions_actor).detach()
            
        # 4. Critic 업데이트
        update_critic(critic, states_actor, returns)
        
        # 5. Actor 업데이트
        update_actor(old_actor, actor, states_actor, actions_actor, old_log_probs, advantages, self.max_kl)


# ----------------------------------------------------------------------------
# 6. 메인 함수
# ----------------------------------------------------------------------------
def main():
    critic = Critic(state_dim)
    actor = Actor(state_dim, action_dim)
    old_actor = Actor(state_dim, action_dim)
    state_stat = Online_mean_std(state_dim)
    
    # TRPO 객체 생성
    trpo = TRPO(actor=actor, max_kl=max_kl)
    
    for episode in range(max_episodes):
        # 1. Old Actor에 현재 Actor 파라미터 복사
        old_actor.load_state_dict(actor.state_dict())
        
        episode_score = 0
        episode_count = 0
        memory = []
        
        # 2. 데이터 수집 (Batch Size만큼)
        while len(memory) < batch_size:
            episode_count += 1
            state = env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                # 상태 정규화 (선택 사항)
                # state_norm = (state - state_stat.mean()) / (state_stat.std() + 1e-8)
                # state = state_norm
                
                # Action 선택
                action = actor.get_action(torch.FloatTensor(state).unsqueeze(0)).detach().numpy()[0]
                
                next_state, reward, done, info = env.step(action)
                
                memory.append((state, action, reward, next_state, done))
                state = next_state
                episode_reward += reward
                
                if done:
                    break
                    
            episode_score += episode_reward
            
            # 상태 통계 업데이트
            for s in memory[-episode_count:]: # 최근 episode의 상태들 업데이트 (단순화)
                state_stat.update(s[0])
        
        # 3. 에피소드 평균 점수
        episode_score /= episode_count
        trpo.scores.append(episode_score)
        
        # 4. TRPO 학습
        trpo.train(critic, actor, old_actor, memory, batch_size)
        
        if episode % 5 == 0:
            print(f"Episode {episode}, Episode Score: {episode_score:.2f}, KL: {trpo.max_kl}")
            
        # 5. 종료 조건
        if episode > 50 and np.mean(trpo.scores[-10:]) <= 0:
            print("Training stopped due to poor performance.")
            break
            
        if trpo.scores[-1] >= solved_reward:
            print(f"# Solved! Score: {trpo.scores[-1]}")
            plot_train_history(trpo.scores)
            break
            
    return actor, state_stat


def evaluate(actor, num_episodes=5):
    """
    학습된 Actor 평가
    """
    env = gym.make("MountainCarContinuous-v0")
    
    total_reward = 0
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            with torch.no_grad():
                action = actor.get_action(torch.FloatTensor(state).unsqueeze(0)).numpy()[0]
            
            next_state, reward, done, info = env.step(action)
            state = next_state
            episode_reward += reward
            
        print(f"Episode {episode}, Reward: {episode_reward:.2f}")
        total_reward += episode_reward
        
    print(f"Average Reward: {total_reward / num_episodes:.2f}")
    env.close()


if __name__ == "__main__":
    t0 = time.time()
    actor, state_stat = main()
    t1 = time.time()
    print(f"Training Time: {t1 - t0:.2f} seconds")
    
    print("\n## After Training, evaluate the trained policy")
    evaluate(actor, num_episodes=5)
    
    env.close()