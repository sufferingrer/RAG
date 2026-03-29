import gym
import numpy as np
import time

# Создаем среду CartPole
env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_n = env.action_space.n

class SimulatedAnnealingAgent():
    def __init__(self, state_dim, action_n, noise_scale=0.1, initial_temp=1.0, cooling_rate=0.99):
        self.state_dim = state_dim
        self.action_n = action_n
        self.weights = np.random.randn(self.state_dim)
        self.noise_scale = noise_scale
        self.temperature = initial_temp
        self.cooling_rate = cooling_rate

    def get_action(self, state):
        """
        Линейная политика: если скалярное произведение наблюдения и весов положительно,
        выбираем действие 1, иначе – 0.
        """
        return 1 if np.dot(state, self.weights) > 0 else 0

    def perturb(self):
        """
        Генерируем кандидата путём добавления случайного шума к текущим весам.
        """
        candidate = self.weights + np.random.randn(self.state_dim) * self.noise_scale
        return candidate

    def update_policy(self, new_weights):
        """
        Обновляем вектор весов.
        """
        self.weights = new_weights

    def cool_down(self):
        """
        Понижаем температуру по схеме экспоненциального охлаждения.
        """
        self.temperature *= self.cooling_rate


def get_trajectory(env, agent, max_len=1000, visualize=False):
    trajectory = {
        'states': [],
        'actions': [],
        'rewards': []
    }
    state = env.reset()
    total_reward = 0

    for _ in range(max_len):
        trajectory['states'].append(state)
        action = agent.get_action(state)
        trajectory['actions'].append(action)
        state, reward, done, _ = env.step(action)
        trajectory['rewards'].append(reward)
        total_reward += reward

        if visualize:
            env.render()
            time.sleep(0.05)

        if done:
            break

    trajectory['total_reward'] = total_reward
    return trajectory


num_iterations = 1000  # число итераций
agent = SimulatedAnnealingAgent(state_dim, action_n, noise_scale=0.1, initial_temp=1.0, cooling_rate=0.99)

trajectory = get_trajectory(env, agent)
current_reward = trajectory['total_reward']
best_reward = current_reward
best_weights = agent.weights.copy()

print("Начальная награда:", current_reward)

for iteration in range(num_iterations):
    current_weights = agent.weights.copy()
    candidate_weights = agent.perturb()
    agent.update_policy(candidate_weights)
    trajectory = get_trajectory(env, agent)
    candidate_reward = trajectory['total_reward']

    if candidate_reward > current_reward:
        current_reward = candidate_reward
        if candidate_reward > best_reward:
            best_reward = candidate_reward
            best_weights = candidate_weights.copy()
        print(f"Итерация {iteration}: улучшение, награда = {candidate_reward}")
    else:
        delta = candidate_reward - current_reward
        acceptance_prob = np.exp(delta / agent.temperature)
        if np.random.rand() < acceptance_prob:
            current_reward = candidate_reward
            print(f"Итерация {iteration}: принята ухудшающая кандидатура, награда = {candidate_reward}")
        else:
            agent.update_policy(current_weights)
    agent.cool_down()

print("\nЛучшая полученная награда:", best_reward)
agent.update_policy(best_weights)

trajectory_demo = get_trajectory(env, agent, max_len=1000, visualize=True)
print("Награда демонстрационного эпизода:", trajectory_demo['total_reward'])
env.close()