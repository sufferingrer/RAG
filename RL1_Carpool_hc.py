import gym
import numpy as np
import time

env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_n = env.action_space.n


class HillClimbingAgent():
    def __init__(self, state_dim, action_n, noise_scale=0.1):
        self.state_dim = state_dim
        self.action_n = action_n
        self.weights = np.random.randn(self.state_dim)
        self.noise_scale = noise_scale

    def get_action(self, state):
        """
        Линейная политика: если скалярное произведение состояния и весов положительно, выбираем действие 1, иначе – 0.
        """
        return 1 if np.dot(state, self.weights) > 0 else 0

    def perturb(self):
        candidate = self.weights + np.random.randn(self.state_dim) * self.noise_scale
        return candidate

    def update_policy(self, new_weights):
        self.weights = new_weights


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


num_iterations = 10000
noise_scale = 0.2

agent = HillClimbingAgent(state_dim, action_n, noise_scale)
best_weights = agent.weights.copy()
trajectory = get_trajectory(env, agent)
best_reward = trajectory['total_reward']

print("Начальная награда:", best_reward)
for iteration in range(num_iterations):
    candidate_weights = agent.perturb()
    agent.update_policy(candidate_weights)
    trajectory = get_trajectory(env, agent)
    candidate_reward = trajectory['total_reward']

    if candidate_reward > best_reward:
        best_reward = candidate_reward
        best_weights = candidate_weights.copy()
        print(f"Итерация {iteration}: улучшение, награда = {best_reward}")
    else:
        agent.update_policy(best_weights)

print("\nЛучшая полученная награда:", best_reward)
agent.update_policy(best_weights)

trajectory_demo = get_trajectory(env, agent, max_len=1000, visualize=True)
print("Награда демонстрационного эпизода:", trajectory_demo['total_reward'])
env.close()
