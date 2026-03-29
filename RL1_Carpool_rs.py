import gym
import numpy as np
import time

env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_n = env.action_space.n
print("State Dimension:", state_dim)
print("Action Space:", action_n)


class RandomSearchAgent():
    def __init__(self, state_dim, action_n):
        self.state_dim = state_dim
        self.action_n = action_n
        self.weights = np.random.randn(self.state_dim)

    def get_action(self, state):
        """
        Линейная политика: если скалярное произведение state и весов положительно,
        выбираем действие 1, иначе – 0.
        """
        return 1 if np.dot(state, self.weights) > 0 else 0

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


num_iterations = 1000
best_reward = -np.inf
best_weights = None

agent = RandomSearchAgent(state_dim, action_n)

for iteration in range(num_iterations):
    candidate_weights = np.random.randn(state_dim)
    agent.update_policy(candidate_weights)

    trajectory = get_trajectory(env, agent)
    total_reward = trajectory['total_reward']

    if total_reward > best_reward:
        best_reward = total_reward
        best_weights = candidate_weights.copy()

    print("Iteration:", iteration, "Total Reward:", total_reward)

print("\nBest obtained reward:", best_reward)
agent.update_policy(best_weights)

trajectory_demo = get_trajectory(env, agent, max_len=1000, visualize=True)
print("Total reward in demonstration:", trajectory_demo['total_reward'])
env.close()