#!/usr/bin/env python3
"""

"""
import numpy as np

def train(env, Q, episodes=5000, max_steps=100, alpha=0.1,
          gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """

    """
    rewards = []
    for ep in range(episodes):
        x = env.reset()
        state = x[0]
        done = False
        total_reward = 0
        p = np.random.uniform(0, 1)
        for step in range(max_steps):
            if p > epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state, :])
            next_state, reward, done, _, _ = env.step(action)
            Q[state][action] = Q[state][action] + alpha * (reward + gamma * Q[next_state][action])
            state = next_state
            total_reward += reward
            if done:
                break
        rewards.append(total_reward)
        epsilon = max(min_epsilon, epsilon * np.exp(-epsilon_decay * ep))
    return Q, rewards
