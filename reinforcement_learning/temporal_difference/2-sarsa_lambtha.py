#!/usr/bin/env python3
import numpy as np

def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):    
    n_actions = env.action_space.n
    def epsilon_greedy_policy(Q, state, epsilon, n_actions):
        if np.random.rand() < epsilon:
            return np.random.choice(n_actions)
        else:
            return np.argmax(Q[state])
    for episode in range(episodes):
        state = env.reset()
        action = epsilon_greedy_policy(Q, state, epsilon, n_actions)
        eligibility_traces = np.zeros_like(Q)
        for _ in range(max_steps):
            next_state, reward, done, _ = env.step(action)
            next_action = epsilon_greedy_policy(Q, next_state, epsilon, n_actions)
            delta = reward + gamma * Q[next_state, next_action] - Q[state, action]
            eligibility_traces[state, action] += 1
            Q += alpha * delta * eligibility_traces
            eligibility_traces *= gamma * lambtha
            if done:
                break
            state, action = next_state, next_action
        epsilon = max(min_epsilon, epsilon * np.exp(-epsilon_decay * episode))
    return Q
