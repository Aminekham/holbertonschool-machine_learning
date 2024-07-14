#!/usr/bin/env python3
import numpy as np
"""
The TD lambtha algorithm
"""
def td_lambtha(env, V, policy, lambtha, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99):
    """
    The TD lambtha algorithm which combines
    the two approaches of q-learning which is a TD(0)
    and the monte carlo based evaluation approach which is TD(1)
    into one single algorithm with a lambtha as a switch
    """
    for ep in range(episodes):
        state = env.reset()
        eligibility = np.zeros_like(V)
        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            delta = reward + gamma * V[next_state] - V[state]
            eligibility[state] += 1
            V += alpha * delta * eligibility
            eligibility *= gamma * lambtha
            if done:
                break
            state = next_state
    return V
