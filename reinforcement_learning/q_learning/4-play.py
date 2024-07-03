#!/usr/bin/env python3
"""

"""
import numpy as np

def play(env, Q, max_steps=100):
    """
    
    """
    agent = Q
    reset = env.reset()
    state = reset[0]
    total_reward = 0
    done = False
    for step in range(max_steps):
        action = np.argmax(agent[state, :])
        next_state, reward, done, _, _ = env.step(action)
        agent[state, action] += reward
        total_reward += reward
        state = next_state
        if done:
            break
    return total_reward