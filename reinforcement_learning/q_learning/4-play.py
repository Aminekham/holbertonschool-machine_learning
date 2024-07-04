#!/usr/bin/env python3
"""

"""
import numpy as np

def play(env, Q, max_steps=100):
    """
    
    """
    state = env.reset()
    env.render()
    total_reward = 0
    state = state[0]
    for step in range(max_steps):
        action = np.argmax(Q[state, :])
        next_state, reward, done, _, _ = env.step(action)
        total_reward += reward
        env.render()
        state = next_state
        if done:
            break
    return total_reward
