#!/usr/bin/env python3
"""
Applying the full training
loop with gradients computing
and updating the policy each episode
based on the monte carlo policy gradients
technique
"""
import numpy as np

def policy(matrix, weight):
    """
    - computing the scores to have the
    unormalized probability of each action given a certain state
    - normalize
    - applying the softmax function to get probabilities
    """
    z = np.exp(np.dot(matrix, weight) - np.max(np.dot(matrix, weight)))
    policy = z / np.sum(z)
    return policy

def policy_gradient(state, weight):
    """
    Computing the needed gradient for
    the monte carlo policy gradient
    REINFORCE
    """
    probs = policy(state, weight)
    action = np.random.choice(probs[0].shape[0], p=probs[0])
    probs[0, action] -= 1
    grad = np.dot(state.T, probs)
    return action, grad

def train(env, nb_episodes, alpha=0.000045, gamma=0.98, show_result=False):
    weight = np.random.rand(env.observation_space.shape[0], env.action_space.n)
    scores = []
    for episode in range(nb_episodes):
        state = env.reset()
        episode_rewards = []
        episode_gradients = []
        done = False
        while not done:
            if show_result and episode % 1000 == 0:
                env.render()
            action, grad = policy_gradient(state, weight)
            next_state, reward, done, _ = env.step(action)
            next_state = next_state[None, :]
            episode_rewards.append(reward)
            episode_gradients.append(grad)
            state = next_state
        returns = []
        for reward in reversed(episode_rewards):
            G = 0
            G = reward + gamma * G
            returns.append(G)
        for t in range(len(episode_rewards)):
            weight += alpha * episode_gradients[t] * returns[t]
        score = sum(episode_rewards)
        scores.append(score)
        print(f"Episode: {episode + 1}/{nb_episodes}, Score: {score}", end="\r", flush=True)
    return scores
