def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99):
    for episode in range(episodes):
        state = env.reset()
        episode_h = []
        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            episode_h.append((state, reward))
            if done:
                break
            state = next_state
        G = 0
        for state, reward in reversed(episode_h):
            G = reward + gamma * G
            V[state] += alpha * (G - V[state])
    return V
