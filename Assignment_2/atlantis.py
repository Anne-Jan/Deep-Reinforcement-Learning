import gymnasium as gym

env = gym.make("ALE/Atlantis-v5", render_mode = "human")
observation, info = env.reset(seed=42)

action = 0
prevaction = 0
skipaction = False

for _ in range(1000):
    if prevaction == 0 and not skipaction:
        action = 1 if action == 3 else action + 1
        # observation, reward, terminated, truncated, info = env.step(int(action))
        prevaction = action
    else:
        # observation, reward, terminated, truncated, info = env.step(int(0))
        prevaction = 0

    print(prevaction)

    observation, reward, terminated, truncated, info = env.step(int(prevaction))

    if terminated or truncated:
        observation, info = env.reset(seed=42)

env.close()