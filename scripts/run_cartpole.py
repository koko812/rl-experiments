# scripts/run_cartpole.py

import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")
obs, _ = env.reset()

for step in range(500):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, _ = env.step(action)

    print(f"Step {step:3d} | obs: {obs} | action: {action} | reward: {reward} | done: {terminated or truncated}")

    if terminated or truncated:
        obs, _ = env.reset()


env.close()
