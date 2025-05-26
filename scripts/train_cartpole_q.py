import gymnasium as gym
from agents.q_learning_agent import QLearningAgent
import numpy as np
import json
from pathlib import Path
from tqdm import trange  # 進捗バーが便利

# ==== ハイパーパラメータ ====
NUM_EPISODES = 5000
MAX_STEPS = 500
ENV_NAME = "CartPole-v1"
OUTPUT_LOG = (
    Path(__file__).resolve().parent.parent
    / "logs"
    / "cartpole"
    / "q_learning_training.json"
)

# ==== 環境とエージェント初期化 ====
env = gym.make(ENV_NAME)
agent = QLearningAgent(n_actions=env.action_space.n)

# ==== ログ用 ====
episode_rewards = []

# ==== 学習ループ ====
for episode in trange(NUM_EPISODES):
    obs, _ = env.reset()
    total_reward = 0

    for step in range(MAX_STEPS):
        action = agent.select_action(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)

        agent.learn(obs, action, reward, next_obs, done=(terminated or truncated))

        obs = next_obs
        total_reward += reward

        if terminated or truncated:
            break

    agent.epsilon = max(agent.epsilon * 0.995, 0.01)
    episode_rewards.append(total_reward)

# ==== ログ保存 ====
OUTPUT_LOG.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_LOG, "w") as f:
    json.dump(episode_rewards, f)

print(f"✅ Training complete! Rewards logged to: {OUTPUT_LOG}")
