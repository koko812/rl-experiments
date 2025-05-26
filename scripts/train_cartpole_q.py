import gymnasium as gym
from agents.q_learning_agent import QLearningAgent
import numpy as np
import json
from pathlib import Path
from tqdm import trange

# ==== ハイパーパラメータ ====
NUM_EPISODES = 10000
MAX_STEPS = 500
ENV_NAME = "CartPole-v1"
INITIAL_EPSILON = 1.0
MIN_EPSILON = 0.01
DECAY_RATE = 0.0005   # ← 緩やかに減衰させる

OUTPUT_LOG = (
    Path(__file__).resolve().parent.parent
    / "logs"
    / "cartpole"
    / "q_learning_training.json"
)

# ==== 環境とエージェント初期化 ====
env = gym.make(ENV_NAME)
agent = QLearningAgent(n_actions=env.action_space.n)
agent.epsilon = INITIAL_EPSILON

# ==== ログ用 ====
episode_rewards = []
epsilon_values = []

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

    # ε減衰
    #agent.epsilon = max(agent.epsilon * 0.995, 0.01)
    agent.epsilon = MIN_EPSILON + (INITIAL_EPSILON - MIN_EPSILON) * np.exp(-DECAY_RATE * episode)

    # ログ
    episode_rewards.append(total_reward)
    epsilon_values.append(agent.epsilon)

# ==== ログ保存 ====
OUTPUT_LOG.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_LOG, "w") as f:
    json.dump({
        "rewards": episode_rewards,
        "epsilons": epsilon_values
    }, f, indent=2)

print(f"✅ Training complete! Rewards and epsilons logged to: {OUTPUT_LOG}")
