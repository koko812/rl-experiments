import gymnasium as gym
import json
import csv
import os
from pathlib import Path

LOG_MODE = "json"  # ← "csv" にするとCSV出力
log_path = Path(__file__).resolve().parent.parent / "logs" / "cartpole" / "log.json"

env = gym.make("CartPole-v1")
obs, _ = env.reset()
log_data = []

for step in range(500):
    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

    log_data.append({
        "step": step,
        "obs": obs.tolist(),
        "action": int(action),
        "reward": float(reward),
        "done": bool(done)
    })

    obs = next_obs
    if done:
        obs, _ = env.reset()

# --- 保存先ディレクトリ確認 ---
os.makedirs("logs", exist_ok=True)

# --- 書き出し ---
if LOG_MODE == "json":
    os.makedirs(log_path.parent, exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2)
    print("✅ Log saved to logs/cartpole_log.json")

elif LOG_MODE == "csv":
    with open("../logs/cartpole_log.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "obs0", "obs1", "obs2", "obs3", "action", "reward", "done"])
        for entry in log_data:
            writer.writerow([entry["step"], *entry["obs"], entry["action"], entry["reward"], entry["done"]])
    print("✅ Log saved to logs/cartpole_log.csv")

else:
    print("❌ Unknown LOG_MODE:", LOG_MODE)

env.close()
