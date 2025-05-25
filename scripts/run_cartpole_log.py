import gymnasium as gym
import json
import csv
import os
from pathlib import Path
import argparse
from agents.rule_agent import SimpleRuleAgent, AngleOnlyAgent, PredictiveAngleAgent

# ===== 引数 =====
parser = argparse.ArgumentParser()
parser.add_argument("--rule", type=str, default="simple", help="使用するルールエージェント名")
parser.add_argument("--logmode", type=str, default="json", choices=["json", "csv"], help="出力フォーマット")
args = parser.parse_args()

# ===== エージェントの切り替え =====
if args.rule == "simple":
    agent = SimpleRuleAgent()
elif args.rule == "angle_only":
    agent = AngleOnlyAgent()
elif args.rule == "predictive":
    agent = PredictiveAngleAgent()
else:
    raise ValueError(f"Unknown rule: {args.rule}")

# ===== 環境とログ準備 =====
env = gym.make("CartPole-v1")
obs, _ = env.reset()
log_data = []

log_path = Path(__file__).resolve().parent.parent / "logs" / "cartpole" / f"{args.rule}_log.{args.logmode}"
os.makedirs(log_path.parent, exist_ok=True)

# ===== プレイ & ログ =====
for step in range(500):
    action = agent.select_action(obs)
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

env.close()

# ===== 保存 =====
if args.logmode == "json":
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2)
    print(f"✅ Log saved to {log_path}")

elif args.logmode == "csv":
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "obs0", "obs1", "obs2", "obs3", "action", "reward", "done"])
        for entry in log_data:
            writer.writerow([entry["step"], *entry["obs"], entry["action"], entry["reward"], entry["done"]])
    print(f"✅ Log saved to {log_path}")
