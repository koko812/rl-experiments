import json
import matplotlib.pyplot as plt
from pathlib import Path

# ログパスの取得
log_path = Path(__file__).resolve().parent.parent / "logs" / "cartpole" / "log.json"

# ログ読み込み
with open(log_path) as f:
    log_data = json.load(f)

# データ抽出
steps = [entry["step"] for entry in log_data]
angles = [entry["obs"][2] for entry in log_data]
done_flags = [entry["done"] for entry in log_data]

# ステップ単位での累積報酬（エピソードごとにリセット）
cumulative_rewards = []
total = 0
for entry in log_data:
    total += entry["reward"]
    cumulative_rewards.append(total)
    if entry["done"]:
        total = 0  # エピソード終了時にリセット

# 可視化
plt.figure(figsize=(12, 6))

# 角度の推移
plt.subplot(1, 2, 1)
plt.plot(steps, angles)
plt.title("Pole Angle over Time")
plt.xlabel("Step")
plt.ylabel("Angle (radians)")
for i, done in enumerate(done_flags):
    if done:
        plt.axvline(x=steps[i], color='red', linestyle='--', alpha=0.3)

# 累積報酬の推移（ステップ単位）
plt.subplot(1, 2, 2)
plt.plot(steps, cumulative_rewards, color='green')
plt.title("Cumulative Reward over Time")
plt.xlabel("Step")
plt.ylabel("Cumulative Reward")
plt.grid(True)

plt.tight_layout()
plt.show()
