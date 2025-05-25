import json
import argparse
import matplotlib.pyplot as plt
from pathlib import Path

# ===== ログフォルダを定義 =====
log_dir = Path(__file__).resolve().parent.parent / "logs" / "cartpole"

# ===== 利用可能なログ一覧を取得 =====
available_logs = [f.name for f in log_dir.glob("*.json")]

# ===== 引数処理 =====
parser = argparse.ArgumentParser()
parser.add_argument("--log", type=str, default="log.json", help="ログファイル名（logs/cartpole/内）")
args = parser.parse_args()

log_filename = Path(args.log).name

# ===== 存在チェック =====
if log_filename not in available_logs:
    print("❌ 指定されたログファイルが見つかりません。利用可能なファイル：")
    for f in available_logs:
        print(f" - {f}")
    exit(1)

log_path = log_dir / log_filename

# ===== ログ読み込み =====
with open(log_path) as f:
    log_data = json.load(f)

# ===== データ抽出 =====
steps = [entry["step"] for entry in log_data]
angles = [entry["obs"][2] for entry in log_data]
done_flags = [entry["done"] for entry in log_data]

# 累積報酬
cumulative_rewards = []
total = 0
for entry in log_data:
    total += entry["reward"]
    cumulative_rewards.append(total)
    if entry["done"]:
        total = 0

# ===== プロット =====
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(steps, angles)
plt.title(f"Pole Angle over Time ({log_filename})")
plt.xlabel("Step")
plt.ylabel("Angle (radians)")
for i, done in enumerate(done_flags):
    if done:
        plt.axvline(x=steps[i], color='red', linestyle='--', alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(steps, cumulative_rewards, color='green')
plt.title(f"Cumulative Reward over Time ({log_filename})")
plt.xlabel("Step")
plt.ylabel("Cumulative Reward")
plt.grid(True)

plt.tight_layout()
plt.show()
