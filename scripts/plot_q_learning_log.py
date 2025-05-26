import json
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

def smooth(data, window=20):
    return [
        sum(data[max(0, i - window):i + 1]) / (i - max(0, i - window) + 1)
        for i in range(len(data))
    ]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=str, required=True, help="報酬ログのJSONファイルパス")
    parser.add_argument("--out", type=str, default="training_curve.png", help="保存先PNGファイル名")
    parser.add_argument("--title", type=str, default="Q-Learning Training Curve", help="グラフタイトル")
    args = parser.parse_args()

    # ログ読み込み
    log_path = Path(args.log)
    with open(log_path) as f:
        rewards = json.load(f)

    # 描画
    plt.figure(figsize=(12, 6))
    plt.plot(rewards, label="Raw")
    plt.plot(smooth(rewards), label="Smoothed", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title(args.title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # 保存
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    print(f"✅ Saved plot to: {out_path}")

if __name__ == "__main__":
    main()
