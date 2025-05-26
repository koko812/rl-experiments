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
    parser.add_argument("--log", type=str, required=True, help="報酬＆εログのJSONファイルパス")
    parser.add_argument("--title", type=str, default="Q-Learning Training Curve (with ε)", help="グラフタイトル")
    parser.add_argument("--outname", type=str, default="q_learning_curve.png", help="保存ファイル名（docs/figs 以下）")
    args = parser.parse_args()

    # ログ読み込み
    log_path = Path(args.log)
    with open(log_path) as f:
        data = json.load(f)
        rewards = data["rewards"]
        epsilons = data["epsilons"]

    episodes = list(range(len(rewards)))

    # プロット
    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total Reward", color="tab:blue")
    ax1.plot(episodes, rewards, label="Reward (Raw)", color="tab:blue", alpha=0.3)
    ax1.plot(episodes, smooth(rewards), label="Reward (Smoothed)", color="tab:blue")
    ax1.tick_params(axis='y', labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Epsilon", color="tab:red")
    ax2.plot(episodes, epsilons, label="Epsilon", color="tab:red")
    ax2.tick_params(axis='y', labelcolor="tab:red")

    fig.suptitle(args.title)
    fig.legend(loc="upper left")
    fig.tight_layout()
    plt.grid(True)

    # === 保存 ===
    save_path = Path("docs") / "figs" / args.outname
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    print(f"✅ Figure saved to: {save_path}")

    # === 表示 ===
    plt.show()

if __name__ == "__main__":
    main()
