import matplotlib.pyplot as plt
import numpy as np
import pickle
from agents.q_learning_agent import QLearningAgent

# ==== パラメータ ====
bins_per_obs = 10
obs_idx_angle = 2       # 角度
obs_idx_angvel = 3      # 角速度

# ==== QLearningAgentから bins を拝借 ====
dummy_agent = QLearningAgent(n_actions=2, bins_per_obs=bins_per_obs)
angle_bins = dummy_agent.bins[obs_idx_angle]
angvel_bins = dummy_agent.bins[obs_idx_angvel]

# ==== Qテーブルの読み込み ====
with open("logs/cartpole/final_q_table.pkl", "rb") as f:
    q_table = pickle.load(f)

# ==== 可視化マトリクス作成 ====
heatmap = np.full((bins_per_obs, bins_per_obs), fill_value=-1, dtype=int)

for i in range(bins_per_obs):         # angle_bin
    for j in range(bins_per_obs):     # ang_vel_bin
        state = (0, 0, i, j)  # cart位置・速度は固定
        if state in q_table:
            action = np.argmax(q_table[state])
            heatmap[i, j] = action

# ==== ヒートマップ表示 ====
# ==== ヒートマップ表示（with 実数ラベル） ====
plt.figure(figsize=(7, 6))
cmap = plt.cm.get_cmap("bwr", 2)
im = plt.imshow(heatmap, cmap=cmap, origin="lower")

plt.title("Q-Policy: Best Action per (angle, angular velocity)")
plt.xlabel("Angular Velocity")
plt.ylabel("Pole Angle")

# 軸ラベルに bins の実数値を使用
xticks = np.arange(bins_per_obs)
xlabels = [f"{x:.2f}" for x in dummy_agent.bins[3]]  # angular velocity
plt.xticks(xticks, xlabels, rotation=45)

yticks = np.arange(bins_per_obs)
ylabels = [f"{x:.2f}" for x in dummy_agent.bins[2]]  # pole angle
plt.yticks(yticks, ylabels)

plt.colorbar(im, ticks=[0, 1], label="Action (0=Left, 1=Right)")
plt.grid(False)
plt.tight_layout()
plt.savefig('docs/figs/visualized_q_table.png')
plt.show()
