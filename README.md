# 🧠 RL Experiment: CartPole Logging & Visualization

This module is part of the **rl-experiment** project, which provides a flexible framework for reinforcement learning experiments across multiple environments (e.g., CartPole, LunarLander).

This component focuses on recording and visualizing episodes of the `CartPole-v1` environment using `gymnasium`.

---

## 📦 Features

* 🎮 Random-agent runner for CartPole
* 📝 Logging of `obs`, `action`, `reward`, `done` per step
* 💾 Supports output format switching between JSON and CSV
* 📊 Visualization of:

  * Pole angle over time (with episode boundary lines)
  * Cumulative reward within each episode (reset after `done=True`)
* 📁 Structured output under `logs/cartpole/`

---

## 🛠 Directory Structure

```
rl-experiment/
├── scripts/
│   ├── run_cartpole_log.py       # logging script
│   └── plot_cartpole_log.py      # visualization script
├── logs/
│   └── cartpole/
│       └── log.json              # logged output
```

> This layout is extendable to other environments like `logs/lunarlander/`, `scripts/train_lunarlander.py`, etc.

---

## 🚀 How to Run

### 1. Generate Log

```bash
uv run scripts/run_cartpole_log.py
```

### 2. Visualize Log

```bash
uv run scripts/plot_cartpole_log.py
```

> Output: A side-by-side graph of pole angle and per-episode reward trajectory.

---

## ⚙️ Customization

### Change log format

Inside `run_cartpole_log.py`:

```python
LOG_MODE = "json"  # or "csv"
```

### Log storage path

Stored in: `logs/cartpole/log.json` (or `.csv`)

---

## 📊 Sample Output

* **Pole Angle Plot**: Blue line with red vertical bars indicating `done=True`
* **Cumulative Reward Plot**: Green line showing reward accumulation within each episode

---

## 📌 Tips for Analysis

* Each episode ends when `done=True`, which corresponds to the pole falling or reaching step limit
* Cumulative reward plot helps evaluate average episode performance
* Log data can be reused for further analysis in Jupyter notebooks

---

## 🧩 Future Extensions

* Add support for rule-based and learned agents (e.g., Q-learning, DQN)
* Expand logging to LunarLander and other environments
* Include value estimates, Q-values, or custom metrics
* Export visualizations as PNG or GIF

---

Happy logging and analyzing! 🚀
