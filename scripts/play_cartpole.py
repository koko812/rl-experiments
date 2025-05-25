import gymnasium as gym
import pygame
import time
import json
import os

# 環境の初期化
env = gym.make("CartPole-v1", render_mode="human")
obs, _ = env.reset()

# Pygame初期化
pygame.init()
screen = pygame.display.set_mode((400, 300))
pygame.display.set_caption("CartPole Controller")

# ログ保存用
log_data = []

action = 0
running = True
step = 0

while running and step < 500:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        action = 0
    elif keys[pygame.K_RIGHT]:
        action = 1
    else:
        action = 0  # 無操作時は左にしておく（適宜変更可）

    next_obs, reward, terminated, truncated, _ = env.step(action)

    log_data.append({
        "step": step,
        "obs": obs.tolist(),
        "action": int(action),
        "reward": float(reward),
        "done": bool(terminated or truncated)
    })

    obs = next_obs
    step += 1

    if terminated or truncated:
        obs, _ = env.reset()

    time.sleep(0.02)

pygame.quit()
env.close()

# ログ保存
os.makedirs("logs/cartpole", exist_ok=True)
with open("logs/cartpole/human_play_log.json", "w") as f:
    json.dump(log_data, f, indent=2)
print("✅ プレイログを logs/cartpole/human_play_log.json に保存しました")
