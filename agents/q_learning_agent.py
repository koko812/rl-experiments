import numpy as np
from agents.base_agent import BaseAgent


class QLearningAgent(BaseAgent):
    def __init__(self, n_actions, bins_per_obs=10, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {}

        # 各obs成分の離散化用bin（左右端は CartPole の typical 値）
        self.bins = [
            np.linspace(-2.4, 2.4, bins_per_obs),  # cart position
            np.linspace(-3.0, 3.0, bins_per_obs),  # cart velocity
            np.linspace(-0.2, 0.2, bins_per_obs),  # pole angle
            np.linspace(-2.0, 2.0, bins_per_obs),  # pole angular velocity
        ]

    def _discretize(self, obs):
        """連続obsを離散化して状態タプルに変換"""
        return tuple(np.digitize(o, b) for o, b in zip(obs, self.bins))

    def select_action(self, obs):
        state = self._discretize(obs)

        # Qテーブルに状態がなければ初期化
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.n_actions)

        if np.random.rand() < self.epsilon:
            # 探索：ランダム行動
            return np.random.randint(self.n_actions)
        else:
            # 活用：最大Q値の行動を選択
            return int(np.argmax(self.q_table[state]))

    def learn(self, obs, action, reward, next_obs, done):
        # 離散化
        state = self._discretize(obs)
        next_state = self._discretize(next_obs)

        # 未知状態なら初期化
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.n_actions)
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(self.n_actions)

        # Q値のターゲットを計算
        if done:
            target = reward  # エピソード終了なら future value はない
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state])

        # Qテーブルを更新
        old_value = self.q_table[state][action]
        new_value = old_value + self.alpha * (target - old_value)
        self.q_table[state][action] = new_value

