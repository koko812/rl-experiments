class BaseAgent:
    def select_action(self, obs):
        """観測obsに対して行動（0または1）を返す"""
        raise NotImplementedError("select_action() を実装してください")

    def learn(self, obs, action, reward, next_obs, done):
        """Q-Learning系エージェント向け：学習処理"""
        # Rule-basedなどでは空実装でもOK
        pass
