class BaseRuleAgent:
    def select_action(self, obs):
        raise NotImplementedError


class SimpleRuleAgent(BaseRuleAgent):
    """カート位置とポール角度を両方使う"""
    def select_action(self, obs):
        cart_pos, cart_vel, pole_angle, pole_vel = obs
        return 1 if pole_angle > 0 else 0


class AngleOnlyAgent(BaseRuleAgent):
    """ポール角度だけを見るルール"""
    def select_action(self, obs):
        return 1 if obs[2] > 0 else 0

class PredictiveAngleAgent(BaseRuleAgent):
    """ポール角と角速度から倒れそうな方向を予測"""
    def select_action(self, obs):
        _, _, angle, angle_velocity = obs
        prediction = angle + 0.5 * angle_velocity
        return 1 if prediction > 0 else 0
