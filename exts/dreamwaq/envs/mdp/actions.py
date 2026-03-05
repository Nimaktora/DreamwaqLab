import torch
from isaaclab.envs.mdp import JointPositionAction, JointPositionActionCfg
from isaaclab.utils import configclass

class JointPositionActionWithHistory(JointPositionAction):
    """
    Joint position action that tracks history up to t-2.
    Required for smoothness reward calculation (u_t - 2*u_{t-1} + u_{t-2}).
    """
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._prev_prev_action = torch.zeros_like(self._raw_actions)
        self._prev_action = torch.zeros_like(self._raw_actions)
        num_joints = len(self._joint_names)
        print(self._joint_names)

    @property
    def action(self):
        """Current raw action (alias for raw_actions to match reward function)."""
        return self.raw_actions

    @property
    def prev_action(self):
        """Previous action (t-1)."""
        return self._prev_action
    
    @property
    def prev_prev_action(self):
        """Previous previous action (t-2)."""
        return self._prev_prev_action

    def process_actions(self, actions: torch.Tensor):
        self._prev_prev_action[:] = self._prev_action[:]
        self._prev_action[:] = self._raw_actions[:]
        super().process_actions(actions)

    def apply_actions(self):
        super().apply_actions()

    def reset_idx(self, env_ids: torch.Tensor | None = None):
        super().reset_idx(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        
        self._prev_prev_action[env_ids] = 0.0
        self._prev_action[env_ids] = 0.0

@configclass
class JointPositionActionWithHistoryCfg(JointPositionActionCfg):
    """Configuration for the custom action term."""
    class_type = JointPositionActionWithHistory



