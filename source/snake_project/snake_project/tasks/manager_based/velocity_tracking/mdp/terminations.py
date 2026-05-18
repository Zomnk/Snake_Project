from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def invalid_state(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_root_lin_vel: float = 5.0,
    max_root_ang_vel: float = 10.0,
    min_root_height: float = -0.2,
    max_root_height: float = 0.5,
) -> torch.Tensor:
    """Terminate when the root state becomes invalid or unreasonably large."""
    asset: Articulation = env.scene[asset_cfg.name]
    root_pos = asset.data.root_pos_w
    root_lin_vel = asset.data.root_lin_vel_w
    root_ang_vel = asset.data.root_ang_vel_w

    finite = (
        torch.isfinite(root_pos).all(dim=1)
        & torch.isfinite(root_lin_vel).all(dim=1)
        & torch.isfinite(root_ang_vel).all(dim=1)
    )

    height = root_pos[:, 2]
    lin_speed = torch.linalg.norm(root_lin_vel, dim=1)
    ang_speed = torch.linalg.norm(root_ang_vel, dim=1)

    return (
        ~finite
        | (height < min_root_height)
        | (height > max_root_height)
        | (lin_speed > max_root_lin_vel)
        | (ang_speed > max_root_ang_vel)
    )
