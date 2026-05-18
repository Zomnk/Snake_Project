import numpy as np
import mujoco
from typing import List, Optional


class VirtualChassis:
    def __init__(self, model: mujoco.MjModel, body_names: List[str]):
        self._body_ids = np.array([model.body(n).id for n in body_names], dtype=int)
        self._U_prev: Optional[np.ndarray] = None
        self.com: np.ndarray = np.zeros(3)
        self.rotation: np.ndarray = np.eye(3)
        self.com_velocity: np.ndarray = np.zeros(3)
        self.ang_velocity: np.ndarray = np.zeros(3)
        self.axes: List[np.ndarray] = [np.zeros(3) for _ in range(3)]

    def update(self, data: mujoco.MjData, dt: float):
        positions = data.xpos[self._body_ids].copy()

        com = np.mean(positions, axis=0)
        centered = positions - com

        U, s, Vt = np.linalg.svd(centered.T)

        if self._U_prev is not None:
            for k in range(3):
                if np.dot(U[:, k], self._U_prev[:, k]) < 0:
                    U[:, k] *= -1
        else:
            head_to_tail = positions[-1] - positions[0]
            if np.dot(U[:, 0], head_to_tail) < 0:
                U[:, 0] *= -1
            if np.dot(U[:, 2], np.array([0, 0, 1])) < 0:
                U[:, 2] *= -1

        if np.linalg.det(U) < 0:
            U[:, 1] *= -1

        self._U_prev = U.copy()

        body_vels = data.cvel[self._body_ids]
        body_lin_vels_w = body_vels[:, 3:6]
        body_ang_vels_w = body_vels[:, 0:3]

        vc_lin_vel_w = np.mean(body_lin_vels_w, axis=0)
        vc_ang_vel_w = np.mean(body_ang_vels_w, axis=0)

        self.com_velocity = U.T @ vc_lin_vel_w
        self.ang_velocity = U.T @ vc_ang_vel_w

        self.com = com
        self.rotation = U
        self.axes = [U[:, i].copy() for i in range(3)]

    @staticmethod
    def plot_results(traj_base, traj_vc, vel_base, vel_vc, ang_vel_base_z, ang_vel_vc_z, times, save_dir, script_name):
        import os
        import matplotlib.pyplot as plt

        traj_base = np.array(traj_base)
        traj_vc = np.array(traj_vc)
        vel_base = np.array(vel_base)
        vel_vc = np.array(vel_vc)
        ang_vel_base_z = np.array(ang_vel_base_z)
        ang_vel_vc_z = np.array(ang_vel_vc_z)
        times = np.array(times)

        os.makedirs(save_dir, exist_ok=True)

        np.savez(
            os.path.join(save_dir, f"{script_name}_vc_data.npz"),
            traj_base=traj_base, traj_vc=traj_vc,
            vel_base=vel_base, vel_vc=vel_vc,
            ang_vel_base_z=ang_vel_base_z, ang_vel_vc_z=ang_vel_vc_z,
            times=times
        )

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(traj_base[:, 0], traj_base[:, 1], label='base_link', linewidth=2.0, color='#1f77b4')
        ax.plot(traj_vc[:, 0], traj_vc[:, 1], label='Virtual Chassis', linewidth=2.0, color='#d62728')
        ax.scatter(traj_base[0, 0], traj_base[0, 1], marker='o', color='#1f77b4', s=30)
        ax.scatter(traj_base[-1, 0], traj_base[-1, 1], marker='x', color='#1f77b4', s=40)
        ax.scatter(traj_vc[0, 0], traj_vc[0, 1], marker='o', color='#d62728', s=30)
        ax.scatter(traj_vc[-1, 0], traj_vc[-1, 1], marker='x', color='#d62728', s=40)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f'{script_name} - Trajectory')
        ax.set_aspect('equal', adjustable='box')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f"{script_name}_trajectory.png"), dpi=200)
        plt.close(fig)

        fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

        axes[0].plot(times, vel_vc[:, 0], 'r-', label='vc_vx', linewidth=1.5)
        axes[0].plot(times, vel_base[:, 0], 'b--', label='base_vx_w', linewidth=1.5)
        axes[0].set_ylabel('vx (m/s)')
        axes[0].legend(loc='upper right')
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(times, vel_vc[:, 1], 'r-', label='vc_vy', linewidth=1.5)
        axes[1].plot(times, vel_base[:, 1], 'b--', label='base_vy_w', linewidth=1.5)
        axes[1].set_ylabel('vy (m/s)')
        axes[1].legend(loc='upper right')
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(times, ang_vel_vc_z, 'r-', label='vc_wz', linewidth=1.5)
        axes[2].plot(times, ang_vel_base_z, 'b--', label='base_wz_w', linewidth=1.5)
        axes[2].set_ylabel('wz (rad/s)')
        axes[2].set_xlabel('Time (s)')
        axes[2].legend(loc='upper right')
        axes[2].grid(True, alpha=0.3)

        fig.suptitle(f'{script_name} - Velocity')
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f"{script_name}_velocity.png"), dpi=200)
        plt.close(fig)
