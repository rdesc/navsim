import json
import os
import pickle
from typing import Optional

from matplotlib import pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline, interp1d
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import AgentInput, SensorConfig, Trajectory


def interp_trajectory(
    traj, in_dt=1.0, out_dt=0.25, end_time=5.0,
    min_disp=0.1, min_step=0.2, no_motion_tol=0.1,
    history=False, compute_heading=False, debug=False
):
    orig_traj = np.array(traj)
    traj = orig_traj[:, :2].copy()  # ensure shape (N, 2)
    num_points = traj.shape[0]
    if history:
        t_tgt = np.arange(0, end_time + 1e-6, out_dt)
    else:
        t_tgt = np.arange(out_dt, end_time + 1e-6, out_dt)
    out_len = len(t_tgt)

    # --- check for degenerate trajectories ---
    total_disp = 0.0
    if num_points > 1:
        total_disp = np.linalg.norm(traj[0]) if history else np.linalg.norm(traj[-1])
    if num_points < 2 or total_disp < min_disp:
        print(f"Degenerate trajectory detected. num_points: {num_points}, total_disp: {total_disp:.2f} m")
        out = np.zeros((out_len, 2), dtype=np.float32) if not compute_heading else np.zeros((out_len, 3), dtype=np.float32)

    else:
        # --- build source time grid ---
        if history:
            traj_with_anchor = traj.copy()  # already has (0,0) at the end
            t_src = np.arange(0, num_points * in_dt, in_dt)
        else:
            traj_with_anchor = np.vstack([[0, 0], traj])
            t_src = np.arange(0, (num_points + 1) * in_dt, in_dt)

        # --- find transition point ---
        diffs = np.linalg.norm(np.diff(traj_with_anchor, axis=0), axis=1)
        close_idx = np.where(diffs < min_step)[0]
        if len(close_idx) > 0:
            cutoff = close_idx[0] + 1
        else:
            cutoff = len(traj_with_anchor)

        # --- interpolate ---
        out = np.zeros((out_len, 2), dtype=np.float32)

        # cubic spline part
        if cutoff > 2:  # need at least 3 points for cubic spline
            cs_x = CubicSpline(t_src[:cutoff], traj_with_anchor[:cutoff, 0], bc_type="natural")
            cs_y = CubicSpline(t_src[:cutoff], traj_with_anchor[:cutoff, 1], bc_type="natural")
            mask = t_tgt <= t_src[cutoff - 1]
            out[mask] = np.stack([cs_x(t_tgt[mask]), cs_y(t_tgt[mask])], axis=-1)
        else:
            cutoff = 1  # fallback to linear if not enough points

        # linear part
        if cutoff < len(traj_with_anchor):
            lin_x = interp1d(t_src[cutoff - 1:], traj_with_anchor[cutoff - 1:, 0], kind="linear")
            lin_y = interp1d(t_src[cutoff - 1:], traj_with_anchor[cutoff - 1:, 1], kind="linear")
            mask = t_tgt >= t_src[cutoff - 1]
            out[mask] = np.stack([lin_x(t_tgt[mask]), lin_y(t_tgt[mask])], axis=-1)

        if compute_heading:
            diffs = np.gradient(out[:, :2], axis=0)
            headings = np.arctan2(diffs[:, 1], diffs[:, 0])

            # compute step sizes (displacement per step)
            step_sizes = np.linalg.norm(diffs, axis=1)

            # forward fill for indices with tiny displacement
            for i in range(1, len(headings)):
                if step_sizes[i] < no_motion_tol:
                    headings[i] = headings[i - 1]

            # backward fill in case the trajectory *starts* with tiny displacements
            for i in range(len(headings) - 2, -1, -1):
                if step_sizes[i] < no_motion_tol:
                    headings[i] = headings[i + 1]

            out = np.hstack([out, headings[:, None]])

    # --- visualization for debugging ---
    if debug:
        print("Orig traj shape:", orig_traj.shape, "Out shape:", out.shape)
        print("Interpolated trajectory:\n", out)
        plt.plot(orig_traj[:, 0], orig_traj[:, 1], 'o', color='red', label='Original Traj')
        plt.quiver(orig_traj[:, 0], orig_traj[:, 1], np.cos(orig_traj[:, 2]), np.sin(orig_traj[:, 2]),
                   angles='xy', scale_units='xy', scale=0.1, color='red', alpha=0.4)
        plt.plot(out[:, 0], out[:, 1], '.', color='black', label='Interpolated Traj')
        if compute_heading:
            plt.quiver(out[:, 0], out[:, 1], np.cos(out[:, 2]), np.sin(out[:, 2]),
                       angles='xy', scale_units='xy', scale=0.01, color='blue', alpha=0.4)
        plt.title(f'Total displacement: {total_disp:.2f} m')
        plt.legend()
        plt.axis('equal')
        plt.show()

    return out


class PoutineAgent(AbstractAgent):
    """Poutine agent interface."""

    def __init__(
        self,
        load_predictions_from_file: Optional[str] = None,
        cache_dataset_to_file: Optional[str] = None,
        jpeg_root_paths: Optional[str] = None,
        trajectory_sampling: TrajectorySampling = TrajectorySampling(time_horizon=4, interval_length=0.25),
    ):
        """
        Initializes the agent interface for EgoStatusMLP.
        :param trajectory_sampling: trajectory sampling specification.
        """
        super().__init__(trajectory_sampling)
        self._load_predictions_from_file = load_predictions_from_file
        self._cache_dataset_to_file = cache_dataset_to_file
        self._jpeg_root_paths = jpeg_root_paths
        
        self._frame_indices = [0, 1, 2, 3]
        self._predictions = None
        self._prediction_count = 0

        if self._cache_dataset_to_file:
            print(f"Caching evaluation dataset to {self._cache_dataset_to_file}... Will not be loading predictions from file.")
            # check if file exists and is not empty
            if os.path.exists(self._cache_dataset_to_file) and os.path.getsize(self._cache_dataset_to_file) > 0:
                with open(self._cache_dataset_to_file, 'r') as f:
                    num_lines = sum(1 for _ in f)
                raise FileExistsError(
                    f"Cache dataset file {self._cache_dataset_to_file} already exists with {num_lines} lines."
                )
            with open(self._cache_dataset_to_file, 'w') as f:
                f.write('')
            self._load_predictions_from_file = None  # disable loading predictions if caching
        elif self._load_predictions_from_file and not os.path.exists(self._load_predictions_from_file):
            raise FileNotFoundError(f"Could not find predictions file {self._load_predictions_from_file}")
        
        assert self._load_predictions_from_file or self._cache_dataset_to_file, \
            "Either load_predictions_from_file or cache_dataset_to_file must be provided."

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def initialize(self) -> None:
        """Inherited, see superclass."""
        if self._load_predictions_from_file:
            self._predictions = pickle.load(open(self._load_predictions_from_file, "rb"))
            print(f"Loaded predictions from {self._load_predictions_from_file}, total {len(self._predictions)} entries.")

    def get_sensor_config(self) -> SensorConfig:
        """Inherited, see superclass."""
        return SensorConfig(
            # NOTE: rdesc: there's a bug with loading `private_test_hard_two_stage` which gives OOM error when loading all cameras
            cam_f0=self._frame_indices,
            cam_l0=self._frame_indices,
            cam_l1=False,  # self._frame_indices,
            cam_l2=False,  # self._frame_indices,
            cam_r0=self._frame_indices,
            cam_r1=False,  # self._frame_indices,
            cam_r2=False,  # self._frame_indices,
            cam_b0=False,  # self._frame_indices,
            lidar_pc=False,
        )

    def compute_trajectory(self, agent_input: AgentInput) -> Trajectory:
        """Inherited, see superclass."""
        if 'token' not in dir(agent_input):
            raise ValueError("AgentInput must have 'token' attribute set to uniquely identify the input instance for PoutineAgent.")
        
        uuid = agent_input.token
        pred_dict_key = f"{uuid}-3"  # seq is always 3
        
        if self._predictions is not None:
            if pred_dict_key not in self._predictions:
                raise ValueError(f"Could not find predictions for token {uuid} in loaded predictions.")
            pred = interp_trajectory(self._predictions[pred_dict_key],
                                    in_dt=0.25,
                                    out_dt=self._trajectory_sampling.interval_length,
                                    end_time=self._trajectory_sampling.time_horizon,
                                    compute_heading=True)
            self._prediction_count += 1
            print(f"Loaded prediction for token {uuid} from file. Total loaded: {self._prediction_count}")
            
            return Trajectory(pred, self._trajectory_sampling)

        ego_velocity_2d = agent_input.ego_statuses[-1].ego_velocity
        ego_acceleration_2d = agent_input.ego_statuses[-1].ego_acceleration
        
        history_trajectory = []
        for frame_idx in range(len(agent_input.ego_statuses)):
            history_trajectory.append(agent_input.ego_statuses[frame_idx].ego_pose.tolist())
            
        future_trajectory = None
        if 'future_traj' in dir(agent_input):
            future_trajectory = agent_input.future_traj

        intent_idx = np.argmax(agent_input.ego_statuses[-1].driving_command)
        intent = ["GO_LEFT", "GO_STRAIGHT", "GO_RIGHT", "UNKNOWN"][intent_idx]
        
        jpeg_frames = []
        assert len(agent_input.cameras) == 4
       
        # get the jpeg paths
        for idx in self._frame_indices:
            frame = agent_input.cameras[idx]
            jpegs_dict = {}
            for cam, cam_name in zip(
                [frame.cam_f0, frame.cam_l0, frame.cam_r0,
                 frame.cam_l1, frame.cam_r1, frame.cam_l2,
                 frame.cam_b0, frame.cam_r2],
                ["FRONT", "FRONT_LEFT", "FRONT_RIGHT", "SIDE_LEFT", "SIDE_RIGHT",
                 "REAR_LEFT", "REAR", "REAR_RIGHT"]
            ):
                for root in self._jpeg_root_paths:  # could either be in original or synthetic sensor path
                    curr_path = f"{root}/{cam.camera_path}"
                    if os.path.exists(curr_path):
                        jpegs_dict[cam_name] = curr_path.split('dataset/')[-1]
                        break
            jpeg_frames.append(jpegs_dict)
        
        # add to cache dataset file
        with open(self._cache_dataset_to_file, 'a') as f:
            f.write(json.dumps({
                "uuid": uuid,
                "seq": 3,
                "intent": intent,
                "history_traj_orig": history_trajectory,  # only 1.5 seconds of history
                "future_traj_orig": future_trajectory.tolist() if future_trajectory is not None else None,
                "history_traj": interp_trajectory(history_trajectory, in_dt=0.5, out_dt=0.25, end_time=1.5, history=True).tolist(),
                "future_traj": (
                    interp_trajectory(future_trajectory, in_dt=0.5, out_dt=0.25, end_time=4, history=False).tolist()
                    if future_trajectory is not None else None
                    ),
                "vel": [ego_velocity_2d.tolist()],
                "accel": [ego_acceleration_2d.tolist()],
                "jpeg_paths": jpeg_frames
            }))
            f.write('\n')
        
        self._prediction_count += 1
        print(f"Cached data for token {uuid}. Total cached: {self._prediction_count}")

        predictions_dummy = np.zeros((self._trajectory_sampling.num_poses, 3), dtype=np.float32)
        
        return Trajectory(predictions_dummy, self._trajectory_sampling)