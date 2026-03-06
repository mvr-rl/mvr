import pathlib
from typing import Any, Dict, Optional, Tuple

import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.envs.mujoco.humanoid_v4 import HumanoidEnv as GymHumanoidEnv
from gymnasium.envs.mujoco.humanoid_v4 import DEFAULT_CAMERA_CONFIG
from gymnasium.spaces import Box
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer, OffScreenViewer, WindowViewer
import mujoco
from itertools import cycle

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,
    "distance": 4.0,
    "lookat": np.array((0.0, 0.0, 1.0)),
    "elevation": 0.0,
    "azimuth": 90,
}
class CustomizedHumanoidEnv(GymHumanoidEnv):
    def __init__(
        self,
        render_mode: str = "rgb_array",
        forward_reward_weight: float = 1.25,
        ctrl_cost_weight: float = 0.1,
        healthy_reward: float = 5.0,
        healthy_z_range: Tuple[float] = (1.0, 2.0),
        reset_noise_scale: float = 1e-2,
        exclude_current_positions_from_observation: bool = True,
        camera_config: Optional[Dict[str, Any]] = DEFAULT_CAMERA_CONFIG,
        textured: bool = True,
        rotating_view: bool = False,
        **kwargs,
    ):
        terminate_when_unhealthy = True
        utils.EzPickle.__init__(
            self,
            forward_reward_weight,
            ctrl_cost_weight,
            healthy_reward,
            terminate_when_unhealthy,
            healthy_z_range,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            render_mode=render_mode,
            **kwargs,
        )

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        if exclude_current_positions_from_observation:
            observation_space = Box(
                low=-np.inf, high=np.inf, shape=(376,), dtype=np.float64
            )
        else:
            observation_space = Box(
                low=-np.inf, high=np.inf, shape=(378,), dtype=np.float64
            )
        env_file_name = None
        if textured:
            env_file_name = "humanoid_textured.xml"
        else:
            env_file_name = "humanoid.xml"
        model_path = str(pathlib.Path(__file__).parent / env_file_name)
        

        MujocoEnv.__init__(
            self,
            model_path,
            5,
            observation_space=observation_space,
            default_camera_config=camera_config,
            render_mode=render_mode,
            **kwargs,
        )

        if hasattr(self, 'mujoco_renderer'):
            self.mujoco_renderer._get_viewer = self.custom_get_viewer

        self.camera_azimuths = [90]
        self.set_view(0)

    @property
    def n_views(self):
        return len(self.camera_azimuths)
    
    def reset(
        self,
        *args, **kwargs):
        # if self.mujoco_renderer.viewer is not None:
        #     azimuth = next(self.camera_azimuths)
        #     self.mujoco_renderer.default_cam_config["azimuth"] = azimuth
        #     self.mujoco_renderer._set_cam_config()
        return super().reset(*args, **kwargs)
    
    def set_view(self, view_ind):
        if self.mujoco_renderer.viewer is None:
            self.render()
        self.mujoco_renderer.default_cam_config["azimuth"] = self.camera_azimuths[view_ind]
        self.mujoco_renderer._set_cam_config()
        
    def custom_get_viewer(self, render_mode: str):
        self.mujoco_renderer.viewer = self.mujoco_renderer._viewers.get(render_mode)
        if self.mujoco_renderer.viewer is None:
            if render_mode == "human":
                self.mujoco_renderer.viewer = WindowViewer(self.mujoco_renderer.model, self.mujoco_renderer.data)
            elif render_mode in {"rgb_array", "depth_array"}:
                self.mujoco_renderer.viewer = CustomOffScreenViewer(self.mujoco_renderer.model, self.mujoco_renderer.data)
            else:
                raise AttributeError(f"Unexpected mode: {render_mode}, expected modes: human, rgb_array, or depth_array")
            self.mujoco_renderer._set_cam_config()
            self.mujoco_renderer._viewers[render_mode] = self.mujoco_renderer.viewer

        if len(self.mujoco_renderer._viewers.keys()) > 1:
            self.mujoco_renderer.viewer.make_context_current()

        return self.mujoco_renderer.viewer
    
class CustomOffScreenViewer(OffScreenViewer):
    def render(
        self,
        render_mode: str,
        camera_id: Optional[int] = None,
        segmentation: bool = False,
    ):
        if camera_id is not None:
            if camera_id == -1:
                self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            else:
                self.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            self.cam.fixedcamid = camera_id

        mujoco.mjv_updateScene(
            self.model,
            self.data,
            self.vopt,
            self.pert,
            self.cam,
            mujoco.mjtCatBit.mjCAT_ALL,
            self.scn,
        )

        if segmentation:
            self.scn.flags[mujoco.mjtRndFlag.mjRND_SEGMENT] = 1
            self.scn.flags[mujoco.mjtRndFlag.mjRND_IDCOLOR] = 1

        for marker_params in self._markers:
            self._add_marker_to_scene(marker_params)

        mujoco.mjr_render(self.viewport, self.scn, self.con)

        for gridpos, (text1, text2) in self._overlays.items():
            mujoco.mjr_overlay(
                mujoco.mjtFontScale.mjFONTSCALE_150,
                gridpos,
                self.viewport,
                text1.encode(),
                text2.encode(),
                self.con,
            )

        if segmentation:
            self.scn.flags[mujoco.mjtRndFlag.mjRND_SEGMENT] = 0
            self.scn.flags[mujoco.mjtRndFlag.mjRND_IDCOLOR] = 0

        rgb_arr = np.zeros(
            3 * self.viewport.width * self.viewport.height, dtype=np.uint8
        )
        depth_arr = np.zeros(
            self.viewport.width * self.viewport.height, dtype=np.float32
        )

        mujoco.mjr_readPixels(rgb_arr, depth_arr, self.viewport, self.con)

        if render_mode == "depth_array":
            depth_img = depth_arr.reshape(self.viewport.height, self.viewport.width)
            # original image is upside-down, so flip it
            return depth_img[::-1, :]
        else:
            rgb_img = rgb_arr.reshape(self.viewport.height, self.viewport.width, 3)

            if segmentation:
                seg_img = (
                    rgb_img[:, :, 0]
                    + rgb_img[:, :, 1] * (2**8)
                    + rgb_img[:, :, 2] * (2**16)
                )
                seg_img[seg_img >= (self.scn.ngeom + 1)] = 0
                seg_ids = np.full(
                    (self.scn.ngeom + 1, 2), fill_value=-1, dtype=np.int32
                )

                for i in range(self.scn.ngeom):
                    geom = self.scn.geoms[i]
                    if geom.segid != -1:
                        seg_ids[geom.segid + 1, 0] = geom.objtype
                        seg_ids[geom.segid + 1, 1] = geom.objid
                rgb_img = seg_ids[seg_img]

            # original image is upside-down, so flip i
            return rgb_img[::-1, :, :]
