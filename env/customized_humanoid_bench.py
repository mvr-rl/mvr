import os

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from dm_control.mujoco import index
from dm_control.mujoco.engine import NamedIndexStructs
from gymnasium.envs.mujoco.mujoco_rendering import  OffScreenViewer, WindowViewer
from humanoid_bench.dmc_deps.dmc_wrapper import MjDataWrapper, MjModelWrapper
from humanoid_bench.wrappers import (
    SingleReachWrapper,
    DoubleReachAbsoluteWrapper,
    DoubleReachRelativeWrapper,
    BlockedHandsLocoWrapper,
    ObservationWrapper,
)
DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,
    "distance": 3.0, #5.0,
    "lookat": np.array((0.0, 0.0, 1.0)),
    "elevation": 0, #-20.0,
    "azimuth": 90,
}
DEFAULT_RANDOMNESS = 0.01

from humanoid_bench.env import TASKS, ROBOTS
from humanoid_bench.envs.kitchen import Kitchen
from humanoid_bench.envs.cube import Cube
from humanoid_bench.envs.bookshelf import BookshelfSimple, BookshelfHard
from humanoid_bench import env as orginal_env


class HumanoidEnv(orginal_env.HumanoidEnv):
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 50,
    }

    def __init__(
        self,
        robot=None,
        control=None,
        task=None,
        render_mode="rgb_array",
        width=256,
        height=256,
        randomness=DEFAULT_RANDOMNESS,
        **kwargs,
    ):
        assert robot and control and task, f"{robot} {control} {task}"
        gym.utils.EzPickle.__init__(self, metadata=self.metadata)

        asset_path = os.path.join(os.path.dirname(orginal_env.__file__), "assets")
        model_path = f"envs/{robot}_{control}_{task}.xml"
        model_path = os.path.join(asset_path, model_path)

        self.robot = ROBOTS[robot](self)
        task_info = TASKS[task](self.robot, None, **kwargs)

        self.obs_wrapper = kwargs.get("obs_wrapper", None)
        if self.obs_wrapper is not None:
            self.obs_wrapper = kwargs.get("obs_wrapper", "False").lower() == "true"
        else:
            self.obs_wrapper = False

        self.blocked_hands = kwargs.get("blocked_hands", None)
        if self.blocked_hands is not None:
            self.blocked_hands = kwargs.get("blocked_hands", "False").lower() == "true"
        else:
            self.blocked_hands = False

        MujocoEnv.__init__(
            self,
            model_path,
            frame_skip=task_info.frame_skip,
            observation_space=task_info.observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            render_mode=render_mode,
            width=width,
            height=height,
            camera_name=task_info.camera_name,
        )

        self.action_high = self.action_space.high
        self.action_low = self.action_space.low
        self.action_space = Box(
            low=-1, high=1, shape=self.action_space.shape, dtype=np.float32
        )
        # Change viewer
        if hasattr(self, 'mujoco_renderer'):
            self.mujoco_renderer._get_viewer = self.custom_get_viewer
        self.camera_azimuths = kwargs.get("camera_azimuths", [90])
        self.task = TASKS[task](self.robot, self, **kwargs)
        # Wrap task to fix render method BEFORE other wrappers
        self.task = TaskWrapper(self.task)
        # Change view
        self.set_view(0)
        if self.blocked_hands:
            self.task = BlockedHandsLocoWrapper(self.task, **kwargs)
            # Re-wrap with TaskWrapper after blocked hands wrapper
            self.task = TaskWrapper(self.task)

        # Wrap for hierarchical control
        if (
            "policy_type" in kwargs
            and kwargs["policy_type"]
            and kwargs["policy_type"] is not None
            and kwargs["policy_type"] != "flat"
        ):
            if kwargs["policy_type"] == "reach_single":
                assert "policy_path" in kwargs and kwargs["policy_path"] is not None
                self.task = SingleReachWrapper(self.task, **kwargs)
            elif kwargs["policy_type"] == "reach_double_absolute":
                assert "policy_path" in kwargs and kwargs["policy_path"] is not None
                self.task = DoubleReachAbsoluteWrapper(self.task, **kwargs)
            elif kwargs["policy_type"] == "reach_double_relative":
                assert "policy_path" in kwargs and kwargs["policy_path"] is not None
                self.task = DoubleReachRelativeWrapper(self.task, **kwargs)
            else:
                raise ValueError(f"Unknown policy_type: {kwargs['policy_type']}")
            # Re-wrap with TaskWrapper after hierarchical control wrapper
            self.task = TaskWrapper(self.task)
        

        if self.obs_wrapper:
            # Note that observation wrapper is not compatible with hierarchical policy
            self.task = ObservationWrapper(self.task, **kwargs)
            # Re-wrap with TaskWrapper after observation wrapper
            self.task = TaskWrapper(self.task)
            self.observation_space = self.task.observation_space

        # Keyframe
        self.keyframe = (
            self.model.key(kwargs["keyframe"]).id if "keyframe" in kwargs else 0
        )

        self.randomness = randomness
        if isinstance(self.task, (BookshelfHard, BookshelfSimple, Kitchen, Cube)):
            self.randomness = 0
        print(isinstance(self.task, (BookshelfHard, BookshelfSimple, Kitchen, Cube)))

        # Set up named indexing.
        data = MjDataWrapper(self.data)
        model = MjModelWrapper(self.model)
        axis_indexers = index.make_axis_indexers(model)
        self.named = NamedIndexStructs(
            model=index.struct_indexer(model, "mjmodel", axis_indexers),
            data=index.struct_indexer(data, "mjdata", axis_indexers),
        )

        assert self.robot.dof + self.task.dof == len(data.qpos), (
            self.robot.dof,
            self.task.dof,
            len(data.qpos),
        )
    @property
    def n_views(self):
        return len(self.camera_azimuths)
    
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
                self.mujoco_renderer.viewer = CustomOffScreenViewer(
                    self.mujoco_renderer.model, 
                    self.mujoco_renderer.data,
                    width=self.mujoco_renderer.width,
                    height=self.mujoco_renderer.height
                )
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
        camera_id = None,
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

# Add Task wrapper to fix render method
class TaskWrapper:
    """Wrapper for humanoid-bench tasks to fix render method incompatibility"""
    
    def __init__(self, task):
        self.task = task
        self._env = task._env
        
    def __getattr__(self, name):
        return getattr(self.task, name)
    
    def render(self):
        # Fixed render method that only passes the render_mode parameter
        return self._env.mujoco_renderer.render(self._env.render_mode)
