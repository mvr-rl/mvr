from gymnasium.envs.registration import register
from metaworld.env_dict import ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE
from gymnasium.core import Env
import numpy as np
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
from gymnasium.envs.mujoco.mujoco_rendering import  OffScreenViewer, WindowViewer
import mujoco

DEFAULT_CAMERA_CONFIG = {
"distance": 2,
"azimuth": 305,
"elevation": -20.0,
"lookat": np.array([0, 0.5, 0.0]),
}
class MetaWorldWrapper(Env):
    def __init__(self, task_name, render_mode="rgb_array", width=224, height=224,**kwargs):  # task_name parameter accepted
        super().__init__()

        # Create the environment with proper render mode
        self.env = ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE[task_name](render_mode=render_mode)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        # Store dimensions
        self.width = width
        self.height = height

        # Set width and height for the environment
        self.env.width = width
        self.env.height = height

        # Set model visualization parameters
        if hasattr(self.env, 'model') and hasattr(self.env.model, 'vis'):
            self.env.model.vis.global_.offwidth = width
            self.env.model.vis.global_.offheight = height

        # Set width and height for the mujoco renderer if it exists
        if hasattr(self.env, 'mujoco_renderer') and self.env.mujoco_renderer is not None:
            self.env.mujoco_renderer.width = width
            self.env.mujoco_renderer.height = height

        # Force initialize random vector
        if hasattr(self.env, '_setup_rand_vec'):
            self.env._setup_rand_vec()

        # Setup camera configuration
        self.camera_azimuths = kwargs.get("camera_azimuths", [305])
        self.current_view_index = 0

        # Override the mujoco renderer with our custom configuration if needed
        if hasattr(self.env, 'mujoco_renderer'):
            # Update the default camera config
            if hasattr(self.env.mujoco_renderer, 'default_cam_config'):
                if self.env.mujoco_renderer.default_cam_config is None:
                    self.env.mujoco_renderer.default_cam_config = DEFAULT_CAMERA_CONFIG.copy()
                else:
                    self.env.mujoco_renderer.default_cam_config.update(DEFAULT_CAMERA_CONFIG)

            # Set custom viewer getter
            self.env.mujoco_renderer._get_viewer = self.custom_get_viewer

    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)

    def step(self, action):
        return self.env.step(action)

    def render(self):
        # Ensure the view is set correctly before rendering
        if hasattr(self, 'current_view_index'):
            self.env.mujoco_renderer.default_cam_config["azimuth"] = self.camera_azimuths[self.current_view_index]
        return self.env.render()

    @property
    def n_views(self):
        return len(self.camera_azimuths)


    def set_view(self, view_ind):
        self.current_view_index = view_ind
        # Only initialize viewer when actually needed
        if hasattr(self.env, 'mujoco_renderer') and self.env.mujoco_renderer is not None:
            self.env.mujoco_renderer.default_cam_config["azimuth"] = self.camera_azimuths[view_ind]
            if self.env.mujoco_renderer.viewer is not None:
                self.env.mujoco_renderer._set_cam_config()

    def custom_get_viewer(self, render_mode: str):
        self.env.mujoco_renderer.viewer = self.env.mujoco_renderer._viewers.get(render_mode)
        if self.env.mujoco_renderer.viewer is None:
            if render_mode == "human":
                self.env.mujoco_renderer.viewer = WindowViewer(self.env.mujoco_renderer.model, self.env.mujoco_renderer.data)
            elif render_mode in {"rgb_array", "depth_array"}:
                # Use the wrapper's stored dimensions
                width = getattr(self.env.mujoco_renderer, 'width', self.width)
                height = getattr(self.env.mujoco_renderer, 'height', self.height)
                self.env.mujoco_renderer.viewer = CustomOffScreenViewer(
                    self.env.mujoco_renderer.model,
                    self.env.mujoco_renderer.data,
                    width=width,
                    height=height
                )
            else:
                raise AttributeError(f"Unexpected mode: {render_mode}, expected modes: human, rgb_array, or depth_array")
            self.env.mujoco_renderer._set_cam_config()
            self.env.mujoco_renderer._viewers[render_mode] = self.env.mujoco_renderer.viewer

        if len(self.env.mujoco_renderer._viewers.keys()) > 1:
            self.env.mujoco_renderer.viewer.make_context_current()

        return self.env.mujoco_renderer.viewer


class CustomOffScreenViewer(OffScreenViewer):
    def __init__(self, model, data, width=None, height=None):
        super().__init__(model, data, width=width, height=height)

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