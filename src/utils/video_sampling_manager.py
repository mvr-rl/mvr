import numpy as np

class VideoSamplingManager():
    def __init__(
        self, video_sampling_configs, env, shared_queue, event, env_id):
        self.video_length = video_sampling_configs["video_length"]
        self.sep = video_sampling_configs["sep"]
        self.should_sample = False
        self.trajectory_length = 0
        self.sampling_freq = video_sampling_configs["sampling_freq"]
        self.rendering_freq = video_sampling_configs["rendering_freq"]
        self.view_mode = video_sampling_configs["view_mode"]
        assert self.sep % self.rendering_freq == 0
        assert self.video_length % self.sep == 0
        self.n_trajectory_since_last_sampling = 0
        self.shared_queue = shared_queue
        self.buffer = []
        self.env = env
        self.event = event
        self.env_id = env_id
        self.trajectory_count = 0
        self.view_ind = 0
        self.dummy_render_array = None
    
    def render(self, mode="all", reset=False):
        if mode == "all":
            render_arrays = []
            for view_ind in range(self.env.unwrapped.n_views):
                self.env.unwrapped.set_view(view_ind)
                render_array = self.env.unwrapped.render()
                render_arrays.append(render_array)
            if len(render_arrays) > 1:
                render_array = np.concatenate(render_arrays, -1)
            else:
                render_array = render_arrays[0]
        elif mode == "alternative":
            if reset:
                self.view_ind = (self.view_ind + 1) \
                    % self.env.unwrapped.n_views 
                self.env.unwrapped.set_view(self.view_ind)    
            render_array = self.env.unwrapped.render()
        return render_array
    
    def on_reset(self, observation):
        self.trajectory_length = 1
        self.trajectory_count += 1
        if len(self.buffer) >= self.video_length:
            #self.event.clear()
            render_arrays = np.array([b["render_array"] for b in self.buffer])
            rendered = np.array([b["rendered"] for b in self.buffer])
            for step in range(
                0, len(self.buffer) - self.video_length, self.sep):
                observations, next_observations = [], []
                actions, rewards = [], []
                #render_arrays = []
                for i in range(self.video_length):
                    observations.append(self.buffer[step+i]["observation"])
                    next_observations.append(self.buffer[step+i]["next_observation"])
                    actions.append(self.buffer[step+i]["action"])
                    rewards.append(self.buffer[step+i]["reward"])
                    #render_arrays.append(self.buffer[step+i]["render_array"])
                info = self.buffer[step+self.video_length-1]["info"]
                info["trajectory_id"] = f"{self.env_id}_{self.trajectory_count}"
                info["step"] = step
                clip_render_arrays = render_arrays[step: step+self.video_length][::self.rendering_freq]
                # clip_rendered = rendered[step: step+self.video_length][::self.rendering_freq].copy()
                self.shared_queue.put(
                    dict(observations=np.array(observations),
                    actions=np.array(actions),
                    rewards=np.array(rewards),
                    next_observations=np.array(next_observations),
                    done=self.buffer[step+self.video_length-1]["done"],
                    info=self.buffer[step+self.video_length-1]["info"],
                    render_arrays=clip_render_arrays ))
        self.event.set() 
        self.buffer.clear()
        if self.should_sample:
            render_array = self.render(mode=self.view_mode, reset=True)
            self.buffer.clear()
            self.buffer.append(
                dict(observation=observation, 
                     render_array=render_array,
                     rendered=1))
        else:
            self.n_trajectory_since_last_sampling += 1        

    def on_step(self, observation, action, reward, done, info):
        self.trajectory_length += 1
        if self.should_sample:
            self.buffer[-1].update(dict(
                next_observation=observation,
                action=action,
                reward=reward, done=done, info=info)) 
            
            if len(self.buffer) % self.rendering_freq == 0:
                render_array = self.render(mode=self.view_mode)
                rendered = 1
            else:
                if self.dummy_render_array is None:
                    self.dummy_render_array = self.buffer[-1]["render_array"].copy()
                render_array = self.dummy_render_array
                rendered = 0
            self.buffer.append(dict(
                observation=observation, render_array=render_array, rendered=rendered))

    def on_done(self):
        if (not self.should_sample)\
            and self.trajectory_length > self.video_length\
            and self.n_trajectory_since_last_sampling >= self.sampling_freq:
            self.should_sample = True
            self.n_trajectory_since_last_sampling = 0
        else:
            self.should_sample = False
      