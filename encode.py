# import torch
# from PIL import Image
# from transformers import AutoProcessor, AutoModelForImageTextToText


# import mediapy as mp
import os
import numpy as np
import tritonclient.grpc as grpcclient

import env
from stable_baselines3.common.env_util import make_vec_env
from src.utils.subproc_vec_env import SubprocVecEnv
from src.training.training_utils import set_egl_env_vars
import gymnasium as gym
import mediapy
import torch
import gc
import pickle
set_egl_env_vars()

def infer(client, text, images):
    inputs = [
        grpcclient.InferInput("text", text.shape, datatype="BYTES"),
        grpcclient.InferInput("images", images.shape, datatype="UINT8")]
    inputs[0].set_data_from_numpy(text)
    inputs[1].set_data_from_numpy(images)
    
    outputs = [grpcclient.InferRequestedOutput("text"), grpcclient.InferRequestedOutput("embedding")]

    responses = client.infer(
        model_name="smolvlm2", inputs=inputs, outputs=outputs)
    # Output
    embedding = responses.as_numpy("embedding")
    text = responses.as_numpy("text")
    return embedding, text

task = "stick-pull"
model_name = "500M"

server_url = os.environ.get("TRITON_SERVER_URL")
if server_url is None:
    raise ValueError("Environment variable TRITON_SERVER_URL must be set to the Triton server address (e.g., 'localhost:8001').")

client = grpcclient.InferenceServerClient(url=server_url)
with open(f"{task}_trajectories.pkl", "rb") as f:
    trajectories = pickle.load(f)
key = "near_object"
frame_seps = [] 
label_seps = [] 
for frames, infos in trajectories:
    labels = infos[key]
    for i in range(0, len(frames), 8):
        if i + 8 > len(frames):
            break
        frame_seps.append(frames[i:i+8])
        label_seps.append(np.mean(labels[i:i+8]))
frame_seps = np.array(frame_seps, dtype=np.uint8)
label_seps = np.array(label_seps)


print("Obtaining embeddings without text prompts")
prompt = ""
prompt_np = np.array([[prompt.encode('utf-8')]])
frame_sep_embs = []
for ind in range(0, len(frame_seps), 8):
    batch_size = min(8, len(frame_seps) - ind)
    frame_sep_emb, _ = infer(client,
      prompt_np.repeat(batch_size, axis=0),
      frame_seps[ind: ind + batch_size][:, ::2],
      )
    frame_sep_embs.append(frame_sep_emb)
frame_sep_embs = np.concatenate(frame_sep_embs, axis=0)
with open(f"{task}_emb_{model_name}_no_text.pkl", "wb") as f:
    pickle.dump((frame_sep_embs, frame_seps, label_seps), f)


key = "near_object"
print(f"Obtaining embeddings for {key}")
frame_seps = [] 
label_seps = [] 
for frames, infos in trajectories:
    labels = infos[key]
    for i in range(0, len(frames), 8):
        if i + 8 > len(frames):
            break
        frame_seps.append(frames[i:i+8])
        label_seps.append(np.mean(labels[i:i+8]))
frame_seps = np.array(frame_seps, dtype=np.uint8)
label_seps = np.array(label_seps)
system = "You are a visual agent and should provide concise answers. Here are four images showing an robot manipulating objects. Based on the images, answer the following question: "
prompt = system + "is the robot gripper close to the blue stick?"
prompt_np = np.array([[prompt.encode('utf-8')]])
frame_sep_embs = []
for ind in range(0, len(frame_seps), 8):
    batch_size = min(8, len(frame_seps) - ind)
    frame_sep_emb, _ = infer(client,
      prompt_np.repeat(batch_size, axis=0),
      frame_seps[ind: ind + batch_size][:, ::2],
      )
    frame_sep_embs.append(frame_sep_emb)
frame_sep_embs = np.concatenate(frame_sep_embs, axis=0)
with open(f"{task}_emb_{model_name}_{key}.pkl", "wb") as f:
    pickle.dump((frame_sep_embs, frame_seps, label_seps), f)
    

key = "grasp_success"
print(f"Obtaining embeddings for {key}")
frame_seps = [] 
label_seps = [] 
for frames, infos in trajectories:
    labels = infos[key]
    for i in range(0, len(frames), 8):
        if i + 8 > len(frames):
            break
        frame_seps.append(frames[i:i+8])
        label_seps.append(np.mean(labels[i:i+8]))
frame_seps = np.array(frame_seps, dtype=np.uint8)
label_seps = np.array(label_seps)
system = "You are a visual agent and should provide concise answers. Here are four images showing an robot manipulating objects. Based on the images, answer the following question: "
prompt = system + "is the robot gripper grasping the blue stick?"
prompt_np = np.array([[prompt.encode('utf-8')]])
frame_sep_embs = []
for ind in range(0, len(frame_seps), 8):
    batch_size = min(8, len(frame_seps) - ind)
    frame_sep_emb, _ = infer(client,
      prompt_np.repeat(batch_size, axis=0),
      frame_seps[ind: ind + batch_size][:, ::2],
      )
    frame_sep_embs.append(frame_sep_emb)
frame_sep_embs = np.concatenate(frame_sep_embs, axis=0)
with open(f"{task}_emb_{model_name}_{key}.pkl", "wb") as f:
    pickle.dump((frame_sep_embs, frame_seps, label_seps), f)
    
key = "in_place_reward"
print(f"Obtaining embeddings for {key}")
frame_seps = [] 
label_seps = [] 
for frames, infos in trajectories:
    labels = infos[key]
    for i in range(0, len(frames), 8):
        if i + 8 > len(frames):
            break
        frame_seps.append(frames[i:i+8])
        label_seps.append(np.mean(labels[i:i+8]))
frame_seps = np.array(frame_seps, dtype=np.uint8)
label_seps = np.array(label_seps)
system = "You are a visual agent and should provide concise answers. Here are four images showing an robot manipulating objects. Based on the images, answer the following question: "
prompt = system + "is the blue stick inside the handle of the gray kettle?"
prompt_np = np.array([[prompt.encode('utf-8')]])
frame_sep_embs = []
for ind in range(0, len(frame_seps), 8):
    batch_size = min(8, len(frame_seps) - ind)
    frame_sep_emb, _ = infer(client,
      prompt_np.repeat(batch_size, axis=0),
      frame_seps[ind: ind + batch_size][:, ::2],
      )
    frame_sep_embs.append(frame_sep_emb)
frame_sep_embs = np.concatenate(frame_sep_embs, axis=0)
with open(f"{task}_emb_{model_name}_{key}.pkl", "wb") as f:
    pickle.dump((frame_sep_embs, frame_seps, label_seps), f)
    
key = "obj_to_target"
print(f"Obtaining embeddings for {key}")
frame_seps = [] 
label_seps = [] 
for frames, infos in trajectories:
    labels = infos[key]
    for i in range(0, len(frames), 8):
        if i + 8 > len(frames):
            break
        frame_seps.append(frames[i:i+8])
        label_seps.append(np.mean(labels[i:i+8]))
frame_seps = np.array(frame_seps, dtype=np.uint8)
label_seps = np.array(label_seps)
system = "You are a visual agent and should provide concise answers. Here are four images showing an robot manipulating objects. Based on the images, answer the following question: "
prompt = system + "is the robot pulling the gray kettle to the green point?"
prompt_np = np.array([[prompt.encode('utf-8')]])
frame_sep_embs = []
for ind in range(0, len(frame_seps), 8):
    batch_size = min(8, len(frame_seps) - ind)
    frame_sep_emb, _ = infer(client,
      prompt_np.repeat(batch_size, axis=0),
      frame_seps[ind: ind + batch_size][:, ::2],
      )
    frame_sep_embs.append(frame_sep_emb)
frame_sep_embs = np.concatenate(frame_sep_embs, axis=0)
with open(f"{task}_emb_{model_name}_{key}.pkl", "wb") as f:
    pickle.dump((frame_sep_embs, frame_seps, label_seps), f)

client.close()
