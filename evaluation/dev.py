from collections import namedtuple

from torch import embedding
from r3meval.utils.gym_env import GymEnv
from r3meval.utils.obs_wrappers import MuJoCoPixelObs, StateEmbedding
from r3meval.utils.sampling import sample_paths
from r3meval.utils.gaussian_mlp import MLP
from r3meval.utils.behavior_cloning import BC
from r3meval.utils.visualizations import place_attention_heatmap_over_images
from tabulate import tabulate
from tqdm import tqdm
import mj_envs, gym 
import numpy as np, time as timer, multiprocessing, pickle, os
import os
from collections import namedtuple

import kitchen_shift

env_name = 'kitchen_knob1_on-v3'
shift = 'none'
render_gpu_id = 0
image_width = 256
image_height = 256
camera_name = 'right_cap2'
embedding_name = 'resnet50'
load_path = 'r3m'
proprio = 9
device = 'cuda'

distractors = ['cracker_box']
textures_slide = ['wood2', 'metal2', 'tile1']
textures_hinge = ['wood1', 'metal1', 'marble1']
textures_floor = ['tile1', 'wood1']
textures_counter = ['wood2']
lightings = ['cast_left', 'cast_right', 'brighter', 'darker']

for distractor in distractors:
    e = gym.make(env_name, model_path=f'/iris/u/kayburns/packages/mj_envs/mj_envs/envs/relay_kitchen/assets/franka_kitchen_distractor_{distractor}.xml')
    ## Wrap in pixel observation wrapper
    e = MuJoCoPixelObs(e, width=image_width, height=image_height, 
                        camera_name=camera_name, device_id=render_gpu_id,
                        shift=shift)

    ## Wrapper which encodes state in pretrained model
    e = StateEmbedding(e, embedding_name=embedding_name, device=device, load_path=load_path, 
                    proprio=proprio, camera_name=camera_name, env_name=env_name)
    e = GymEnv(e)
    img = e.env.env.get_image()

    import cv2
    cv2.imwrite(f'photos/distractor_{distractor}.png', img[:, :, ::-1])

for texture in textures_counter:
    e = gym.make(env_name, model_path=f'/iris/u/kayburns/packages/mj_envs/mj_envs/envs/relay_kitchen/assets/franka_kitchen_counter_{texture}.xml')
    ## Wrap in pixel observation wrapper
    e = MuJoCoPixelObs(e, width=image_width, height=image_height, 
                        camera_name=camera_name, device_id=render_gpu_id,
                        shift=shift)

    ## Wrapper which encodes state in pretrained model
    e = StateEmbedding(e, embedding_name=embedding_name, device=device, load_path=load_path, 
                    proprio=proprio, camera_name=camera_name, env_name=env_name)
    e = GymEnv(e)
    img = e.env.env.get_image()

    import cv2
    cv2.imwrite(f'photos/counter_{texture}.png', img[:, :, ::-1])


for texture in textures_floor:
    e = gym.make(env_name, model_path=f'/iris/u/kayburns/packages/mj_envs/mj_envs/envs/relay_kitchen/assets/franka_kitchen_floor_{texture}.xml')
    ## Wrap in pixel observation wrapper
    e = MuJoCoPixelObs(e, width=image_width, height=image_height, 
                        camera_name=camera_name, device_id=render_gpu_id,
                        shift=shift)

    ## Wrapper which encodes state in pretrained model
    e = StateEmbedding(e, embedding_name=embedding_name, device=device, load_path=load_path, 
                    proprio=proprio, camera_name=camera_name, env_name=env_name)
    e = GymEnv(e)
    img = e.env.env.get_image()

    import cv2
    cv2.imwrite(f'photos/floor_{texture}.png', img[:, :, ::-1])

for texture in textures_slide:
    e = gym.make(env_name, model_path=f'/iris/u/kayburns/packages/mj_envs/mj_envs/envs/relay_kitchen/assets/franka_kitchen_slide_{texture}.xml')
    ## Wrap in pixel observation wrapper
    e = MuJoCoPixelObs(e, width=image_width, height=image_height, 
                        camera_name=camera_name, device_id=render_gpu_id,
                        shift=shift)

    ## Wrapper which encodes state in pretrained model
    e = StateEmbedding(e, embedding_name=embedding_name, device=device, load_path=load_path, 
                    proprio=proprio, camera_name=camera_name, env_name=env_name)
    e = GymEnv(e)
    img = e.env.env.get_image()

    import cv2
    cv2.imwrite(f'photos/slide_{texture}.png', img[:, :, ::-1])

for texture in textures_hinge:
    e = gym.make(env_name, model_path=f'/iris/u/kayburns/packages/mj_envs/mj_envs/envs/relay_kitchen/assets/franka_kitchen_hinge_{texture}.xml')
    ## Wrap in pixel observation wrapper
    e = MuJoCoPixelObs(e, width=image_width, height=image_height, 
                        camera_name=camera_name, device_id=render_gpu_id,
                        shift=shift)

    ## Wrapper which encodes state in pretrained model
    e = StateEmbedding(e, embedding_name=embedding_name, device=device, load_path=load_path, 
                    proprio=proprio, camera_name=camera_name, env_name=env_name)
    e = GymEnv(e)
    img = e.env.env.get_image()

    import cv2
    cv2.imwrite(f'photos/hinge_{texture}.png', img[:, :, ::-1])

for lighting in lightings:
    e = gym.make(env_name, model_path=f'/iris/u/kayburns/packages/mj_envs/mj_envs/envs/relay_kitchen/assets/franka_kitchen_{lighting}.xml')
    ## Wrap in pixel observation wrapper
    e = MuJoCoPixelObs(e, width=image_width, height=image_height, 
                        camera_name=camera_name, device_id=render_gpu_id,
                        shift=shift)

    ## Wrapper which encodes state in pretrained model
    e = StateEmbedding(e, embedding_name=embedding_name, device=device, load_path=load_path, 
                    proprio=proprio, camera_name=camera_name, env_name=env_name)
    e = GymEnv(e)
    img = e.env.env.get_image()

    import cv2
    cv2.imwrite(f'photos/lighting_{lighting}.png', img[:, :, ::-1])

