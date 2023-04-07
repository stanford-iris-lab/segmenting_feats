from collections import namedtuple
# from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
from PIL import Image
import torch

from collections import namedtuple
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

import mvp

from r3meval.utils.obs_wrappers import MuJoCoPixelObs, StateEmbedding
from r3meval.utils.visualizations import place_attention_heatmap_over_images


def visualize_shifts_metaworld():
    env_to_model_name = {
        'assembly-v2-goal-observable':'sawyer_assembly_peg',
        'bin-picking-v2-goal-observable':'sawyer_bin_picking',
        'button-press-topdown-v2-goal-observable':'sawyer_button_press_topdown',
        'drawer-open-v2-goal-observable':'sawyer_drawer',
        'hammer-v2-goal-observable':'sawyer_hammer',
    }
    model_path = f'/iris/u/kayburns/packages/metaworld_r3m/metaworld/envs/assets_v2/sawyer_xyz/'
    # model_path = '/iris/u/kayburns/packages/metaworld_r3m/metaworld/envs/assets_v2/sawyer_xyz/sawyer_assembly_peg_blue-woodtable.xml'
    # for shift in ['_distractor_medium', '_granite_table', '_metal1_table', '_cast_right', '_cast_left', '_darker', '_brighter', '']:
    for camera_name in ['top_cap2']:#, 'right_cap2', 'left_cap2']:
        for env_name in ['assembly-v2-goal-observable', 'bin-picking-v2-goal-observable', 'button-press-topdown-v2-goal-observable', 'drawer-open-v2-goal-observable', 'hammer-v2-goal-observable']:
            for shift in ['_distractor_easy', '_distractor_medium', '_distractor_hard', '_blue-woodtable', '_dark-woodtable', '_darkwoodtable', '_cast_right', '_cast_left', '_darker', '_brighter', '']:
                model_name = model_path+env_to_model_name[env_name]+shift+'.xml'
                # model_name=model_path
                e = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_name](model_name=model_name)
                e._freeze_rand_vec = False
                e.spec = namedtuple('spec', ['id', 'max_episode_steps'])
                e.spec.id = env_name
                e.spec.max_episode_steps = 500
                e = MuJoCoPixelObs(e, camera_name=camera_name, width=256, height=256)
                e = StateEmbedding(e, embedding_name='resnet50', load_path='r3m', 
                            proprio=9, camera_name=camera_name, env_name=env_name)
                im = e.render()
                image = Image.fromarray(im)
                env_name_prefix = env_name.split('-')[0]
                image.save(f'photos_of_envs/test_{env_name_prefix}{shift}_{camera_name}.jpg')

shifts = [
    'slide_metal2', 'darker', 'distractor_hard'
    # 'distractor_cracker_box', \
    # 'distractor_medium', \
    # 'distractor_hard', \
    # 'cast_left', 'cast_right', 'brighter', 'darker' \
]

def visualize_heatmap(model='dino', visualize_shift=False):
    camera_name = 'left_cap2'
    if model == 'dino':
        embedding_model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16').cuda()
    else:
        embedding_model= mvp.load("vitb-mae-egosoup").cuda()

    env_name = 'kitchen_knob1_on-v3'
    for shift in shifts:
        e = gym.make(env_name, model_path = f'/iris/u/kayburns/packages/mj_envs/mj_envs/envs/relay_kitchen/assets/franka_kitchen_{shift}.xml')
        e = MuJoCoPixelObs(e, camera_name=camera_name, width=256, height=256)
        e = StateEmbedding(e, embedding_name='resnet50', load_path='r3m', 
                            proprio=9, camera_name=camera_name, env_name=env_name)
        e.reset()
        im = e.env.get_image()
        image = Image.fromarray(im.astype('uint8'), mode='RGB')
        image.save(f'heads_with distractors/train_env_{shift}.png')
    import pdb; pdb.set_trace()

    for head in range(6):
        e = gym.make(env_name)
        e = MuJoCoPixelObs(e, camera_name=camera_name, width=256, height=256)
        e = StateEmbedding(e, embedding_name='resnet50', load_path='r3m', 
                            proprio=9, camera_name=camera_name, env_name=env_name)
        e.reset()
        im = e.env.get_image()
        attention_vis = place_attention_heatmap_over_images([im], embedding_model, model, head=head)
        image = Image.fromarray(attention_vis[0].astype('uint8'), mode='RGB')
        image.save(f'heads_with distractors/{model}_head_{head}_{camera_name}.png')

    if visualize_shift:
        for shift in shifts:
            for head in range(6):
                e = gym.make(env_name, model_path = f'/iris/u/kayburns/packages/mj_envs/mj_envs/envs/relay_kitchen/assets/franka_kitchen_{shift}.xml')
                e = MuJoCoPixelObs(e, camera_name=camera_name, width=256, height=256)
                e = StateEmbedding(e, embedding_name='resnet50', load_path='r3m', 
                                    proprio=9, camera_name=camera_name, env_name=env_name)
                e.reset()
                im = e.env.get_image()
                attention_vis = place_attention_heatmap_over_images([im], embedding_model, model, head=head)
                image = Image.fromarray(attention_vis[0].astype('uint8'), mode='RGB')
                image.save(f'heads_with distractors/{model}_{shift}_head_{head}.png')

visualize_heatmap(model='dino', visualize_shift=True)