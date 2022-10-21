# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
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


import metaworld
from metaworld.envs import (ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE,
                            ALL_V2_ENVIRONMENTS_GOAL_HIDDEN)


def env_constructor(env_name, device='cuda', image_width=256, image_height=256,
                    camera_name=None, embedding_name='resnet50', pixel_based=True,
                    render_gpu_id=0, load_path="", proprio=False, lang_cond=False,
                    gc=False, model_path=None, shift=None):

    ## If pixel based will wrap in a pixel observation wrapper
    if pixel_based:
        ## Need to do some special environment config for the metaworld environments
        if "v2" in env_name:
            e  = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_name]()
            e._freeze_rand_vec = False
            e.spec = namedtuple('spec', ['id', 'max_episode_steps'])
            e.spec.id = env_name
            e.spec.max_episode_steps = 500
        else:
            if model_path:
                e = gym.make(env_name, model_path=model_path)
            else:
                e = gym.make(env_name) # probably unnecessary
        ## Wrap in pixel observation wrapper
        e = MuJoCoPixelObs(e, width=image_width, height=image_height, 
                           camera_name=camera_name, device_id=render_gpu_id,
                           shift=shift)
        ## Wrapper which encodes state in pretrained model
        e = StateEmbedding(e, embedding_name=embedding_name, device=device, load_path=load_path, 
                        proprio=proprio, camera_name=camera_name, env_name=env_name)
        e = GymEnv(e)
    else:
        print("Only supports pixel based")
        assert(False)
    return e


def make_bc_agent(env_kwargs:dict, bc_kwargs:dict, demo_paths:list, epochs:int, seed:int, pixel_based=True, model_path=None):
    ## Creates environment
    env_kwargs['model_path'] = model_path
    e = env_constructor(**env_kwargs)

    ## Creates MLP (Where the FC Network has a batchnorm in front of it)
    policy = MLP(e.spec, hidden_sizes=(256, 256), seed=seed)
    policy.model.proprio_only = False
    if bc_kwargs["load_path"]:
        policy = pickle.load(open(bc_kwargs["load_path"], 'rb'))
        
    ## Pass the encoder params to the BC agent (for finetuning)
    if pixel_based:
        enc_p = e.env.embedding.parameters()
    else:
        print("Only supports pixel based")
        assert(False)
    bc_agent = BC(demo_paths, policy=policy, epochs=epochs, set_transforms=False, encoder_params=enc_p, **bc_kwargs)

    ## Pass the environmetns observation encoder to the BC agent to encode demo data
    if pixel_based:
        bc_agent.encodefn = e.env.encode_batch
    else:
        print("Only supports pixel based")
        assert(False)
    return e, bc_agent


def configure_cluster_GPUs(gpu_logical_id: int) -> int:
    # get the correct GPU ID
    if "SLURM_STEP_GPUS" in os.environ.keys():
        physical_gpu_ids = os.environ.get('SLURM_STEP_GPUS')
        gpu_id = int(physical_gpu_ids.split(',')[gpu_logical_id])
        print("Found slurm-GPUS: <Physical_id:{}>".format(physical_gpu_ids))
        print("Using GPU <Physical_id:{}, Logical_id:{}>".format(gpu_id, gpu_logical_id))
    else:
        gpu_id = 0 # base case when no GPUs detected in SLURM
        print("No GPUs detected. Defaulting to 0 as the device ID")
    return gpu_id

def eval_model_path(job_data, demo_paths, model_path, texture, init=False):
    env_kwargs = job_data['env_kwargs']
    e, agent = make_bc_agent(env_kwargs=env_kwargs, bc_kwargs=job_data['bc_kwargs'], 
                            demo_paths=demo_paths, epochs=1, seed=job_data['seed'], pixel_based=job_data["pixel_based"],
                            model_path=model_path)
    if init:
        agent.logger.init_wb(job_data, project='r3m_shift_eval_2')

    # perform evaluation rollouts every few epochs
    agent.policy.model.eval()
    if job_data['pixel_based']:
        e.env.embedding.eval()
    paths = sample_paths(num_traj=job_data['eval_num_traj'], env=e, #env_constructor, 
                            policy=agent.policy, eval_mode=True, horizon=e.horizon, 
                            base_seed=job_data['seed'], num_cpu=job_data['num_cpu'], 
                            env_kwargs=env_kwargs)
    
    try:
        ## Success computation and logging for Adroit and Kitchen
        success_percentage = e.env.unwrapped.evaluate_success(paths)
        for i, path in enumerate(paths):
            if (i < 1) and job_data['pixel_based']:
                vid = path['images']
                filename = f'./iterations/vid_{i}.gif'
                from moviepy.editor import ImageSequenceClip
                cl = ImageSequenceClip(vid, fps=20)
                cl.write_gif(filename, fps=20)

                if job_data.embedding == 'dino' or job_data.embedding == 'mvp':
                    for j in range(6):
                        heatmap_vid = place_attention_heatmap_over_images(vid, e.env.embedding, job_data.embedding, head=j)
                        filename = f'./iterations/heatmap_vid_{i}_{j}_{texture}.gif'
                        cl = ImageSequenceClip(heatmap_vid, fps=20)
                        cl.write_gif(filename, fps=20)
    except:
        ## Success computation and logging for MetaWorld
        sc = []
        for i, path in enumerate(paths):
            sc.append(path['env_infos']['success'][-1])
            if (i < 3) and job_data['pixel_based']:
                vid = path['images']
                filename = f'./iterations/vid_{i}.gif'
                from moviepy.editor import ImageSequenceClip
                cl = ImageSequenceClip(vid, fps=20)
                cl.write_gif(filename, fps=20)

                for j in range(6):
                    heatmap_vid = place_attention_heatmap_over_images(vid, e.env.embedding, head=j)
                    filename = f'./iterations/heatmap_vid_{i}_{j}.gif'
                    cl = ImageSequenceClip(heatmap_vid, fps=20)
                    cl.write_gif(filename, fps=20)
        success_percentage = np.mean(sc) * 100
    if not model_path:
        agent.logger.log_kv('eval_success', success_percentage)
    else:
        texture = model_path.split('_')[-1].split('.')[0]
        agent.logger.log_kv(f'eval_success{texture}', success_percentage)
    agent.logger.save_wb(step=0)
    
    print_data = sorted(filter(lambda v: np.asarray(v[1]).size == 1,
                                agent.logger.get_current_log().items()))
    print(tabulate(print_data))

def eval_loop(job_data:dict) -> None:

    # configure GPUs
    os.environ['GPUS'] = os.environ.get('SLURM_STEP_GPUS', '0')
    physical_gpu_id = 0 #configure_cluster_GPUs(job_data['env_kwargs']['render_gpu_id'])
    job_data['env_kwargs']['render_gpu_id'] = physical_gpu_id

    # Infers the location of the demos
    ## V2 is metaworld, V0 adroit, V3 kitchen
    data_dir = '/iris/u/kayburns/data/r3m/'
    if "v2" in job_data['env_kwargs']['env_name']:
        demo_paths_loc = data_dir + 'final_paths_multiview_meta_200/' + job_data['camera'] + '/' + job_data['env_kwargs']['env_name'] + '.pickle'
    elif "v0" in job_data['env_kwargs']['env_name']:
        demo_paths_loc = data_dir + 'final_paths_multiview_adroit_200/' + job_data['camera'] + '/' + job_data['env_kwargs']['env_name'] + '.pickle'
    else:
        demo_paths_loc = data_dir + 'final_paths_multiview_rb_200/' + job_data['camera'] + '/' + job_data['env_kwargs']['env_name'] + '.pickle'

    ## Loads the demos
    demo_paths = pickle.load(open(demo_paths_loc, 'rb'))
    demo_paths = demo_paths[:job_data['num_demos']]
    print(len(demo_paths))
    demo_score = np.mean([np.sum(p['rewards']) for p in demo_paths])
    print("Demonstration score : %.2f " % demo_score)

    # Make log dir
    if os.path.isdir(job_data['job_name']) == False: os.mkdir(job_data['job_name'])
    previous_dir = os.getcwd()
    os.chdir(job_data['job_name']) # important! we are now in the directory to save data
    if os.path.isdir('iterations') == False: os.mkdir('iterations')
    if os.path.isdir('logs') == False: os.mkdir('logs')

    ## Creates agent and environment
    model_paths = None
    eval_model_path(job_data, demo_paths, None, None, init=True)

    textures = ['wood2', 'metal2', 'tile1']
    for texture in textures:
        model_path = f'/iris/u/kayburns/packages/mj_envs/mj_envs/envs/relay_kitchen/assets/franka_kitchen_slide_{texture}.xml'
        eval_model_path(job_data, demo_paths, model_path, texture, init=False)
    lightings = ['cast_left', 'cast_right', 'brighter', 'darker']
    for lighting in lightings:
        model_path = f'/iris/u/kayburns/packages/mj_envs/mj_envs/envs/relay_kitchen/assets/franka_kitchen_{lighting}.xml'
        eval_model_path(job_data, demo_paths, model_path, lighting, init=False)

        

