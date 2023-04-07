# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import time as timer
import hydra
import multiprocessing
from omegaconf import DictConfig, OmegaConf
from eval_loop import eval_loop

cwd = os.getcwd()

sweep_dir = '/iris/u/kayburns/new_arch/r3m/evaluation/outputs/main_sweep_noft/'

def is_target_task(target_job_data, query_job_data):
    # envs
    slide_door = query_job_data.env == 'hammer-v2-goal-observable'

    # cameras
    left_cap = query_job_data.camera == 'left_cap2'

    # embeddings
    deit_sin = (query_job_data.embedding == 'deit_s_sin') and (query_job_data.env_kwargs.load_path == 'deit_s_sin')

    # num_demos
    ten_demos = query_job_data.num_demos == 10

    # seeds
    seed_123 = query_job_data.seed == 123
    
    # if (slide_door and left_cap and deit_sin and ten_demos and seed_123):
    #     import pdb; pdb.set_trace()

    target_conditions_met = [
        target_job_data.env == query_job_data.env, 
        target_job_data.camera == query_job_data.camera,
        target_job_data.embedding == query_job_data.embedding,
        target_job_data.env_kwargs.load_path == query_job_data.env_kwargs.load_path,
        target_job_data.bc_kwargs.finetune == query_job_data.bc_kwargs.finetune,
        target_job_data.num_demos == query_job_data.num_demos,
        target_job_data.seed == query_job_data.seed,
        target_job_data.proprio == query_job_data.proprio,
        target_job_data.get('ft_only_last_layer', False) == query_job_data.get('ft_only_last_layer', False)
    ]

    return all(target_conditions_met)

# ===============================================================================
# Process Inputs and configure job
# ===============================================================================
@hydra.main(config_name="eval_config", config_path="config")
def configure_jobs(job_data:dict) -> None:
    os.environ['GPUS'] = os.environ.get('SLURM_STEP_GPUS', '0')
    
    print("========================================")
    print("Job Configuration")
    print("========================================")

    job_data = OmegaConf.structured(OmegaConf.to_yaml(job_data))

    job_data['cwd'] = cwd

    run_paths = os.listdir(sweep_dir)

    for run_path in run_paths:

        old_config_path = os.path.join(sweep_dir, run_path, 'job_config.json')
        embedding_path = os.path.join(sweep_dir, run_path, 'try_r3m/iterations/embedding_best.pickle')
        policy_path = os.path.join(sweep_dir, run_path, 'try_r3m/iterations/policy_best.pickle')

        if not os.path.exists(old_config_path):
            continue

        with open(old_config_path, 'r') as fp:
            old_job_data = OmegaConf.load(fp)

        if not is_target_task(job_data, old_job_data):
            continue

        if not os.path.isfile(policy_path):
            print(f'No weights for ' \
                  f'{old_job_data.env}, ' \
                  f'{old_job_data.bc_kwargs.finetune}, ' \
                  f'{old_job_data.num_demos}, ' \
                  f'{old_job_data.seed}, ' \
                  f'{old_job_data.env_kwargs.load_path}, ' \
                  f'{os.path.join(sweep_dir, run_path)}, ' \
                  f'{old_job_data.camera}')
            continue

        job_data.env_kwargs = old_job_data.env_kwargs
        job_data.bc_kwargs.load_path = policy_path
        if job_data.bc_kwargs.finetune:
            job_data.env_kwargs.load_path = embedding_path
        else:
            job_data.env_kwargs.load_path = old_job_data.env_kwargs.load_path

        with open('job_config.json', 'w') as fp:
            OmegaConf.save(config=job_data, f=fp.name)
        print(OmegaConf.to_yaml(job_data))
        eval_loop(job_data)

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    configure_jobs()