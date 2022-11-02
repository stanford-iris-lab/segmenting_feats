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

sweep_dir = '/iris/u/kayburns/new_arch/r3m/evaluation/r3meval/core/outputs/main_sweep_1/'

def is_target_task(job_data):
    slide_door = job_data.env == 'kitchen_sdoor_open-v3'
    left_cap = job_data.camera == 'left_cap2'
    return slide_door

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
    # run_paths = [
    #             '/iris/u/kayburns/new_arch/r3m/evaluation/r3meval/core/outputs/main_sweep_1/2022-10-16_17-27-29/', # 123, left_cap2, dino ft
    #             '/iris/u/kayburns/new_arch/r3m/evaluation/r3meval/core/outputs/main_sweep_1/2022-10-17_00-46-40/', # 123, left_cap2, r3m ft
    #             '/iris/u/kayburns/new_arch/r3m/evaluation/r3meval/core/outputs/main_sweep_1/2022-10-16_09-58-24/', # 123, left_cap2, r3m
    #             '/iris/u/kayburns/new_arch/r3m/evaluation/r3meval/core/outputs/BC_pretrained_rep/2022-10-10_15-30-43/', # dino
    #             '/iris/u/kayburns/new_arch/r3m/evaluation/r3meval/core/outputs/main_sweep_1/2022-10-19_11-20-04/', # mvp
    #             '/iris/u/kayburns/new_arch/r3m/evaluation/r3meval/core/outputs/main_sweep_1/2022-10-17_06-21-43/', # mvp
    #             '/iris/u/kayburns/new_arch/r3m/evaluation/r3meval/core/outputs/main_sweep_1/2022-10-19_10-30-13/', # mvp
    #         ]

    for run_path in run_paths:

        old_config_path = os.path.join(sweep_dir, run_path, 'job_config.json')
        embedding_path = os.path.join(sweep_dir, run_path, 'try_r3m/iterations/embedding_best.pickle')
        policy_path = os.path.join(sweep_dir, run_path, 'try_r3m/iterations/policy_best.pickle')

        if not os.path.exists(old_config_path):
            continue

        with open(old_config_path, 'r') as fp:
            old_job_data = OmegaConf.load(fp)

        if not is_target_task(old_job_data):
            continue
        
        job_data.env_kwargs = old_job_data.env_kwargs
        job_data.embedding = old_job_data.embedding
        job_data.num_demos = old_job_data.num_demos
        job_data.camera = old_job_data.camera
        job_data.env = old_job_data.env
        job_data.seed = old_job_data.seed
        job_data.proprio = old_job_data.proprio
        job_data.bc_kwargs.load_path = policy_path
        job_data.bc_kwargs.finetune = old_job_data.bc_kwargs.finetune
        job_data.env_kwargs.load_path = embedding_path

        with open('job_config.json', 'w') as fp:
            OmegaConf.save(config=job_data, f=fp.name)
        print(OmegaConf.to_yaml(job_data))
        eval_loop(job_data)

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    configure_jobs()