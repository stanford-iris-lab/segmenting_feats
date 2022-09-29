#!/bin/bash
#SBATCH --partition=iris-hi
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --exclude=iris4,iris2,iris-hp-z8
#SBATCH --job-name="eval"
#SBATCH --time=3-0:0

cd /iris/u/kayburns/new_arch/r3m/evaluation/
python r3meval/core/hydra_launcher.py hydra/launcher=local hydra/output=local \
    env="kitchen_sdoor_open-v3" camera="left_cap2" pixel_based=true \
    embedding=resnet50 num_demos=5 env_kwargs.load_path=r3m \
    bc_kwargs.finetune=false proprio=9 job_name=r3m_repro seed=125