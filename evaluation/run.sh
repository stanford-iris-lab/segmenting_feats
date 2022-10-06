#!/bin/bash
#SBATCH --partition=iris-hi
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --exclude=iris4,iris2,iris-hp-z8
#SBATCH --job-name="fancy new architecture"
#SBATCH --time=3-0:0

source /sailhome/kayburns/.bashrc
conda activate py3.8_torch1.10.1
cd /iris/u/kayburns/new_arch/r3m/evaluation/
# python r3meval/core/hydra_launcher.py hydra/launcher=local hydra/output=local \
#     env="kitchen_sdoor_open-v3" camera="left_cap2" pixel_based=true \
#     embedding=resnet50 num_demos=5 env_kwargs.load_path=r3m \
#     bc_kwargs.finetune=false proprio=9 job_name=r3m_repro seed=125

# fine-tune all heads, last layer
cd /iris/u/kayburns/new_arch/r3m/evaluation/
python r3meval/core/hydra_launcher.py hydra/launcher=local hydra/output=local \
    env="kitchen_sdoor_open-v3" camera="left_cap2" pixel_based=true \
    embedding=dino num_demos=5 env_kwargs.load_path=dino \
    bc_kwargs.finetune=true proprio=9 job_name=r3m_repro_all seed=123

# blind baseline
# cd /iris/u/kayburns/new_arch/r3m/evaluation/
# python r3meval/core/hydra_launcher.py hydra/launcher=local hydra/output=local \
#     env="kitchen_sdoor_open-v3" camera="left_cap2" pixel_based=true \
#     embedding=ignore_input num_demos=5 env_kwargs.load_path=ignore_input \
#     bc_kwargs.finetune=false proprio=9 job_name=r3m_repro_random seed=125