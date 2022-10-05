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
#     bc_kwargs.finetune=false proprio=9 job_name=r3m_repro seed=125 \

python r3meval/core/hydra_launcher.py hydra/launcher=local hydra/output=local \
    env="kitchen_sdoor_open-v3" camera="left_cap2" pixel_based=true \
    embedding=dino-3 num_demos=5 env_kwargs.load_path=dino-3 \
    bc_kwargs.finetune=false proprio=9 job_name=eval_3 seed=125 \
    eval.eval=True env_kwargs.shift=bottom_left_white_rectangle \
    bc_kwargs.load_path=/iris/u/kayburns/new_arch/r3m/evaluation/outputs/BC_pretrained_rep/2022-09-28_23-47-08/r3m_repro/iterations/policy_2857.pickle # 52, 66 -> 24

    # head 2
    # bc_kwargs.load_path="/iris/u/kayburns/new_arch/r3m/evaluation/outputs/BC_pretrained_rep/2022-10-04_11-11-50/r3m_repro/iterations/policy_2857.pickle" # 42, 42
    # head 3
    # bc_kwargs.load_path=/iris/u/kayburns/new_arch/r3m/evaluation/outputs/BC_pretrained_rep/2022-09-28_23-47-08/r3m_repro/iterations/policy_2857.pickle # 52, 66 -> 24
    # bc_kwargs.load_path=/iris/u/kayburns/new_arch/r3m/evaluation/outputs/BC_pretrained_rep/2022-10-03_17-57-00/r3m_repro/iterations/policy_2857.pickle # 44
    # r3m
    # bc_kwargs.load_path="/iris/u/kayburns/new_arch/r3m/evaluation/outputs/BC_pretrained_rep/2022-10-04_11-22-39/r3m_repro/iterations/policy_2857.pickle"


# head 2
# head 3
# load_path       :   "/iris/u/kayburns/new_arch/r3m/evaluation/outputs/BC_pretrained_rep/2022-09-28_23-47-08/r3m_repro/iterations/policy_999.pickle"


