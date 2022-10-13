#!/bin/bash
#SBATCH --partition=iris-hi
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --exclude=iris4,iris2,iris-hp-z8
#SBATCH --job-name="fancy new architecture"
#SBATCH --time=4-0:0

source /sailhome/kayburns/.bashrc
conda activate py3.8_torch1.10.1
cd /iris/u/kayburns/new_arch/r3m/evaluation/r3meval/core/
# python r3meval/core/hydra_launcher.py hydra/launcher=local hydra/output=local \
#     env="kitchen_sdoor_open-v3" camera="left_cap2" pixel_based=true \
#     embedding=resnet50 num_demos=5 env_kwargs.load_path=r3m \
#     bc_kwargs.finetune=false proprio=9 job_name=r3m_repro seed=125

# fine-tune all heads, last layer
# python r3meval/core/hydra_launcher.py hydra/launcher=local hydra/output=local \
#     env="kitchen_sdoor_open-v3" camera="left_cap2" pixel_based=true \
#     embedding=dino num_demos=5 env_kwargs.load_path=dino \
#     bc_kwargs.finetune=true proprio=9 job_name=r3m_repro_all seed=123

# python hydra_launcher.py --multirun hydra/launcher=local hydra/output=local \
#     env=["kitchen_knob1_on-v3","kitchen_light_on-v3","kitchen_sdoor_open-v3","kitchen_ldoor_open-v3","kitchen_micro_open-v3"] \
#     camera=["default","left_cap2","right_cap2"] \
#     pixel_based=true embedding=resnet50 num_demos=25 env_kwargs.load_path=r3m \
#     bc_kwargs.finetune=false proprio=9 job_name=try_r3m

# python hydra_launcher.py --multirun hydra/launcher=local hydra/output=local \
#     env="kitchen_sdoor_open-v3" camera="left_cap2" \
#     pixel_based=true embedding=resnet50 env_kwargs.load_path=r3m \
#     bc_kwargs.finetune=true proprio=9 job_name=try_r3m

for env in kitchen_knob1_on-v3 kitchen_light_on-v3 kitchen_sdoor_open-v3 kitchen_ldoor_open-v3 kitchen_micro_open-v3 kitchen_ldoor_open-v3
do
    for num_demos in 5 10 25
    do
        for camera in default left_cap2 right_cap2
        do
            for seed in 123 124 125
            do
                python hydra_launcher.py hydra/launcher=local hydra/output=local \
                    pixel_based=true embedding=dino env_kwargs.load_path=dino \
                    bc_kwargs.finetune=true proprio=9 job_name=try_r3m \
                    seed=$seed num_demos=$num_demos env=$env camera=$camera
            done
        done
    done
done


# for env in bin-picking-v2-goal-observable #assembly-v2-goal-observable #bin-picking-v2-goal-observable button-press-topdown-v2-goal-observable drawer-open-v2-goal-observable hammer-v2-goal-observable
# do
#     for camera in left_cap2 #right_cap2 top_cap2 left_cap2 right_cap2
#     do
#         for num_demos in 25 #10 5
#         do
#             for seed in 123 124 125
#             do
#                 python hydra_launcher.py hydra/launcher=local hydra/output=local \
#                     pixel_based=true embedding=dino_fix_norm env_kwargs.load_path=dino_fix_norm \
#                     bc_kwargs.finetune=true proprio=4 job_name=try_r3m \
#                     seed=$seed num_demos=$num_demos env=$env camera=$camera
#             done
#         done
#     done
# done
