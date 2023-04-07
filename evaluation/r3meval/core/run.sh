#!/bin/bash
#SBATCH --partition=iris-hi
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --exclude=iris4,iris2,iris-hp-z8
#SBATCH --job-name="AMPHIBIOUS GAZE IMPROVEMENT"
#SBATCH --time=3-0:0
#SBATCH --account=iris

export ENV_NAME=${1}
export SEED=${2}
export CAM_NAME=${3}
export EMB_NAME=${4}
export LOAD_PATH=${5}
export NUM_DEMOS=10


if [[ "${1}" == *"v2"* ]]; then
    echo "Using proprio=4 for Meta-World environment."
    export PROPRIO=4
else
    echo "Using proprio=9 for FrankaKitchen environment."
    export PROPRIO=9
fi

export PYTHONPATH='/iris/u/kayburns/new_arch/Intriguing-Properties-of-Vision-Transformers/'
source /sailhome/kayburns/.bashrc
conda activate py3.8_torch1.10.1
cd /iris/u/kayburns/new_arch/r3m/evaluation/
python r3meval/core/hydra_launcher.py hydra/launcher=local hydra/output=local \
    env=${ENV_NAME} camera=${CAM_NAME} pixel_based=true \
    embedding=${EMB_NAME} num_demos=${NUM_DEMOS} env_kwargs.load_path=${LOAD_PATH} \
    bc_kwargs.finetune=false proprio=${PROPRIO} job_name=try_r3m seed=${SEED}

# source /sailhome/kayburns/.bashrc
# conda activate py3.8_torch1.10.1
# cd /iris/u/kayburns/new_arch/r3m/evaluation/r3meval/core/

# # export PYTHONPATH='/iris/u/kayburns/new_arch/Intriguing-Properties-of-Vision-Transformers/'
# # for env in kitchen_sdoor_open-v3 kitchen_knob1_on-v3 kitchen_light_on-v3 kitchen_ldoor_open-v3 kitchen_micro_open-v3 # kitchen_sdoor_open-v3 kitchen_knob1_on-v3 kitchen_light_on-v3 kitchen_ldoor_open-v3 kitchen_micro_open-v3
# # # for env in assembly-v2-goal-observable # assembly-v2-goal-observable bin-picking-v2-goal-observable button-press-topdown-v2-goal-observable drawer-open-v2-goal-observable hammer-v2-goal-observable
# # do
# #     for num_demos in 10 # 5 10 25
# #     do
# #         for camera in left_cap2 # default left_cap2 right_cap2
# #         do
# #             for seed in 123 124 125 # 123 124 125
# #             do
# #                 python hydra_launcher.py hydra/launcher=local hydra/output=local \
# #                     pixel_based=true embedding=dino_ensemble env_kwargs.load_path=dino_ensemble \
# #                     bc_kwargs.finetune=true ft_only_last_layer=true proprio=9 job_name=try_r3m \
# #                     seed=$seed num_demos=$num_demos env=$env camera=$camera
# #             done
# #         done
# #     done
# # done

# export PYTHONPATH='/iris/u/kayburns/new_arch/Intriguing-Properties-of-Vision-Transformers/:/iris/u/kayburns/new_arch/'
# for env in kitchen_knob1_on-v3 # kitchen_light_on-v3 kitchen_ldoor_open-v3 kitchen_micro_open-v3 # kitchen_sdoor_open-v3 kitchen_knob1_on-v3 kitchen_light_on-v3 kitchen_ldoor_open-v3 kitchen_micro_open-v3
# do
#     for num_demos in 10 # 5 10 25
#     do
#         for camera in left_cap2
#         do
#             for seed in 123 124 125 # 123 124 125
#             do
#                 python hydra_launcher.py hydra/launcher=local hydra/output=local \
#                     pixel_based=true embedding=keypoints env_kwargs.load_path=keypoints \
#                     bc_kwargs.finetune=false proprio=9 job_name=try_r3m \
#                     seed=$seed num_demos=$num_demos env=$env camera=$camera
#             done
#         done
#     done
# done

# # export PYTHONPATH='/iris/u/kayburns/new_arch/Intriguing-Properties-of-Vision-Transformers/'
# # for env in hammer-v2-goal-observable drawer-open-v2-goal-observable # assembly-v2-goal-observable bin-picking-v2-goal-observable button-press-topdown-v2-goal-observable drawer-open-v2-goal-observable hammer-v2-goal-observable
# # do
# #     for num_demos in 10 # 5 10 25
# #     do
# #         for camera in left_cap2 # default left_cap2 right_cap2
# #         do
# #             for seed in 123 124 125 # 123 124 125
# #             do
# #                 python hydra_launcher.py hydra/launcher=local hydra/output=local \
# #                     pixel_based=true embedding=resnet50_dino env_kwargs.load_path=resnet50_dino \
# #                     bc_kwargs.finetune=false proprio=4 job_name=try_r3m \
# #                     seed=$seed num_demos=$num_demos env=$env camera=$camera
# #             done
# #         done
# #     done
# # done

# # export PYTHONPATH='/iris/u/kayburns/new_arch/Intriguing-Properties-of-Vision-Transformers/'
# # for env in assembly-v2-goal-observable bin-picking-v2-goal-observable button-press-topdown-v2-goal-observable drawer-open-v2-goal-observable hammer-v2-goal-observable # assembly-v2-goal-observable bin-picking-v2-goal-observable button-press-topdown-v2-goal-observable drawer-open-v2-goal-observable hammer-v2-goal-observable
# # do
# #     for num_demos in 10 # 5 10 25
# #     do
# #         for camera in left_cap2 # default left_cap2 right_cap2
# #         do
# #             for seed in 123 124 125 # 123 124 125
# #             do
# #                 python hydra_launcher.py hydra/launcher=local hydra/output=local \
# #                     pixel_based=true embedding=dino_ensemble env_kwargs.load_path=dino_ensemble \
# #                     bc_kwargs.finetune=true ft_only_last_layer=true proprio=4 job_name=try_r3m \
# #                     seed=$seed num_demos=$num_demos env=$env camera=$camera
# #             done
# #         done
# #     done
# # done
