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

export PYTHONPATH='/iris/u/kayburns/new_arch/mvp/'
python hydra_eval_launcher.py hydra/launcher=local hydra/output=local \
