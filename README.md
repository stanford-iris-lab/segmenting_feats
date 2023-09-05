# Spatial Features

Code for evaluating different visual pre-training strategies.

## Installation

```
conda activate your_conda_env
alias pip=$(which pip)
pip install git+https://github.com/openai/CLIP.git
```

## Change the following files for your envs/paths
```
evaluation/r3meval/core/run.sh
evaluation/r3meval/core/eval.sh
```
## Change the following files to match your experimental condition
```
evaluation/r3meval/core/launch_run.sh
evaluation/r3meval/core/launch_eval.sh
evaluation/r3meval/core/hydra_eval_launcher.py (sweep_dir=/iris/u/ur_name/...)
```
*Note: include a new run_id when you need to rerun an experiment*


## Running Evaluation

To train policies on top of each representation:
```
cd evaluation/r3meval/core/
./launch_run.sh
```

Run zero-shot performance:
```
cd evaluation/r3meval/core/
./launch_eval.sh
```

## Evaluate segmentation performance 

Use the following script modified from [Intriuing-Properties-of-Vision-Transformers](https://arxiv.org/abs/2105.10497)
```
cd evaluate_segmentation/
./evaluate_segmentation.sh
```
Add your own model by modifying [evaluate_segmentation/utils.py](./evaluate_segmentation/utils.py).

## License

R3M is licensed under the MIT license.

## Ackowledgements

Adapted from the [R3M](https://github.com/facebookresearch/r3m) codebase.
