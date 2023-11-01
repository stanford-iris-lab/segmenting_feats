# Segmenting Features

Code for evaluating different visual pre-training strategies.

## Installation

To install R3M from an existing conda environment, simply run `pip install -e .` from this directory. 

You can alternatively build a fresh conda env from the r3m_base.yaml file [here](https://github.com/facebookresearch/r3m/blob/main/r3m/r3m_base.yaml) and then install from this directory with `pip install -e .`

## Running Evaluation

To train policies on top of each representation:
```
cd evaluation/r3meval/core/
./run.sh
```

## Testing Transfer

To test transfer with kitchen shift:
```
cd evaluation/r3meval/core/
./eval.sh
```

## License

R3M is licensed under the MIT license.

## Ackowledgements

Adapted from the [R3M](https://github.com/facebookresearch/r3m) codebase.
