#!/bin/bash

MY_PYTHON="python"

# experiment settings
EXP1="--n_layers 2 --n_hiddens 100 --cuda no  --seed 0"

# conduct main.py
$MY_PYTHON main.py --o mnist_permutations.pt --seed 0 --n_tasks 20
