#!/bin/bash

MY_PYTHON="python"
MNIST="--n_layers 2 --n_hiddens 100 --data_path data/ --save_path results/ --batch_size 10 --log_every 100 --samples_per_task 1000 --data_file mnist_rotations.pt    --cuda no  --seed 0"
CIFAR_100="--n_layers 2 --n_hiddens 100 --data_path data/ --save_path results/ --batch_size 10 --log_every 100 --samples_per_task 2500 --data_file cifar100.pt           --cuda yes --seed 0"


# Download mnist, cifar10, cifar100
cd datasets/
cd raw/
$MY_PYTHON downloads.py
cd ..

# Build dataset
$MY_PYTHON mnist_permutations.py \
	--o mnist_permutations.pt \
	--seed 0 \
	--n_tasks 20

$MY_PYTHON mnist_rotations.py \
	--o mnist_rotations.pt\
	--seed 0 \
	--min_rot 0 \
	--max_rot 180 \
	--n_tasks 20

$MY_PYTHON cifar100.py \
	--o cifar100.pt \
	--seed 0 \
	--n_tasks 20

cd ..

# experiment example test
$MY_PYTHON main.py $MNIST --model single --lr 0.003

# experiment example 2
$MY_PYTHON main.py $CIFAR_100 --model single --lr 0.03

# You can add more experiments below ...
$MY_PYTHON main.py $CIFAR_100 --model single --lr test.0

# plot results
cd results/
$MY_PYTHON plot_results.py
cd ..