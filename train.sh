#!/usr/bin/bash

#SBATCH --account=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=V100_32GB
#SBATCH --time=4:00:00

module load cuda cudnn anaconda
conda activate contrastive-optimization

export PYTHONPATH=/home/jsetpal/git/contrastive-optimization/

cd ~/git/pytorch-cifar

python main.py
