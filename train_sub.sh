#!/bin/sh
#SBATCH --output=/scratch_net/bmicdl03/jonatank/log/%j.out     # where to store the output (%j is the JOBID), subdirectory "log" must exist
#SBATCH --error=/scratch_net/bmicdl03/jonatank/log/%j.err  # where to store error messages
#SBATCH --gres=gpu:1
#SBATCH --constraint='ampere'

python scripts/image_train.py --data_dir /scratch_net/bmicdl03_second/jonatank/data/fastMRI_train_mid/train/ --single_coil True --num_channels 128 --num_res_blocks 4 --diffusion_steps 1000 --noise_schedule linear --lr 1e-4
