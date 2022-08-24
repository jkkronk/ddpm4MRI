#!/bin/sh

#SBATCH --output=/scratch_net/bmicdl03/jonatank/log/%j.out     # where to store the output (%j is the JOBID), subdirectory "log" must exist
#SBATCH --error=/scratch_net/bmicdl03/jonatank/log/%j.err  # where to store error messages
#SBATCH --gres=gpu:1
python scripts/image_sample.py --model_path /scratch_net/bmicdl03/jonatank/DDPM_log/train_54/model135000.pt --single_coil True --num_channels 128 --num_res_blocks 3 --rescale_timesteps False --rescale_learned_sigmas False --learn_sigma False --diffusion_steps 1000 --noise_schedule linear

#python scripts/image_sample.py --model_path /scratch_net/bmicdl03/jonatank/DDPM_log//model010000.pt --num_channels 128 --num_res_blocks 3 --diffusion_steps 4000 --noise_schedule cosine --learn_sigma False --batch_size 4
#python scripts/image_sample.py --model_path /scratch_net/bmicdl03/jonatank/logs/DDPM/2022-04-21-11-42-07-634617/model180000558569.pt --image_size 256 --num_channels 128 --num_res_blocks 3 --diffusion_steps 4000 --noise_schedule linear --batch_size 4

#2022-04-20-16-06-00-980013/model580000558031.pt

# --resume_checkpoint /scratch_net/bmicdl03/jonatank/logs/DDPM/model290000.pt --learn_sigma True 

# --job_id %j  --image_size 256 --num_channels 128 --num_res_blocks 3 --diffusion_steps 4000  --batch_size 8 

