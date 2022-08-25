# diffusion models for MRI rec

This is the codebase for diffusion models for MRI rec The code is very much based on [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672). So, if you have any problems check out their repo first...

## Preparing Data

The training data is stored on the cluster at /itet-stor/username/bmicdatasets_bmicnas01/Processed/fastMRI_with_espiritcoils/fastMRI_train_mid

## Training with 
script/image_train.py

## Inference with 
script/image_sample.py
This script is a bit adhoc. Works but needs improvements. 

## Other
Complex values is stored as channels. 

Since fastMRI is different sizes, batchsize of 1 is implemented for now.

If you have any questions, feel free to reach out to me: Jonatan Kronander jonatan@kronander.se

## DDPM papers
First sota paper:
Denoising Diffusion Probabilistic Models
https://proceedings.neurips.cc/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf

Improving DDPMS:  
Improved Denoising Diffusion Probabilistic Models 
https://arxiv.org/abs/2102.09672

Latent spaced DDPMs:
High-Resolution Image Synthesis with Latent Diffusion Models https://arxiv.org/abs/2112.10752


Diffusion for MRI:
Towards performant and reliable undersampled MR reconstruction via diffusion model sampling https://arxiv.org/pdf/2203.04292.pdf
High-Frequency Space Diffusion Models for Accelerated MRI https://arxiv.org/pdf/2208.05481.pdf
Measurement-conditioned Denoising Diffusion Probabilistic Model for Under-sampled Medical Image Reconstruction https://arxiv.org/pdf/2203.03623.pdf
Adaptive Diffusion Priors for Accelerated MRI Reconstruction https://arxiv.org/pdf/2207.05876.pdf

My "paper": 
I wrote a short summary of our plan that didn't work due to sensitivity maps (Ender have more infos on why it didn't work). 
Note that this paper is not read by anyone but me so there is no proof reading done and should be read with a grain of salt. 
You can find it paper.pdf

## Ways forward for this project
I found the the model to work ok for single coil images. Needs to be check more though... 
However, I never got the multicoil setting to work. This is probably because the sensitivity maps, talk to Ender for more information. 
One way forward would be to fix this. Another way forward would be to do the diffusion process in latent space (ie. latent varibles of AE. Read more here https://arxiv.org/abs/2112.10752)
