"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os
import wandb
import numpy as np
import torch as th
import h5py
from improved_diffusion import logger
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from improved_diffusion.mri_util import FFT_expand, iFFT_reduce, iFFT_RSS, get_mask, scale_to_PIL, FFT


def main():
    args = create_argparser().parse_args()

    logger.configure(dir=args.log_dir)
    
    wandb.login()

    wandb.init(project='DDPM_MRI_inference')
    wandb.config.update(args)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(th.load(args.model_path), strict=False)
    
    wandb.watch(model)

    if th.cuda.is_available():
        model = model.to('cuda:0')

    model.eval()

    logger.log("sampling...")

    all_images = os.listdir(args.sample_path)


                
    for subj in all_images:
        print('subj: ', subj)

        try:
            with h5py.File(args.sample_path+subj, "r") as f: 
                ksp = f['kspace'][:]
                if args.single_coil:
                    ksp = th.from_numpy(np.concatenate((np.array(ksp.real,ndmin=4),np.array(ksp.imag,ndmin=4)),axis=0)).unsqueeze(0)
                    rss_img = th.from_numpy(np.concatenate((iFFT_RSS(ksp).detach().cpu().numpy(),np.zeros_like(iFFT_RSS(ksp).detach().cpu().numpy())),axis=0)).unsqueeze(0)
                    ksp = FFT(rss_img)[:,0].detach().cpu().numpy() + 1j * FFT(rss_img)[:,1].detach().cpu().numpy()
            mask = get_mask(args.us_rate,ksp.shape[-1],int(ksp.shape[-1]*args.acl_frac))
            ksp_us = ksp*mask 
        except AssertionError:
            print('path not found...', args.sample_path+subj)
            raise 

        # Normalise 
        abs_ksp = (np.abs(ksp) - np.min(np.abs(ksp))) / (np.max(np.abs(ksp))-np.min(np.abs(ksp)))
        ksp = abs_ksp * np.exp(1j * np.angle(ksp))
        
        ksp =  th.from_numpy(np.concatenate((np.array(ksp.real,ndmin=4),np.array(ksp.imag,ndmin=4)),axis=0)).unsqueeze(0)
        mask = th.from_numpy(mask).unsqueeze(0)

        ksp_us_norm = mask * ksp

        model_kwargs = {}
        sample = diffusion.p_sample_loop(
            model,
            ksp_us_norm,
            ksp,
            mask,
            model_kwargs=model_kwargs,
            no_model=args.no_model,
        )
        import IPython ; IPython.embed() ; exit()
        exit() #Just no. Dont go further just yet

    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        num_samples=1,
        batch_size=1,
        model_path="",
        sample_path="/scratch_net/bmicdl03_second/jonatank/data/fastMRI_train_mid/train/", #"/scratch_net/bmicdl03/jonatank/data/singlecoil_train/", 
        us_rate=8,
        acl_frac=0.04,
        log_dir='/scratch_net/bmicdl03/jonatank/DDPM_log/inference/',
        no_model=False,
        single_coil=False,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
