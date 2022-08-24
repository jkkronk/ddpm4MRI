import copy
import functools
import os
import wandb
import blobfile as bf
import numpy as np
import torch as th

from torch.optim import AdamW
from improved_diffusion.mri_util import FFT_expand, iFFT_reduce, iFFT_RSS, scale_to_PIL, FFT
from . import gaussian_diffusion as gd
from . import logger
from .nn import update_ema, mean_flat, zero_grad
from .resample import LossAwareSampler, UniformSampler

class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        init_log_loss_scale=20,
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size 

        self.model_params = list(self.model.parameters())
        self.master_params = self.model_params
        self.lg_loss_scale = init_log_loss_scale
        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()
        
        self.opt = AdamW(self.master_params, lr=self.lr, weight_decay=self.weight_decay)
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
            
        else:
            self.ema_params = [
                copy.deepcopy(self.master_params) for _ in range(len(self.ema_rate))
            ]

        if self.sync_cuda:
            self.use_ddp = False
            self.model = self.model.to('cuda:0')


    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            
            logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
            self.model.load_state_dict(
                    th.load(resume_checkpoint)
            )

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            
            logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
            state_dict = self.model.load_state_dict(
                th.load(ema_checkpoint)
            )
            ema_params = self._state_dict_to_master_params(state_dict)
        #dist_util.sync_params(ema_import IPython ; IPython.embed() ; exit()params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = th.load(opt_checkpoint)
            self.opt.load_state_dict(state_dict)

    def run_loop(self):
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):
            batch, batch_us, mask = next(self.data)
            self.run_step(batch, batch_us, mask)
        
            if self.step % self.log_interval == 0:
                logger.dumpkvs()
            if self.step % self.save_interval == 0:
                self.save()
                self.run_step_log(batch.to('cuda:0'), batch_us.to('cuda:0'), mask.to('cuda:0'))
                
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step_log(self, batch, batch_us, mask):
        t, weights = self.schedule_sampler.sample(batch.shape[0], batch.device)
        
        ksp_us = mask*batch
        noise = (1-mask) * th.normal(ksp_us.mean(), ksp_us.std(), size=ksp_us.shape, device=ksp_us.device) #th.randn_like(ksp_us[:,:,:,:,:])
        
        x_t = self.diffusion.q_sample(batch, t, mask, noise=noise)
        model_output = (1-mask) * self.model(x_t, t, mask)

        target = noise 
        mse = mean_flat((target - model_output) ** 2)
        rss_batch = iFFT_RSS(batch).detach().cpu().numpy()
        rss_xt = iFFT_RSS(x_t).detach().cpu().numpy()
        rss_target = iFFT_RSS(target).detach().cpu().numpy()
        rss_model = iFFT_RSS(model_output).detach().cpu().numpy()

        wandb.log({
            'image MSE': mse,
            'x_0':wandb.Image(scale_to_PIL(rss_batch)),
            'x_t':wandb.Image(scale_to_PIL(rss_xt), caption=str(t[0].detach().cpu().numpy())),
            'noise_t':wandb.Image(scale_to_PIL(rss_target), caption=str(t[0].detach().cpu().numpy())),
            'model_t':wandb.Image(scale_to_PIL(rss_model), caption=str(t[0].detach().cpu().numpy())),
            })

    def run_step(self, batch, batch_us, mask):
        self.forward_backward(batch, batch_us, mask)
        self.optimize_normal()
        self.log_step()

    def forward_backward(self, batch, cm, mask):
        zero_grad(self.model_params)
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to('cuda:0')
            micro_cm = cm[i : i + self.microbatch].to('cuda:0')
            micro_mask = mask[i : i + self.microbatch].to('cuda:0')

            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], 'cuda:0')

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.model,
                micro,
                t,
                micro_cm,
                micro_mask,
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )

            loss.backward()

    ##### HELPER         
    def optimize_normal(self):
        self._log_grad_norm()
        self._anneal_lr()
        self.opt.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)

    def _log_grad_norm(self):
        sqsum = 0.0
        for p in self.master_params:
            sqsum += (p.grad ** 2).sum().item()
        logger.logkv_mean("grad_norm", np.sqrt(sqsum))

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self):
        def save_checkpoint(rate, params):
            #import IPython ; IPython.embed() ; exit()
            state_dict = self._master_params_to_state_dict(params)
            logger.log(f"saving model {rate}...")
            if not rate:
                filename = f"model{(self.step+self.resume_step):06d}.pt"
            else:
                filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
            with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                th.save(state_dict, f)

        save_checkpoint(0, self.master_params)
        try: 
            for rate, params in zip(self.ema_rate, self.ema_params):
                save_checkpoint(rate, list(params))
        except Exception: 
            pass

        with bf.BlobFile(
            bf.join(get_blob_logdir(), f"opt{(self.step+self.resume_step):06d}.pt"),
            "wb",
        ) as f:
            th.save(self.opt.state_dict(), f)

    def _master_params_to_state_dict(self, master_params):
        state_dict = self.model.state_dict()
        for i, (name, _value) in enumerate(self.model.named_parameters()):
            assert name in state_dict
            state_dict[name] = master_params[i]
        return state_dict

    def _state_dict_to_master_params(self, state_dict):
        params = self.model.named_parameters() #[state_dict[name] for name, _ in ]
        
        
        return params


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0

def get_blob_logdir():
    return os.environ.get("DIFFUSION_BLOB_LOGDIR", logger.get_dir())

def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None

def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None

def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
