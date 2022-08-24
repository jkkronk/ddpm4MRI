"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py
"""
import enum
import math
import wandb
import numpy as np
import torch as th

from .nn import mean_flat
from improved_diffusion.mri_util import FFT_expand, iFFT_reduce, iFFT_RSS, scale_to_PIL, FFT

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        # Set hyperparameters for variance schedule read first section of
        # Experiments Ho et al.
        beta_start = scale * 0.0001 
        beta_end = scale * 0.02 
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        # Cosine schedule from Improved Denoising Diffusion Probabilistic Models 
        # written by Nichol et al.
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.0005): #0.025
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.

    """

    def __init__(
        self,
        *,
        betas,
    ):

        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        self.alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod_one_minus = np.sqrt(1.0 / self.alphas_cumprod - 1)

        self.sqrt_recip_alpha = np.sqrt(1/(self.alphas))
        self.beta_sqrt_recip_alpha_sqrt_one_m_alpha_cumprod = self.sqrt_recip_alpha * betas / np.sqrt(1-self.alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(self.alphas)
            / (1.0 - self.alphas_cumprod)
        )

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def q_sample(self, x_start, t, mask, noise):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param mask: Undersampling mask
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        return (mask * x_start + 
                (1-mask) * (
                _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
                + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
            )
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean =(_extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t)
        
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )

        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
        self, model, x_t, t, mask, model_kwargs=None
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param mask: Undersampling mask.
       
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x_t.shape[:2]
        assert t.shape == (B,)

        model_output = (1-mask) * model(x_t, t, mask, **model_kwargs)
        
        model_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        model_log_variance = _extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)

        model_mean = (1-mask) * self._pred_x_next(x_t,t,model_output) 

        assert (
            model_mean.shape == model_log_variance.shape == model_variance.shape == x_t.shape
        )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
        }

    def _pred_x_next(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return(_extract_into_tensor(self.sqrt_recip_alpha, t, x_t.shape) * x_t
            - _extract_into_tensor(self.beta_sqrt_recip_alpha_sqrt_one_m_alpha_cumprod, t, x_t.shape) * eps
        ) 

    def p_sample(
        self, model, x, ksp, t, mask, model_kwargs=None, no_model=False,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param ksp_us: Raw kspace, During real time inference us_ksp.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param mask: undersampling mask.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param no_model: Run reversed process knowing x_0 (gt inference)

        :return: sample: sample a random sample from the model
        """

        noise = (1-mask) * th.normal(ksp.mean(), ksp.std(), size=ksp.shape, device=ksp.device)  

        if not no_model:
            out = self.p_mean_variance(
                model,
                x,
                t,
                mask,
                model_kwargs=model_kwargs,
            )
            
        nonzero_mask = ((t > 0).double().view(-1, *([1] * (len(x.shape) - 1))))  # no noise when t == 0
        
        if no_model:
            sample = self.q_sample(ksp, t, mask, noise=noise)
        else:
            sample = out["mean"] + noise * nonzero_mask * th.exp(0.5 * out["log_variance"])  

        return sample

    def p_sample_loop(
        self,
        model,
        ksp_us,
        ksp,
        mask,
        noise=None,
        model_kwargs=None,
        device=None,
        progress=False,
        no_model=False,
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param ksp_us: undersampled kspace data.
        :param ksp: kspace data.
        :param mask: undersampling mask.
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :param no_model: if True, sampling run without model, ie. gt inference.

        :return: a non-differentiable batch of samples.
        """
        final = None
        for sample in self.p_sample_loop_progressive(
            model,
            ksp_us,
            ksp,
            mask,
            noise=noise,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            no_model=no_model,
        ):
            final = sample
        return final

    def p_sample_loop_progressive(
        self,
        model,
        ksp_us,
        ksp,
        mask,
        noise=None,
        model_kwargs=None,
        device=None,
        progress=False,
        no_model=True,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        
        # Create noise
        noise = (1-mask) * th.normal(ksp_us.mean(), ksp_us.std(), size=ksp_us.shape, device=ksp_us.device) 
        uT_x_i = noise + ksp_us 

        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        out_rss_log = iFFT_RSS(uT_x_i).detach().cpu().numpy()
        gt_rss_log = iFFT_RSS(ksp).detach().cpu().numpy()
        out_log_ksp = th.log(th.abs(th.view_as_complex(th.moveaxis(uT_x_i, 1, -1).contiguous()))[:,0]).detach().cpu().numpy()

        wandb.log({'x_t':wandb.Image(scale_to_PIL(out_rss_log)),
                'gt_rss':wandb.Image(scale_to_PIL(gt_rss_log)), 
                'Fx_t_c0':wandb.Image(scale_to_PIL(out_log_ksp)),
                'mse_U_ksp': mean_flat(((1-mask)*(ksp - uT_x_i))**2),
                'mse_img': np.mean((gt_rss_log - out_rss_log)**2)
        })

    
        uT_x_i = uT_x_i.to(device)
        mask = mask.to(device)
        ksp_us = ksp_us.to(device)
        ksp = ksp.to(device)
        
        for i in indices:
            t = th.tensor([i] * ksp_us.shape[0], device=device)
            
            with th.no_grad():
                out = self.p_sample(
                    model,
                    uT_x_i,
                    ksp if no_model else ksp_us,
                    t,
                    mask,
                    model_kwargs=model_kwargs, 
                    no_model = no_model, 
                )
                
                # DC 
                uT_x_i =  (1-mask) * out + mask * uT_x_i 

                out_rss_log = iFFT_RSS(uT_x_i).detach().cpu().numpy()  
                out_log_ksp = th.log(th.abs(th.view_as_complex(th.moveaxis(uT_x_i, 1, -1).contiguous()))[:,0]).detach().cpu().numpy()


                wandb.log({
                    'x_t':wandb.Image(scale_to_PIL(out_rss_log)), 
                    'x_t_ksp':wandb.Image(scale_to_PIL(out_log_ksp)), 
                    'mse_U_ksp': mean_flat(((1-mask)*(ksp - uT_x_i))**2),
                    'mse_img': np.mean((gt_rss_log - out_rss_log)**2)
                    })

                yield uT_x_i

    def training_losses(self, model, x_start, t, mask, model_kwargs=None):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param mask: a batch of undersampling masks.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        if model_kwargs is None:
            model_kwargs = {}
            
        ksp_us = mask*x_start
        noise = (1-mask) * th.normal(ksp_us.mean(), ksp_us.std(), size=ksp_us.shape, device=ksp_us.device) 
        x_t = self.q_sample(x_start, t, mask, noise=noise)

        model_output = (1-mask) * model(x_t, t, mask)

        terms = {}

        assert model_output.shape == noise.shape 
        
        target = noise #+ mask * self.q_posterior_mean_variance(x_start=x_start, x_t=x_t, t=t)[0]# #noise + ksp_us #

        terms["mse"] = mean_flat((target-model_output)**2)
        #gt_c = th.view_as_complex(th.moveaxis(noise, 1, -1).contiguous())
        #model_c = th.view_as_complex(th.moveaxis(model_output, 1, -1).contiguous())
        terms["loss"] = mean_flat(th.abs(target-model_output)) 

        return terms


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].double()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)
