import argparse
import inspect

from . import gaussian_diffusion as gd
from .unet import UNetModel
from .gaussian_diffusion import GaussianDiffusion


def model_and_diffusion_defaults():
    """
    Defaults for image training.
    """
    return dict(
        num_channels=128,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        attention_resolutions="16,8",
        dropout=0.0,
        diffusion_steps=1000,
        noise_schedule="linear",#cosine",
        use_checkpoint=False,
        single_coil=False,
    )


def create_model_and_diffusion(
    num_channels,
    num_res_blocks,
    num_heads,
    num_heads_upsample,
    attention_resolutions,
    dropout,
    diffusion_steps,
    noise_schedule,
    use_checkpoint,
    single_coil,
):
    model = create_model(
        num_channels,
        num_res_blocks,
        use_checkpoint=use_checkpoint,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        dropout=dropout,
        single_coil=single_coil,
    )
    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        noise_schedule=noise_schedule,
    )
    return model, diffusion


def create_model(
    num_channels,
    num_res_blocks,
    use_checkpoint,
    attention_resolutions,
    num_heads,
    num_heads_upsample,
    dropout,
    single_coil
):
    image_size = 320
    channel_mult = (1, 1, 2, 2, 2, 2)

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    return UNetModel(
        in_channels=2, 
        model_channels=num_channels,
        out_channels=2, 
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        use_checkpoint=use_checkpoint,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        single_coil=single_coil,
    )

def create_gaussian_diffusion(
    *,
    steps=1000,
    noise_schedule="linear",
):
    betas = gd.get_named_beta_schedule(noise_schedule, steps)
    
    return GaussianDiffusion(
        betas=betas,
        )


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")
