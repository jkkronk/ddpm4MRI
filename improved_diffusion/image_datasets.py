from PIL import Image
import blobfile as bf
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch as th
import h5py
from improved_diffusion.mri_util import get_mask, iFFT_RSS, FFT

def load_data(
    *, data_dir, batch_size, deterministic=False, single_coil=False,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files = _list_image_files_recursively(data_dir)
    
    dataset = ImageDataset(
        all_files, single_coil=single_coil,
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True,
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["h5"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results

class ImageDataset(Dataset):
    def __init__(self, image_paths, shard=0, num_shards=1, us_rate=8, single_coil=False):
        super().__init__()
        self.local_images = image_paths[shard:][::num_shards]
        self.us_rate = us_rate
        self.single_coil = single_coil
        if self.us_rate == 8:
            self.acl_frac = 0.04
        else:
            self.acl_frac = 0.08


    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]

        try:
            with h5py.File(path, "r") as f: 
                ksp = f['kspace'][:]
                if self.single_coil:
                    ksp = th.from_numpy(np.concatenate((np.array(ksp.real,ndmin=4),np.array(ksp.imag,ndmin=4)),axis=0)).unsqueeze(0)
                    rss_img = th.from_numpy(np.concatenate((iFFT_RSS(ksp).detach().cpu().numpy(),np.zeros_like(iFFT_RSS(ksp).detach().cpu().numpy())),axis=0)).unsqueeze(0)
                    ksp = FFT(rss_img)[:,0].detach().cpu().numpy() + 1j * FFT(rss_img)[:,1].detach().cpu().numpy() 
                    
            mask = get_mask(self.us_rate,ksp.shape[-1],int(ksp.shape[-1]*self.acl_frac))
        except AssertionError:
            print('path not found...', path)
            raise 

        try:
            assert mask.shape[-1] == ksp.shape[-1]
        except AssertionError:
            print('Mask and ksp shape not equal!')
            print(mask.shape, ksp.shape)
            raise

        ## Normalization of absolute value to [0,1]
        # Maybe look at this normalization again. Might not be optimal. See Ho et al. 3.3
        abs_ksp = (np.abs(ksp) - np.min(np.abs(ksp))) / (np.max(np.abs(ksp))-np.min(np.abs(ksp)))
        ksp = abs_ksp * np.exp(1j * np.angle(ksp))
        
        ksp_us = ksp*mask
        ksp_us = np.concatenate((np.array(ksp_us.real,ndmin=4),np.array(ksp_us.imag,ndmin=4)),axis=0)
        ksp = np.concatenate((np.array(ksp.real,ndmin=4),np.array(ksp.imag,ndmin=4)),axis=0)

        return ksp, ksp_us, mask
