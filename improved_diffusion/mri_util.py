import torch as th
import numpy as np
import cv2

def FFT_expand(x, cm, norm='ortho'):
    # inp: [nx, ny]
    # out: [nx, ny, ns]
    cm_tmp = th.view_as_complex(th.moveaxis(cm, 1, -1).contiguous())
    x_tmp = th.view_as_complex(th.moveaxis(x, 1, -1).contiguous())
    fft_x = th.fft.fftshift(th.fft.fftn(th.fft.ifftshift(cm_tmp * x_tmp.unsqueeze(1), dim=(-1,-2)), dim=(-1,-2), norm=norm), dim=(-1,-2))

    return th.moveaxis(th.view_as_real(fft_x), -1, 1)

def iFFT_reduce(x, cm, norm='ortho'):
    # inp: [nx, ny, ns]
    # out: [nx, ny]
    
    img_x = th.fft.fftshift(th.fft.ifftn(th.fft.ifftshift(th.view_as_complex(th.moveaxis(x, 1, -1).contiguous()), dim=(-1,-2)), dim=(-1,-2), norm=norm), dim=(-1,-2))
    cm_tmp = th.view_as_complex(th.moveaxis(cm, 1, -1).contiguous())
    img_x = th.moveaxis(th.view_as_real(th.sum(img_x * th.conj(cm_tmp), axis=-3) / (th.sum(cm_tmp * th.conj(cm_tmp), axis=-3)+1e-6)), -1, 1)
    return img_x

def iFFT_RSS(x, norm='ortho'):
    img_x = th.fft.fftshift(th.fft.ifftn(th.view_as_complex(th.moveaxis(x, 1, -1).contiguous()), dim=(-1,-2), norm=norm), dim=(-1,-2))
    return th.sqrt(th.sum(th.pow(th.abs(img_x),2),axis=-3))

def FFT(x, norm='ortho'):
    # inp: [nx, ny]
    # out: [nx, ny, ns]
    x_tmp = th.view_as_complex(th.moveaxis(x, 1, -1).contiguous())
    fft_x = th.fft.fftshift(th.fft.fftn(th.fft.ifftshift(x_tmp, dim=(-1,-2)), dim=(-1,-2), norm=norm), dim=(-1,-2))

    return th.moveaxis(th.view_as_real(fft_x), -1, 1)

def iFFT(x, norm='ortho'):
    # inp: [nx, ny]
    # out: [nx, ny, ns]
    x_tmp = th.view_as_complex(th.moveaxis(x, 1, -1).contiguous())
    fft_x = th.fft.fftshift(th.fft.ifftn(th.fft.ifftshift(x_tmp, dim=(-1,-2)), dim=(-1,-2), norm=norm), dim=(-1,-2))

    return th.moveaxis(th.view_as_real(fft_x), -1, 0)

def get_mask(us_rate=4, num_samps_pdir=320, num_acl=24):
    mask = np.zeros((1,1,1,np.around(num_samps_pdir/2).astype(np.uint)))
    mask[:,:,:,:np.around(num_acl/2).astype(np.uint)] = 1
    num_half_acl = np.sum(mask)
    num_samps = int((num_samps_pdir/us_rate)/2)
    
    num_samps = num_samps - num_half_acl
    accel_samples =  np.around(np.arange(num_half_acl, mask.shape[-1], num_samps-3)).astype(np.uint)

    mask[:,:,:,accel_samples] = 1
    mask = np.concatenate((np.flip(mask, axis=-1),mask), axis=-1)
    
    if (num_samps_pdir % 2) != 0:
        mask_tmp = np.zeros((1,1,1,num_samps_pdir))
        mask_tmp[:,:,:,:mask.shape[-1]] =  mask
        mask = mask_tmp

    return mask

def scale_to_PIL(arr):
    return ((arr - arr.min()) * (1/(arr.max() - arr.min()) * 255)).astype('uint8')
