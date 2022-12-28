import numpy as np
import torch

def reshape_patch(img_tensor, patch_size):
    batch_size = img_tensor.shape[0]
    seq_length = img_tensor.shape[1]
    img_height = img_tensor.shape[2]
    img_width = img_tensor.shape[3]
    num_channels = img_tensor.shape[4]
    a = torch.reshape(img_tensor, [batch_size, seq_length,
                                img_height//patch_size, patch_size,
                                img_width//patch_size, patch_size,
                                num_channels])
    b = a.permute(0, 1, 2, 4, 3, 5, 6)
    patch_tensor = torch.reshape(b, [batch_size, seq_length,
                                  img_height//patch_size,
                                  img_width//patch_size,
                                  patch_size*patch_size*num_channels])
    return patch_tensor

def reshape_patch_back(patch_tensor, patch_size):
    batch_size = patch_tensor.shape[0]
    seq_length = patch_tensor.shape[1]
    patch_height = patch_tensor.shape[2]
    patch_width = patch_tensor.shape[3]
    channels = patch_tensor.shape[4]
    img_channels = channels // (patch_size*patch_size)
    a = torch.reshape(patch_tensor, [batch_size, seq_length,
                                  patch_height, patch_width,
                                  patch_size, patch_size,
                                  img_channels])
    b = a.permute(0, 1, 2, 4, 3, 5, 6)
    img_tensor = torch.reshape(b, [batch_size, seq_length,
                                patch_height * patch_size,
                                patch_width * patch_size,
                                img_channels])
    return img_tensor

def derive_psnr(mse, maxx = 1):
    if mse == 0:
        return 99
    return 10 * np.log10(maxx * maxx / mse)