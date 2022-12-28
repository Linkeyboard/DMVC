import random
import torch
import torch.nn.functional as F

def random_crop_and_pad_image_and_labels(image, labels, size):
    combined = torch.cat([image, labels], 0)
    last_image_dim = image.size()[0]
    image_shape = image.size()
    combined_pad = F.pad(combined, (0, max(size[1], image_shape[2]) - image_shape[2], 0, max(size[0], image_shape[1]) - image_shape[1]))
    freesize0 = random.randint(0, max(size[0], image_shape[1]) - size[0])
    freesize1 = random.randint(0,  max(size[1], image_shape[2]) - size[1])
    combined_crop = combined_pad[:, freesize0:freesize0 + size[0], freesize1:freesize1 + size[1]]
    return (combined_crop[:last_image_dim, :, :], combined_crop[last_image_dim:, :, :])

def random_crop_frames(rec_frames, org_frames, crop_size):
    img_h, img_w = rec_frames.shape[-2], rec_frames.shape[-1]
    crop_h_start = random.randint(0, img_h - crop_size[0])
    crop_w_start = random.randint(0, img_w - crop_size[1])
    crop_rec_frames = rec_frames[:, :, crop_h_start : crop_h_start + crop_size[0], crop_w_start : crop_w_start + crop_size[1]]
    crop_org_frames = org_frames[:, :, crop_h_start : crop_h_start + crop_size[0], crop_w_start : crop_w_start + crop_size[1]]
    return crop_rec_frames, crop_org_frames

def random_flip(images, labels):
    
    # augmentation setting....
    horizontal_flip = 1
    vertical_flip = 1
    transforms = 1


    if transforms and vertical_flip and random.randint(0, 1) == 1:
        images = torch.flip(images, [1])
        labels = torch.flip(labels, [1])
    if transforms and horizontal_flip and random.randint(0, 1) == 1:
        images = torch.flip(images, [2])
        labels = torch.flip(labels, [2])


    return images, labels

def random_flip_frames(rec_frames, org_frames):
    #vertical_flip = random.randint(0, 1)
    vertical_flip = 0
    horizontal_flip = random.randint(0, 1)
    
    if vertical_flip:
        rec_frames = torch.flip(rec_frames, [-2])
        org_frames = torch.flip(org_frames, [-2])
    if horizontal_flip:
        rec_frames = torch.flip(rec_frames, [-1])
        org_frames = torch.flip(org_frames, [-1])

    return rec_frames, org_frames
