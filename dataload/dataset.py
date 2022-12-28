import os
import torch
import logging
import cv2
from PIL import Image
import imageio
import numpy as np
import torch.utils.data as data
from os.path import join, exists
import math
import random
import sys
import json
import random
from dataload.augmentation import random_flip_frames, random_crop_and_pad_image_and_labels, random_crop_frames
import re
from utils.info import classes_dict

class UVGDataSet(data.Dataset):
    def __init__(self, root_dir, rec_dir, test_class, qp):
        self.qp = qp
        self.test_class = test_class
        #self.v_frames = 12
        self.gop_size = 12
        self.clip = []

        for i, seq in enumerate(classes_dict[test_class]["sequence_name"]):
            #num = self.v_frames // self.gop_size
            num = classes_dict[test_class]["frameNum"][i] // self.gop_size
            for j in range(num):
                rec_frames_path = [os.path.join(rec_dir, str(qp), seq, 'im' + str(j * self.gop_size + 1).zfill(3) +'.png')]
                bin_path = os.path.join(rec_dir, str(qp), seq, 'im' + str(j * self.gop_size + 1).zfill(3) +'.bin')
                org_frames_path = []

                for k in range(self.gop_size):
                    input_path = os.path.join(root_dir, seq, 'im' + str(j * self.gop_size + 1 + k).zfill(3) +'.png')
                    org_frames_path.append(input_path)

                intra_bits = self.get_intra_bits(bin_path)
                self.clip.append((org_frames_path, rec_frames_path, intra_bits))

    def get_intra_bits(self, bin_path):
        bits = os.path.getsize(bin_path) * 8
        return bits
    
    def __len__(self):
        return len(self.clip)
    
    def read_img(self, img_path):
        img = imageio.imread(img_path)
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        #[3, H, W]
        return img[:, :, : ]

    def __getitem__(self, index):
        '''
        ref_image = imageio.imread(self.ref[index]).transpose(2, 0, 1).astype(np.float32) / 255.0
        h = (ref_image.shape[1] // 64) * 64
        w = (ref_image.shape[2] // 64) * 64
        ref_image = np.array(ref_image[:, :h, :w])
        input_images = []
        refpsnr = None
        refmsssim = None
        for filename in self.input[index]:
            input_image = (imageio.imread(filename).transpose(2, 0, 1)[:, :h, :w]).astype(np.float32) / 255.0
            if refpsnr is None:
                refpsnr = CalcuPSNR(input_image, ref_image)
                refmsssim = ms_ssim(torch.from_numpy(input_image[np.newaxis, :]), torch.from_numpy(ref_image[np.newaxis, :]), data_range=1.0).numpy()
            else:
                input_images.append(input_image[:, :h, :w])

        input_images = np.array(input_images)
        return input_images, ref_image, self.refbpp[index], refpsnr, refmsssim
        '''
        index = index % len(self.clip)
        org_frames = [self.read_img(img_path) for img_path in self.clip[index][0]]
        rec_frames = [self.read_img(img_path) for img_path in self.clip[index][1]]
        org_frames = torch.stack(org_frames, 0)
        rec_frames = torch.stack(rec_frames, 0)
        h, w = rec_frames.shape[-2], rec_frames.shape[-1]
        intra_bpp = self.clip[index][2] / (h * w)
        return org_frames, rec_frames, intra_bpp

class UVGBPGDataSet(data.Dataset):
    def __init__(self, root_dir, rec_dir, test_class, qp):
        self.qp = qp
        self.test_class = test_class
        self.clip = []

        for i, seq in enumerate(classes_dict[test_class]["sequence_name"]):
            v_frames = classes_dict[test_class]["frameNum"][i]
            gop_size = classes_dict[test_class]["gop_size"]
            num = v_frames // gop_size
            print(seq, v_frames, gop_size)
            for j in range(num):
                rec_frames_path = [os.path.join(rec_dir, str(self.qp), seq, 'im' + str(j * gop_size + 1).zfill(3) +'.png')]
                org_frames_path = []

                for k in range(gop_size):
                    input_path = os.path.join(root_dir, seq, 'im' + str(j * gop_size + 1 + k).zfill(3) +'.png')
                    org_frames_path.append(input_path)

                bin_path = os.path.join(rec_dir, str(self.qp), seq, 'im' + str(j * self.gop_size + 1).zfill(3) +'.bin')
                intra_bits = self.get_intra_bits(bin_path)
                self.clip.append((org_frames_path, rec_frames_path, intra_bits))
    
    def __len__(self):
        return len(self.clip)
    
    def get_intra_bits(self, bin_path):
        bits = os.path.getsize(bin_path) * 8
        return bits

    def read_img(self, img_path):
        img = imageio.imread(img_path)
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        #[3, H, W]
        return img[:, :, : ]

    def __getitem__(self, index):
        index = index % len(self.clip)
        org_frames = [self.read_img(img_path) for img_path in self.clip[index][0]]
        rec_frames = [self.read_img(img_path) for img_path in self.clip[index][1]]
        org_frames = torch.stack(org_frames, 0)
        rec_frames = torch.stack(rec_frames, 0)
        intra_bpp = self.clip[index][2] / (org_frames.size(2) * org_frames.size(3)) 
        return org_frames, rec_frames, intra_bpp

class CTS(data.Dataset):
    def __init__(self, root_dir, test_class, return_intra_status, intra_model, rec_dir = None, qp = None):
        self.qp = qp
        self.test_class = test_class
        self.return_intra_status = return_intra_status
        self.clip = []

        for i, seq in enumerate(classes_dict[test_class]["sequence_name"]):
            v_frames = classes_dict[test_class]["frameNum"][i]
            gop_size = classes_dict[test_class]["gop_size"]
            num = v_frames // gop_size
            i_frame_path = []
            frame_path = []
            intra_bpp_list = []

            for j in range(v_frames):
                if j % gop_size == 0:
                    if return_intra_status:
                        if intra_model == 'vtm':
                            i_frame_path.append(os.path.join(rec_dir, str(self.qp), seq, 'im' + str(j + 1).zfill(3) + '.png'))
                        elif intra_model == 'x265':
                            i_frame_path.append(os.path.join(rec_dir, seq, str(self.qp), 'im' + str(j + 1).zfill(3) + '.png'))
                        elif intra_model == 'bpg':
                            i_frame_path.append(os.path.join(rec_dir, str(self.qp), seq, 'im' + str(j + 1).zfill(3) + '.png'))
                            bin_path = os.path.join(rec_dir, str(self.qp), seq, 'im' + str(j + 1).zfill(3) +'.bin')
                            intra_bits = self.get_intra_bits(bin_path)
                            w = int(classes_dict[test_class]["resolution"].split('x')[0])
                            h = int(classes_dict[test_class]["resolution"].split('x')[1])
                            intra_bpp = intra_bits / (w * h)
                            intra_bpp_list.append(intra_bpp)
                    else:
                        i_frame_path.append(os.path.join(root_dir, seq, 'im' + str(j + 1).zfill(3) + '.png'))

                    '''
                    bin_path = os.path.join(rec_dir, str(self.qp), seq, 'im' + str(j + 1).zfill(3) +'.bin')
                    intra_bits = self.get_intra_bits(bin_path)
                    w = int(classes_dict[test_class]["resolution"].split('x')[0])
                    h = int(classes_dict[test_class]["resolution"].split('x')[1])
                    intra_bpp = intra_bits / (w * h)
                    '''

                frame_path.append(os.path.join(root_dir, seq, 'im' + str(j + 1).zfill(3) + '.png'))

                '''
                if j % gop_size == gop_size - 1:
                    #intra_bpp = classes_dict[test_class]['cheng_bpp'][self.qp][i]
                    #intra_bpp = classes_dict[test_class]['nic_bpp'][self.qp][i]
                    #intra_bpp = 0
                    self.clip.append((i_frame_path, frame_path, intra_bpp, int(gop_size)))
                    i_frame_path = []
                    frame_path = []
                '''

            if return_intra_status:
                if intra_model == 'vtm':
                    intra_bpp = classes_dict[test_class]['vtm_bpp'][self.qp][i]
                elif intra_model == 'x265':
                    intra_bpp = classes_dict[test_class]['intra_bpp'][self.qp][i]
                elif intra_model == 'bpg':
                    intra_bpp = np.mean(intra_bpp_list)
            else:
                intra_bpp = 0
            self.clip.append((i_frame_path, frame_path, intra_bpp, int(gop_size)))
    
    def __len__(self):
        return len(self.clip)
    
    def get_intra_bits(self, bin_path):
        bits = os.path.getsize(bin_path) * 8
        return bits

    def read_img(self, img_path):
        img = imageio.imread(img_path)
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        #[3, H, W]
        return img[:, :, : ]

    def __getitem__(self, index):
        index = index % len(self.clip)
        i_frames = [self.read_img(img_path) for img_path in self.clip[index][0]]
        frames = [self.read_img(img_path) for img_path in self.clip[index][1]]
        i_frames = torch.stack(i_frames, 0)
        frames = torch.stack(frames, 0)
        intra_bpp = self.clip[index][2]
        gop_size = self.clip[index][3]
        return frames, intra_bpp, gop_size, i_frames

class UVG265DataSet(data.Dataset):
    def __init__(self, root_dir, rec_dir, test_class, qp):
        self.qp = qp
        self.test_class = test_class
        self.clip = []

        for i, seq in enumerate(classes_dict[test_class]["sequence_name"]):
            v_frames = classes_dict[test_class]["frameNum"][i]
            gop_size = classes_dict[test_class]["gop_size"]
            num = v_frames // gop_size
            print(seq, v_frames, gop_size)
            for j in range(num):
                rec_frames_path = [os.path.join(rec_dir, seq, str(self.qp), 'im' + str(j * gop_size + 1).zfill(3) +'.png')]
                org_frames_path = []

                for k in range(gop_size):
                    input_path = os.path.join(root_dir, seq, 'im' + str(j * gop_size + 1 + k).zfill(3) +'.png')
                    org_frames_path.append(input_path)

                intra_bpp = classes_dict[test_class]['intra_bpp'][self.qp][i]
                self.clip.append((org_frames_path, rec_frames_path, intra_bpp))
    
    def __len__(self):
        return len(self.clip)
    
    def read_img(self, img_path):
        img = imageio.imread(img_path)
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        #[3, H, W]
        return img[:, :, : ]

    def __getitem__(self, index):
        '''
        ref_image = imageio.imread(self.ref[index]).transpose(2, 0, 1).astype(np.float32) / 255.0
        h = (ref_image.shape[1] // 64) * 64
        w = (ref_image.shape[2] // 64) * 64
        ref_image = np.array(ref_image[:, :h, :w])
        input_images = []
        refpsnr = None
        refmsssim = None
        for filename in self.input[index]:
            input_image = (imageio.imread(filename).transpose(2, 0, 1)[:, :h, :w]).astype(np.float32) / 255.0
            if refpsnr is None:
                refpsnr = CalcuPSNR(input_image, ref_image)
                refmsssim = ms_ssim(torch.from_numpy(input_image[np.newaxis, :]), torch.from_numpy(ref_image[np.newaxis, :]), data_range=1.0).numpy()
            else:
                input_images.append(input_image[:, :h, :w])

        input_images = np.array(input_images)
        return input_images, ref_image, self.refbpp[index], refpsnr, refmsssim
        '''
        index = index % len(self.clip)
        org_frames = [self.read_img(img_path) for img_path in self.clip[index][0]]
        rec_frames = [self.read_img(img_path) for img_path in self.clip[index][1]]
        org_frames = torch.stack(org_frames, 0)
        rec_frames = torch.stack(rec_frames, 0)
        intra_bpp = self.clip[index][2]
        return org_frames, rec_frames, intra_bpp

class data_provider(data.Dataset):
    def __init__(self, rootdir = r"/backup1/klin/data/vimeo_septuplet/sequences", img_height=256, img_width=256):
        self.image_input_list, self.image_ref_list = self.get_vimeo(rootdir)
        self.img_height = img_height
        self.img_width = img_width
        print("The number of training samples: ", len(self.image_input_list))

    def get_vimeo(self, rootdir):
        #with open(filefolderlist) as f:
            #data = f.readlines()
        data = []
        for root, dirs, files in os.walk(rootdir):
            template = re.compile("im[1-9].png")
            data += [str(os.path.join(root, f)) for f in files if template.match(f) and int(f[-5:-4]) >= 2]
            
        fns_train_input = []
        fns_train_ref = []

        for n, line in enumerate(data, 1):
            y = os.path.join(rootdir, line.rstrip())
            fns_train_input += [y]

            curr_num = int(y[-5:-4])
            ref_frames = []
            for j in range(3, 4):
                ref_num = curr_num - (4 - j)
                assert ref_num >= 1
                ref_name = y[:-5] + str(ref_num) + '.png'
                ref_frames.append(ref_name)
            fns_train_ref += [ref_frames]

        return fns_train_input, fns_train_ref

    def __len__(self):
        return len(self.image_input_list)

    def read_img(self, img_path):
        img = imageio.imread(img_path)
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        #[3, H, W]
        return img

    def __getitem__(self, index):
        input_frame = [self.read_img(self.image_input_list[index])]
        ref_frames = [self.read_img(ref_img_path) for ref_img_path in self.image_ref_list[index]]

        rec_frames = torch.stack(ref_frames, 0)
        org_frames = torch.stack(input_frame, 0)
        #ref_frames.append(input_frame)

        rec_frames, org_frames = random_crop_frames(rec_frames, org_frames, [self.img_height, self.img_width])
        rec_frames, org_frames = random_flip_frames(rec_frames, org_frames)

        #input_image, ref_image = random_crop_and_pad_image_and_labels(input_image, ref_image, [self.img_height, self.img_width])
        #input_image, ref_image = random_flip(input_image, ref_image)

        #quant_noise_feature, quant_noise_z, quant_noise_mv = torch.nn.init.uniform_(torch.zeros_like(self.featurenoise), -0.5, 0.5), torch.nn.init.uniform_(torch.zeros_like(self.znoise), -0.5, 0.5), torch.nn.init.uniform_(torch.zeros_like(self.mvnois), -0.5, 0.5)
        #return input_image, ref_image, quant_noise_feature, quant_noise_z, quant_noise_mv
        #return frames, quant_noise_feature, quant_noise_z, quant_noise_mv
        return org_frames, rec_frames
        
class vimeo_provider(data.Dataset):
    def __init__(self, rootdir = r"/data/klin/vimeo_septuplet/sequences/", img_height=256, img_width=256, qp = 37):
        self.data_list = self.get_vimeo(rootdir)
        self.img_height = img_height
        self.img_width = img_width
        self.qp = qp
        print("The number of training samples: ", len(self.data_list))

    def get_vimeo(self, rootdir):
        data_list = []
        for root, dirs, files in os.walk(rootdir):
            template = re.compile("im1.png")
            data_list += [str(os.path.join(root, f)) for f in files if template.match(f)]
        return data_list           

    def __len__(self):
        return len(self.data_list)

    def read_img(self, img_path):
        img = imageio.imread(img_path)
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        #[3, H, W]
        return img

    def __getitem__(self, index):
        org_frames = []
        rec_frames = []

        first_frame_path = self.data_list[index]
        for i in range(1, 8):
            org_frames.append(self.read_img(first_frame_path.replace('im1', 'im' + str(i))))

        #rec_frames.append(self.read_img(first_frame_path.replace('im1', 'im1_bpg444_QP{}'.format(self.qp))))
        for i in range(1, 2):
        #for i in range(1, 3):
        #for i in range(1, 5):
            #rec_frames.append(self.read_img(first_frame_path.replace('im1', 'im' + str(i) + "_3")))
            rec_frames.append(self.read_img(first_frame_path.replace('im1', 'im1')))

        org_frames = torch.stack(org_frames, 0)
        rec_frames = torch.stack(rec_frames, 0)
   
        rec_frames, org_frames = random_crop_frames(rec_frames, org_frames, [self.img_height, self.img_width])
        rec_frames, org_frames = random_flip_frames(rec_frames, org_frames)

        return org_frames, rec_frames