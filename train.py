import os
import argparse
import torch
import numpy as np
from models.Network import *
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader
import sys
import math
from dataload.dataset import vimeo_provider, CTS
from tensorboardX import SummaryWriter
torch.backends.cudnn.enabled = True
import datetime
from pytorch_msssim import MS_SSIM
from compressai.zoo import cheng2020_anchor, cheng2020_attn, mbt2018, mbt2018_mean
import random

metric_list = ['mse', 'ms-ssim']
parser = argparse.ArgumentParser(description='DMVC')
# add config parameters
parser.add_argument('-l', '--log', default='', help='output training details')
parser.add_argument('-p', '--pretrain', default = '', help='load pretrain model')

# training parameters
parser.add_argument('--root_dir', default = r'/home/hdda/klin/data/vimeo_septuplet', type = str)
parser.add_argument('--plot_dir', default = './plots/default', type = str)
parser.add_argument('--checkpoint_dir', default = './checkpoints/default', type = str)
parser.add_argument('--train_lambda', default = 256, type = int, help = '[256, 512, 1024, 2048] for MSE, [8, 16, 32, 64] for MS-SSIM')
parser.add_argument('--metric', default = 'mse', choices = metric_list, help = 'mse or ms-ssim')
parser.add_argument('--lr', default = 1e-4, type = float)
parser.add_argument('--lr_decay', default = 0.1, type = float)
parser.add_argument('--max_epoch', default = 1000, type = int)
parser.add_argument('--start_epoch', default = 0, type = int)
parser.add_argument('--max_step', default = 30000000, type = int)
parser.add_argument('--batch_size', default = 4, type = int)
parser.add_argument('--decay_step', default = 20000000, type = int)
parser.add_argument('--print_step', default = 600, type = int)

args = parser.parse_args()
# global parameters
global_step = 0
cur_lr = 1e-4

if args.metric == "mse":
    lambda_to_qp_dict = {64: 22, 32: 27, 16: 32, 8: 37}
    # cheng2020_anchor
    #lambda_to_qp_dict = {2048: 6, 1024: 5, 512: 4, 256: 3, 128: 2}
    # vtm15.2
    #lambda_to_qp_dict = {2048: 25, 1024: 27, 512: 31, 256: 33}
    # x265 medium
    #lambda_to_qp_dict = {2048: 20, 1024: 23, 512: 26, 256: 29}
    # bpg
    #lambda_to_qp_dict = {2048: 24, 1024: 28, 512: 32, 256: 36}
else:
    lambda_to_qp_dict = {64: 22, 32: 27, 16: 32, 8: 37}

train_dataset = vimeo_provider(args.root_dir, qp = lambda_to_qp_dict[args.train_lambda])
train_loader = DataLoader(dataset = train_dataset, shuffle = True, num_workers = 6, batch_size = args.batch_size, pin_memory = True)

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def adjust_learning_rate(optimizer):
    global cur_lr
    if global_step < args.decay_step:
        lr = args.lr
    else:
        lr = args.lr * (args.lr_decay ** (global_step // args.decay_step))

    for param_group in optimizer.param_groups:
        if global_step >= args.decay_step:
            param_group['lr'] = param_group['lr'] * (args.lr_decay ** (global_step // args.decay_step))

    cur_lr = lr

def save_model(model, save_dir, iter):
    torch.save(model.state_dict(), os.path.join(save_dir, "iter{}.model".format(iter)))

def train_one_epoch(net, optimizer, epoch):
    net.train()
    global global_step
    cal_cnt = 0 
    sum_ms_ssim = 0
    sum_psnr = 0
    sum_next_psnr = 0
    sum_warp_psnr = 0
    sum_pred_psnr = 0
    sum_bpp = 0
    sum_bpp_h = 0
    sum_bpp_hp = 0
    sum_bpp_y = 0
    sum_bpp_z = 0
    sum_rd_loss = 0
    tb_logger = SummaryWriter(args.plot_dir)
    t0 = datetime.datetime.now()

    for batch_idx, (org_frames, rec_frames) in enumerate(train_loader):
        batch_size, frame_length, _, h, w = org_frames.shape
        org_frames, rec_frames = org_frames.cuda(), rec_frames.cuda()

        for frame_idx in range(1, frame_length):
            with torch.no_grad():
                x_cur = org_frames[:, frame_idx]
                ref_list = rec_frames

            x_hat, recon_loss, warp_next_loss, pred_next_loss, pred_loss, bpp, bpp_y, bpp_z, bpp_h, bpp_hp = net(x_cur, ref_list)
            pred_psnr = 10 * (torch.log(1 * 1 / pred_loss) / np.log(10)).cpu().detach().numpy()
            pred_next_psnr = 10 * (torch.log(1 * 1 / pred_next_loss) / np.log(10)).cpu().detach().numpy()
            recon_psnr = 10 * (torch.log(1 * 1 / recon_loss) / np.log(10)).cpu().detach().numpy()
            warp_psnr = 10 * (torch.log(1 * 1 / warp_next_loss) / np.log(10)).cpu().detach().numpy()

            ms_ssim_module = MS_SSIM(data_range = 1, size_average= True, channel = 3)
            ms_ssim_loss = 1 - ms_ssim_module(x_hat, x_cur)

            if args.metric == 'mse':
                rd_loss = args.train_lambda * recon_loss + bpp
            else:
                rd_loss = args.train_lambda * ms_ssim_loss + bpp

            optimizer.zero_grad()
            rd_loss.backward()
            torch.nn.utils.clip_grad_norm(net.parameters(), 10, norm_type=2)
            optimizer.step()

            with torch.no_grad():
                rec_frames = torch.cat([rec_frames, x_hat.unsqueeze(1).clamp(0., 1.).detach()], 1)
                if rec_frames.size(1) > 4:
                    rec_frames = rec_frames[:, -4 : ]

            global_step = global_step + 1
            cal_cnt += 1
            sum_psnr += recon_psnr
            sum_next_psnr += pred_next_psnr
            sum_warp_psnr += warp_psnr
            sum_pred_psnr += pred_psnr
            sum_bpp += bpp.cpu().detach()
            sum_bpp_h += bpp_h.cpu().detach()
            sum_bpp_hp += bpp_hp.cpu().detach()
            sum_bpp_y += bpp_y.cpu().detach()
            sum_bpp_z += bpp_z.cpu().detach()
            sum_rd_loss += rd_loss.cpu().detach()
            sum_ms_ssim += (1 - ms_ssim_loss).cpu().detach()

            if global_step % args.print_step == (args.print_step - 1):
                tb_logger.add_scalar('recon_psnr', sum_psnr / cal_cnt, global_step)
                tb_logger.add_scalar('ms_ssim', sum_ms_ssim / cal_cnt, global_step)
                tb_logger.add_scalar('warp_psnr', sum_warp_psnr / cal_cnt, global_step)
                tb_logger.add_scalar('pred_next_psnr', sum_next_psnr / cal_cnt, global_step)
                tb_logger.add_scalar('pred_psnr', sum_pred_psnr / cal_cnt, global_step)
                tb_logger.add_scalar('sum_bpp', sum_bpp / cal_cnt, global_step)
                tb_logger.add_scalar('sum_bpp_h', sum_bpp_h / cal_cnt, global_step)
                tb_logger.add_scalar('sum_bpp_hp', sum_bpp_hp / cal_cnt, global_step)
                tb_logger.add_scalar('sum_bpp_y', sum_bpp_y / cal_cnt, global_step)
                tb_logger.add_scalar('sum_bpp_z', sum_bpp_z / cal_cnt, global_step)
                tb_logger.add_scalar('sum_rd_loss', sum_rd_loss / cal_cnt, global_step)
                t1 = datetime.datetime.now()
                deltatime = t1 - t0
                log = 'Train Epoch : {:02} [{:4}/{:4} ({:3.0f}%)] ms_ssim:{:.6f} psnr:{:.2f} next_psnr:{:.2f} warp_psnr:{:.2f} pred_psnr:{:.2f} bpp:{:.6f} bpp_y:{:.6f} bpp_z:{:.6f} bpp_h:{:.6f} bpp_hp:{:.6f} loss:{:.6f} lr:{} {}'.format(epoch, batch_idx, 
                    len(train_loader), 100. * batch_idx / len(train_loader), sum_ms_ssim / cal_cnt, sum_psnr / cal_cnt, sum_next_psnr / cal_cnt, sum_warp_psnr / cal_cnt, sum_pred_psnr / cal_cnt, sum_bpp / cal_cnt, sum_bpp_y / cal_cnt, sum_bpp_z / cal_cnt, sum_bpp_h / cal_cnt, sum_bpp_hp / cal_cnt, sum_rd_loss / cal_cnt, cur_lr, (deltatime.seconds + 1e-6 * deltatime.microseconds) / cal_cnt)
                print(log)

                cal_cnt = 0
                sum_psnr = 0
                sum_ms_ssim = 0
                sum_warp_psnr = 0
                sum_next_psnr = 0
                sum_pred_psnr = 0
                sum_bpp_h = 0
                sum_bpp_hp = 0
                sum_rd_loss = 0
                sum_bpp = 0
                sum_bpp_y = 0
                sum_bpp_z = 0
                t0 = t1

            if global_step % (args.print_step * 10) == 0:
                save_model(net, args.checkpoint_dir, global_step)

def check_dir_exist(check_dir):
    if not os.path.exists(check_dir):
        os.makedirs(check_dir)

def main():
    print(args)
    global global_step

    check_dir_exist(args.plot_dir)
    check_dir_exist(args.checkpoint_dir)

    model = DMVC()

    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print('Total number of the parameters:', num_params)

    if args.pretrain != '':
        print('load the whole module from {}'.format(args.pretrain))
        pretrained_dict = torch.load(args.pretrain)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        '''
        for k, v in pretrained_dict.items():
            print(k)
        '''

        f = args.pretrain
        if f.find('iter') != -1 and f.find('.model') != -1:
            st = f.find('iter') + 4
            ed = f.find('.model', st)
            global_step = int(f[st:ed])
            print('Global Step Start from ', global_step)

    if torch.cuda.device_count() > 1:
        net = CustomDataParallel(model)
    else:
        net = model.cuda()
    params = [
        {"params": net.feature_agg.parameters(), "lr": args.lr},
    ]
    #optimizer = optim.Adam(params)
    optimizer = optim.Adam(net.parameters(), args.lr)

    for epoch in range(args.start_epoch, args.max_epoch):
        adjust_learning_rate(optimizer)
        if global_step > args.max_step:
            save_model(model, args.checkpoint_dir, global_step)
            break

        train_one_epoch(net, optimizer, epoch)
        save_model(model, args.checkpoint_dir, global_step)


if __name__ == "__main__":
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    random.seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    main()