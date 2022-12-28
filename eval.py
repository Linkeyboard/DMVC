import os
import argparse
import torch
import random
import math
import datetime
import numpy as np
from models.Network import *
from torch.utils.data import DataLoader
from dataload.dataset import CTS
from pytorch_msssim import MS_SSIM
from compressai.zoo import cheng2020_anchor, cheng2020_attn, mbt2018, mbt2018_mean
from tqdm import tqdm

metric_list = ['mse', 'ms-ssim']
parser = argparse.ArgumentParser(description='DMVC evaluation')

# evaluating parameters
intra_models_zoo = [
    'cheng2020_anchor',
    'cheng2020_attn',
    'mbt2018',
    'mbt2018_mean',
    'vtm',
    'x265',
    'bpg',
]

test_class_list = [
    'ClassB',
    'ClassC',
    'ClassD',
    'ClassE',
    'ClassF',
    'UVG',
    'MCLJCV',
]

parser.add_argument('--pretrain', default = '', help='Load pretrain model')
parser.add_argument('--img_dir', default = '')
parser.add_argument('--eval_lambda', default = 256, type = int, help = '[256, 512, 1024, 2048] for MSE, [8, 16, 32, 64] for MS-SSIM')
parser.add_argument('--metric', default = 'mse', choices = metric_list, help = 'mse or ms-ssim')
parser.add_argument('--intra_model', choices = intra_models_zoo, help = 'The intra coding method')
parser.add_argument('--test_class', default = 'ClassD', type = str, choices = test_class_list, help = 'Choose from the test dataset')
parser.add_argument('--gop_size', default = '0', type = int, help = 'The length of the gop')

args = parser.parse_args()

if args.metric == "mse":
    # cheng2020_anchor
    if args.intra_model == 'cheng2020_anchor':
        lambda_to_qp_dict = {2048: 6, 1024: 5, 512: 4, 256: 3, 128: 2}
    elif args.intra_model == 'cheng2020_attn':
        lambda_to_qp_dict = {2048: 6, 1024: 5, 512: 4, 256: 3, 128: 2}
    elif args.intra_model == 'mbt2018':
        lambda_to_qp_dict = {2048: 6, 1024: 5, 512: 4, 256: 3, 128: 2}
    elif args.intra_model == 'mbt2018_mean':
        lambda_to_qp_dict = {2048: 6, 1024: 5, 512: 4, 256: 3, 128: 2}
    elif args.intra_model == 'vtm':
        lambda_to_qp_dict = {2048: 25, 1024: 27, 512: 31, 256: 33}
    elif args.intra_model == 'x265':
        lambda_to_qp_dict = {2048: 20, 1024: 23, 512: 26, 256: 29}
    elif args.intra_model == 'bpg':
        lambda_to_qp_dict = {2048: 24, 1024: 28, 512: 32, 256: 36}
else:
    if args.intra_model == 'cheng2020_anchor':
        lambda_to_qp_dict = {64: 6, 32: 5, 16: 4, 8: 3, 4: 2}
    elif args.intra_model == 'mbt2018':
        lambda_to_qp_dict = {64: 6, 32: 5, 16: 4, 8: 3, 4: 2}
    elif args.intra_model == 'mbt2018_mean':
        lambda_to_qp_dict = {64: 6, 32: 5, 16: 4, 8: 3, 4: 2}
    elif args.intra_model == 'vtm':
        lambda_to_qp_dict = {64: 25, 32: 27, 16: 31, 8: 33}
    elif args.intra_model == 'x265':
        lambda_to_qp_dict = {64: 20, 32: 23, 16: 26, 8: 29}
    elif args.intra_model == 'bpg':
        lambda_to_qp_dict = {64: 24, 32: 28, 16: 32, 8: 36}

if args.intra_model == 'vtm':
    images_folder = 'images_intra'
elif args.intra_model == 'x265':
    images_folder = 'h265'
elif args.intra_model == 'bpg':
    images_folder = 'bpg'
else:
    images_folder = 'images_intra'

return_intra_status = True if args.intra_model == 'vtm' or args.intra_model == 'x265' or args.intra_model == 'bpg' else False
test_dataset = CTS(args.img_dir, args.test_class, return_intra_status, args.intra_model, None, lambda_to_qp_dict[args.eval_lambda])
test_loader = DataLoader(dataset = test_dataset, shuffle = False, num_workers = 1, batch_size = 1)

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def eval_model(net):
    print("Evaluating...")
    net.eval()

    sum_psnr = 0
    sum_bpp = 0
    sum_intra_bpp = 0
    sum_inter_bpp = 0
    sum_intra_psnr = 0
    sum_inter_psnr = 0
    sum_ms_ssim = 0
    t0 = datetime.datetime.now()
    cnt = 0

    if args.intra_model == 'cheng2020_anchor':
        intra_model = cheng2020_anchor(quality = lambda_to_qp_dict[args.eval_lambda], pretrained = True, metric = args.metric)
    elif args.intra_model == 'cheng2020_attn':
        intra_model = cheng2020_attn(quality = lambda_to_qp_dict[args.eval_lambda], pretrained = True, metric = args.metric)
    elif args.intra_model == 'mbt2018':
        intra_model = mbt2018(quality = lambda_to_qp_dict[args.eval_lambda], pretrained = True, metric = args.metric)
    elif args.intra_model == 'mbt2018_mean':
        intra_model = mbt2018_mean(quality = lambda_to_qp_dict[args.eval_lambda], pretrained = True, metric = args.metric)

    if not return_intra_status:
        intra_model.cuda()
        intra_model.eval()

    # intra bits, motion bits, resi bits
    sum_bpp_i = 0
    sum_bpp_m = 0
    sum_bpp_r = 0

    for batch_idx, (frames, intra_bpp, gop_size, i_frames) in enumerate(test_loader):
        batch_size, frame_length, _, h, w = frames.shape
        if args.gop_size > 0:
            gop_size = args.gop_size
        else:
            gop_size = gop_size.item()
        #frames = frames.cuda()
        #i_frames = i_frames.cuda()

        rec_frames = []
        ms_ssim_module = MS_SSIM(data_range = 1, size_average= True, channel = 3)

        last_state = None
        for frame_idx in tqdm(range(frame_length)):
            with torch.no_grad():
                if frame_idx % gop_size == 0:
                    if frame_idx:
                        frames = frames[:, gop_size : ]
                    
                    if not return_intra_status:
                        result = intra_model(frames[:, frame_idx % gop_size].cuda())
                        intra_bits = sum((torch.log(likelihoods).sum() / (-math.log(2))) for likelihoods in result["likelihoods"].values())
                        x_hat = result["x_hat"].clamp(0., 1.)
                        intra_mse = torch.mean((x_hat - frames[:, frame_idx % gop_size].cuda()).pow(2))
                        intra_psnr = torch.mean(10 * (torch.log(1. / intra_mse) / np.log(10))).cpu().detach().numpy()
                        intra_ms_ssim = ms_ssim_module(x_hat, frames[:, frame_idx % gop_size].cuda())
                        sum_bpp += intra_bits.detach().cpu().numpy().item() / (batch_size * h * w)
                        sum_intra_bpp += intra_bits.detach().cpu().numpy().item() / (batch_size * h * w)
                        sum_bpp_i += intra_bits.detach().cpu().numpy().item() / (batch_size * h * w)
                    else:
                        intra_id = frame_idx // gop_size
                        intra_mse = torch.mean((i_frames[:, intra_id].cuda() - frames[:, frame_idx % gop_size].cuda()).pow(2))
                        intra_psnr = torch.mean(10 * (torch.log(1. / intra_mse) / np.log(10))).cpu().detach().numpy()
                        intra_ms_ssim = ms_ssim_module(i_frames[:, intra_id].cuda(), frames[:, frame_idx % gop_size].cuda())
                        sum_bpp += intra_bpp.detach().numpy().item()
                        sum_intra_bpp += intra_bpp.detach().numpy().item()
                        sum_bpp_i += intra_bpp.detach().cpu().numpy().item() / (batch_size * h * w)

                    sum_psnr += intra_psnr
                    sum_intra_psnr += intra_psnr
                    sum_ms_ssim += intra_ms_ssim
                    cnt += 1
                    #print('[{}] recon_psnr:{:.2f} bpp:{:.6f}'.format(frame_idx, intra_psnr, sum_intra_bpp))

                    if frame_idx == 0:
                        if not return_intra_status:
                            rec_frames = x_hat.unsqueeze(1)
                        else:
                            rec_frames = i_frames[:, intra_id].unsqueeze(1).cuda()
                    else:
                        if not return_intra_status:
                            rec_frames = torch.cat([rec_frames, x_hat.unsqueeze(1)], 1)
                        else:
                            rec_frames = torch.cat([rec_frames, i_frames[:, intra_id].unsqueeze(1).cuda()], 1)

                    continue

                x_curr = frames[:, frame_idx % gop_size].cuda()

                '''
                if last_state is None:
                    x_hat, last_state, recon_loss, warp_next_loss, pred_next_loss, pred_loss, bpp, bpp_y, bpp_z, bpp_h, bpp_hp = net(x_curr, rec_frames[:, :], return_state = True)
                else:
                    x_hat, last_state, recon_loss, warp_next_loss, pred_next_loss, pred_loss, bpp, bpp_y, bpp_z, bpp_h, bpp_hp = net(x_curr, rec_frames[:, -1:], last_state, return_state = True)
                '''
                x_hat, last_state, recon_loss, warp_next_loss, pred_next_loss, pred_loss, bpp, bpp_y, bpp_z, bpp_h, bpp_hp = net(x_curr, rec_frames, return_state = True)
                '''
                x_hat_write = (x_hat.squeeze().clamp(0., 1.).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                imageio.imwrite("reco_{}.png".format(frame_idx), x_hat_write)
                '''

                pred_psnr = 10 * (torch.log(1 * 1 / pred_loss) / np.log(10)).cpu().detach().numpy()
                pred_next_psnr = 10 * (torch.log(1 * 1 / pred_next_loss) / np.log(10)).cpu().detach().numpy()
                recon_psnr = 10 * (torch.log(1 * 1 / recon_loss) / np.log(10)).cpu().detach().numpy()
                warp_psnr = 10 * (torch.log(1 * 1 / warp_next_loss) / np.log(10)).cpu().detach().numpy()
                #print("[{}] recon_psnr:{:.8f} pred_next_psnr:{:.8f} warp_psnr:{:.8f} bpp:{:.8f} bpp_y:{:.8f} bpp_z:{:.8f} bpp_h:{:.8f} bpp_hp:{:.8f} bpp_m:{:.8f} bpp_r:{:.8f}".format(frame_idx, 
                #recon_psnr, pred_next_psnr, warp_psnr, bpp, bpp_y, bpp_z, bpp_h, bpp_hp, bpp_h + bpp_hp, bpp_y + bpp_z))

                sum_bpp_m += (bpp_h + bpp_hp)
                sum_bpp_r += (bpp_y + bpp_z)

                rec_frames = torch.cat([rec_frames, x_hat.unsqueeze(1).clamp(0., 1.).detach()], 1)
                inter_ms_ssim = ms_ssim_module(x_hat, x_curr)

                if rec_frames.size(1) > 3:
                    rec_frames = rec_frames[:, -3 :]

                cnt += 1
                sum_psnr += recon_psnr
                sum_bpp += bpp.cpu().detach().numpy()

                sum_inter_bpp += bpp.cpu().detach().numpy()
                sum_inter_psnr += recon_psnr
                sum_ms_ssim += inter_ms_ssim
        
        intra_frame_length = frame_length // gop_size
        inter_frame_length = frame_length - intra_frame_length
        #print('intra_bpp: {:.6f} inter_bpp:{:.6f} intra_psnr:{:.4f} inter_psnr:{:.4f}'.format(sum_intra_bpp / intra_frame_length, sum_inter_bpp/ inter_frame_length, 
                #sum_intra_psnr / intra_frame_length, sum_inter_psnr / inter_frame_length))
        
        #exit(0)
        sum_intra_psnr = 0
        sum_inter_psnr = 0
        sum_intra_bpp = 0
        sum_inter_bpp = 0

    t1 = datetime.datetime.now()
    deltatime = t1 - t0
    #print(deltatime, t0, t1)
    print("recon_psnr:{:.4f} ms_ssim:{:.6f} bpp:{:.6f} time:{:.4f}".format(sum_psnr / cnt, sum_ms_ssim / cnt, sum_bpp / cnt, (deltatime.seconds + 1e-6 * deltatime.microseconds) / cnt))
    sum_tmp = sum_bpp_i + sum_bpp_m + sum_bpp_r
    #print("bpp_i:{:.2f} bpp_m:{:.6f} hpp_r:{:.6f} ".format(sum_bpp_i / sum_tmp, sum_bpp_m / sum_tmp, sum_bpp_r / sum_tmp))

def check_dir_exist(check_dir):
    if not os.path.exists(check_dir):
        os.makedirs(check_dir)

def main():
    print(args)

    model = DMVC()
    model.cuda()

    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print('The total number of the learnable parameters:', num_params)

    if args.pretrain != '':
        print('Load the model from {}'.format(args.pretrain))
        pretrained_dict = torch.load(args.pretrain)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        '''
        for k, v in pretrained_dict.items():
            print(k)
        '''

    eval_model(model)

if __name__ == "__main__":
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    random.seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True

    main()
