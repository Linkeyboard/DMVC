import torch
import torch.nn as nn
from torch.nn import functional as F

import numpy as np

from .FullFactorizedModel import FullFactorizedModel
from .ConditionalGaussianModel import ConditionalGaussianModel
from .MaskedConv2d import MaskedConv2d
from .quantize import quantize
from .ResUnit import EncResUnit, DecResUnit
from .OpticalFlow import ME_Spynet, flow_warp 
from .convlstm import ConvLSTM
#from .flowlib import flow_to_image
#from utils.preprocess import reshape_patch, reshape_patch_back
import imageio
from torch import Tensor

#from pytorch_msssim import ms_ssim, MS_SSIM
#def crop(x: Tensor, padding: Tuple[int, ...]) -> Tensor:
def crop(x, padding):
    return F.pad(x, tuple(-p for p in padding))

class HiddenEncoder(nn.Module):
    def __init__(self, in_channels, latent_channels, out_channels):
        super().__init__()
        self._nic = in_channels
        self._nlc = latent_channels
        self._noc = out_channels

        self._model = nn.Sequential(
            nn.Conv2d(self._nic, self._nlc, 4, 2, 1),
            EncResUnit(self._nlc, self._nlc, 1),
            EncResUnit(self._nlc, self._nlc, 1),
            EncResUnit(self._nlc, self._nlc, 1),
            EncResUnit(self._nlc, self._nlc, 1),
            nn.Conv2d(self._nlc, self._nlc, 4, 2, 1),
            EncResUnit(self._nlc, self._nlc, 1),
            EncResUnit(self._nlc, self._nlc, 1),
            EncResUnit(self._nlc, self._nlc, 1),
            EncResUnit(self._nlc, self._nlc, 1),
            nn.Conv2d(self._nlc, self._nlc, 4, 2, 1),
            EncResUnit(self._nlc, self._nlc, 1),
            EncResUnit(self._nlc, self._nlc, 1),
            EncResUnit(self._nlc, self._nlc, 1),
            EncResUnit(self._nlc, self._nlc, 1),
            nn.Conv2d(self._nlc, self._noc, 4, 2, 1),
        )

    def forward(self, inputs):
        return self._model(inputs)


class HiddenDecoder(nn.Module):
    def __init__(self, in_channels, latent_channels, out_channels):
        super().__init__()
        self._nic = in_channels
        self._nlc = latent_channels
        self._noc = out_channels

        self._model = nn.Sequential(
            nn.ConvTranspose2d(self._nic, self._nlc, 4, 2, 1),
            DecResUnit(self._nlc, self._nlc, 1),
            DecResUnit(self._nlc, self._nlc, 1),
            DecResUnit(self._nlc, self._nlc, 1),
            DecResUnit(self._nlc, self._nlc, 1),
            nn.ConvTranspose2d(self._nlc, self._nlc, 4, 2, 1),
            DecResUnit(self._nlc, self._nlc, 1),
            DecResUnit(self._nlc, self._nlc, 1),
            DecResUnit(self._nlc, self._nlc, 1),
            DecResUnit(self._nlc, self._nlc, 1),
            #nn.ConvTranspose2d(self._nlc, self._nlc, 4, 2, 1),
            nn.Conv2d(self._nlc, self._nlc, 3, 1, 1),
            DecResUnit(self._nlc, self._nlc, 1),
            DecResUnit(self._nlc, self._nlc, 1),
            DecResUnit(self._nlc, self._nlc, 1),
            DecResUnit(self._nlc, self._nlc, 1),
            nn.Conv2d(self._nlc, self._noc, 3, 1, 1),
            #nn.ConvTranspose2d(self._nlc, self._noc, 4, 2, 1),
        )

    def forward(self, inputs):
        return self._model(inputs)

class Encoder(nn.Module):
    def __init__(self, in_channels, latent_channels, out_channels):
        super().__init__()
        self._nic = in_channels
        self._nlc = latent_channels
        self._noc = out_channels

        self._model = nn.Sequential(
            nn.Conv2d(self._nic, self._nlc, 4, 2, 1),
            EncResUnit(self._nlc, self._nlc, 1),
            EncResUnit(self._nlc, self._nlc, 1),
            EncResUnit(self._nlc, self._nlc, 1),
            EncResUnit(self._nlc, self._nlc, 1),
            nn.Conv2d(self._nlc, self._nlc, 4, 2, 1),
            EncResUnit(self._nlc, self._nlc, 1),
            EncResUnit(self._nlc, self._nlc, 1),
            EncResUnit(self._nlc, self._nlc, 1),
            EncResUnit(self._nlc, self._nlc, 1),
            nn.Conv2d(self._nlc, self._nlc, 4, 2, 1),
            EncResUnit(self._nlc, self._nlc, 1),
            EncResUnit(self._nlc, self._nlc, 1),
            EncResUnit(self._nlc, self._nlc, 1),
            EncResUnit(self._nlc, self._nlc, 1),
            nn.Conv2d(self._nlc, self._noc, 4, 2, 1),
        )

    def forward(self, inputs):
        return self._model(inputs)


class Decoder(nn.Module):
    def __init__(self, in_channels, latent_channels, out_channels):
        super().__init__()
        self._nic = in_channels
        self._nlc = latent_channels
        self._noc = out_channels

        self._model = nn.Sequential(
            nn.ConvTranspose2d(self._nic, self._nlc, 4, 2, 1),
            DecResUnit(self._nlc, self._nlc, 1),
            DecResUnit(self._nlc, self._nlc, 1),
            DecResUnit(self._nlc, self._nlc, 1),
            DecResUnit(self._nlc, self._nlc, 1),
            nn.ConvTranspose2d(self._nlc, self._nlc, 4, 2, 1),
            DecResUnit(self._nlc, self._nlc, 1),
            DecResUnit(self._nlc, self._nlc, 1),
            DecResUnit(self._nlc, self._nlc, 1),
            DecResUnit(self._nlc, self._nlc, 1),
            nn.ConvTranspose2d(self._nlc, self._nlc, 4, 2, 1),
            DecResUnit(self._nlc, self._nlc, 1),
            DecResUnit(self._nlc, self._nlc, 1),
            DecResUnit(self._nlc, self._nlc, 1),
            DecResUnit(self._nlc, self._nlc, 1),
            nn.ConvTranspose2d(self._nlc, self._noc, 4, 2, 1),
        )

    def forward(self, inputs):
        return self._model(inputs)


class HyperEncoder(nn.Module):
    def __init__(self, in_channels, latent_channels, out_channels):
        super().__init__()
        self._nic = in_channels
        self._nlc = latent_channels
        self._noc = out_channels

        self._hyper_encoder = nn.Sequential(
            nn.Conv2d(self._nic, self._nic, 1, 1),
            nn.PReLU(self._nic),
            nn.Conv2d(self._nic, self._nlc, 2, 2),
            nn.PReLU(self._nlc),
            nn.Conv2d(self._nlc, self._noc, 2, 2),
            nn.PReLU(self._noc),
            nn.Conv2d(self._noc, self._noc, 1, 1),
        )

    def forward(self, inputs):
        return self._hyper_encoder(inputs)


class HyperDecoder(nn.Module):
    def __init__(self, in_channels, latent_channels, out_channels):
        super(HyperDecoder, self).__init__()
        self._nic = in_channels
        self._nlc = latent_channels
        self._noc = out_channels

        self._hyper_decoder = nn.Sequential(
            nn.ConvTranspose2d(self._nic, self._nic, 1, 1),
            nn.PReLU(self._nic),
            nn.ConvTranspose2d(self._nic, self._nlc, 2, 2),
            nn.PReLU(self._nlc),
            nn.ConvTranspose2d(self._nlc, self._noc, 2, 2),
            nn.PReLU(self._noc),
            nn.ConvTranspose2d(self._noc, self._noc, 1, 1),
        )

    def forward(self, inputs):
        return self._hyper_decoder(inputs)


class EntropyParameters(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EntropyParameters, self).__init__()
        self._ncs = [int(item) for item in np.linspace(in_channels, out_channels, 4)]

        self._entropy_parameters = nn.Sequential(
            nn.Conv2d(self._ncs[0], self._ncs[1], 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self._ncs[1], self._ncs[2], 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self._ncs[2], self._ncs[3], 1)
        )

    def forward(self, inputs):
        return self._entropy_parameters(inputs)

class FeatureAggregation(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FeatureAggregation, self).__init__()
        self._ncs = [int(item) for item in np.linspace(in_channels, out_channels, 4)]

        self.combine = nn.Sequential(
            nn.Conv2d(self._ncs[0], self._ncs[1], 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self._ncs[1], self._ncs[2], 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self._ncs[2], self._ncs[3], 1)
        )

    def forward(self, inputs):
        return self.combine(inputs)

class FeatureExtraction(nn.Module):
    def __init__(self, n_c):
        super().__init__()
        self.n_c = n_c
        self.conv1 = nn.Conv2d(3, self.n_c, 4, 2, 1)
        self.conv2 = nn.Sequential(
            EncResUnit(self.n_c, self.n_c, 1),
            EncResUnit(self.n_c, self.n_c, 1),
            nn.Conv2d(self.n_c, self.n_c, 4, 2, 1),
        )

    def forward(self, x):
        f1 = self.conv1(x)
        f2 = self.conv2(f1)
        return [f1, f2]

class FeatureRestore(nn.Module):
    def __init__(self, n_c):
        super().__init__()
        self.n_c = n_c
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(self.n_c * 2, self.n_c, 4, 2, 1),
            DecResUnit(self.n_c, self.n_c, 1),
            DecResUnit(self.n_c, self.n_c, 1),
        )
        self.deconv2 = nn.ConvTranspose2d(self.n_c * 2, 3, 4, 2, 1)
    
    def forward(self, x, f1, f2):
        out = self.deconv1(torch.cat([x, f2], 1))
        out = self.deconv2(torch.cat([out, f1], 1))
        return out

class Refine_Net(nn.Module):
    def __init__(self, in_channels, latent_channels, out_channels):
        super().__init__()
        self._nic = in_channels
        self._nlc = latent_channels
        self._noc = out_channels

        self._model = nn.Sequential(
            nn.Conv2d(self._nic, self._nlc, 3, 1, 1),
            nn.PReLU(),
            EncResUnit(self._nlc, self._nlc, 1),
            EncResUnit(self._nlc, self._nlc, 1),
            EncResUnit(self._nlc, self._nlc, 1),
            EncResUnit(self._nlc, self._nlc, 1),
            nn.Conv2d(self._nlc, self._nlc, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(self._nlc, self._noc, 1),
        )

    def forward(self, inputs):
        return self._model(inputs)

class DMVC(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_c = 128
        self.num_layer = 4
        self.hidden_dim = [self.n_c // 2 for i in range(self.num_layer)]
        self.conv_lstm = ConvLSTM(self.n_c // 2, self.hidden_dim, (5, 5), self.num_layer, True, True, True) 
        self.warp_lstm = ConvLSTM(self.n_c // 2, [self.n_c // 2], (5, 5), 1, True, True, True) 
        self.mv_encoder = HiddenEncoder(6, self.n_c, self.n_c)
        self.mv_decoder = HiddenDecoder(self.n_c, self.n_c, self.n_c)
        self.mv_hyper_encoder = HyperEncoder(self.n_c, self.n_c, self.n_c)
        self.mv_hyper_decoder = HyperDecoder(self.n_c, self.n_c, self.n_c*2)
        self.factorized_mv = FullFactorizedModel(self.n_c, (3, 3, 3), 1e-9)

        self.feature_embedding = FeatureExtraction(self.n_c // 2)
        self.feature_restore = FeatureRestore(self.n_c // 2)

        self.encoder = Encoder(3, self.n_c, self.n_c)
        self.decoder = Decoder(self.n_c, self.n_c, 3)
        self.hyper_encoder = HyperEncoder(self.n_c, self.n_c, self.n_c)
        self.hyper_decoder = HyperDecoder(self.n_c, self.n_c, self.n_c*2)
        # Define the entropy model
        self.conditional = ConditionalGaussianModel(1e-3, 1e-9)
        self.factorized_z = FullFactorizedModel(self.n_c, (3, 3, 3), 1e-9)
        self.inter_refine = Refine_Net(6, self.n_c // 2, 3)
        self.loop_filter = Refine_Net(6, self.n_c // 2, 3)
        self.temporal_filter = Refine_Net(6, self.n_c // 2, 3)
        self.feature_agg = FeatureAggregation(self.n_c * 3 // 2, self.n_c)

        #self.optical_flow = ME_Spynet()
        #self.context_model = MaskedConv2d(self.n_c, self.n_c*2, 3, 1, 1)
        #self.entropy_parameters = EntropyParameters(self.n_c*4, self.n_c*2)
    
    def forward(self, x, ref_list, pre_state = None, return_state = False, padding = None):
        ref = ref_list[:, -1]
        seq_len = ref_list.size(1)
        batch, c, h, w = ref.shape

        feature_pool = []
        h_pool = []

        for t in range(seq_len):
            feature_pool_t = self.feature_embedding(ref_list[:, t])
            h_pool.append(feature_pool_t[-1])
            feature_pool.append(feature_pool_t)
        h_pool = torch.stack(h_pool, dim = 1)

        if pre_state is not None:
            layer_output, last_state = self.conv_lstm(h_pool, pre_state)
        else:
            layer_output, last_state = self.conv_lstm(h_pool)

        hidden_feature = layer_output[-1][:, t]

        #mv = self.optical_flow(x, x)
        mv_feature = self.mv_encoder(torch.cat([ref, x], 1))
        mv_feature_hat = quantize(mv_feature, self.training)
        mv_hat = self.mv_decoder(mv_feature_hat)

        mv_prior = self.feature_agg(torch.cat([hidden_feature, mv_hat], 1))

        warp_feature, _ = self.warp_lstm(hidden_feature.unsqueeze(1), [[mv_prior[:, : self.n_c // 2], mv_prior[:, self.n_c // 2 : ]]])
        warp = self.feature_restore(warp_feature[-1][:, 0], feature_pool[t][0], feature_pool[t][1])
        pred = self.inter_refine(torch.cat([ref, warp], 1)) + warp
        hidden = self.feature_restore(hidden_feature, feature_pool[t][0], feature_pool[t][1])

        mv_hyper = self.mv_hyper_encoder(mv_feature)
        mv_hyper_hat, mv_hyper_prob = self.factorized_mv(mv_hyper)
        mv_p = self.mv_hyper_decoder(mv_hyper_hat)
        mv_loc, mv_scale_minus_one = torch.split(mv_p, self.n_c, dim = 1)
        mv_prob = self.conditional(mv_feature_hat, mv_loc, mv_scale_minus_one)
        bpp_mv = torch.sum(-torch.log2(mv_prob)) / batch / (h * w)
        bpp_mv_hyper = torch.sum(-torch.log2(mv_hyper_prob)) / batch / (h * w)

        resi = x - pred
        y = self.encoder(resi)
        y_hat = quantize(y, self.training)
        resi_hat = self.decoder(y_hat)
        x_hat = resi_hat + pred
        x_refine = self.loop_filter(torch.cat([ref, x_hat], 1)) + x_hat


        z = self.hyper_encoder(y)
        z_hat, z_prob = self.factorized_z(z)
        p = self.hyper_decoder(z_hat)
        #u = self.hyper_decoder(z_hat)
        #v = self.context_model(y_hat)
        #p = self.entropy_parameters(torch.cat((u, v), dim=1))
        loc, scale_minus_one = torch.split(p, self.n_c, dim=1)
        y_prob = self.conditional(y_hat, loc, scale_minus_one)

        if padding is not None:
            recon_loss = F.mse_loss(crop(x_refine, padding), crop(x, padding))
            warp_next_loss = F.mse_loss(crop(warp, padding), crop(x, padding))
            pred_next_loss = F.mse_loss(crop(pred, padding), crop(x, padding))
            hidden_next_loss = F.mse_loss(crop(hidden, padding), crop(x, padding))
            tmp = crop(pred, padding)
            h, w = tmp.size(2), tmp.size(3)
        else:
            recon_loss = F.mse_loss(x_refine, x)
            warp_next_loss = F.mse_loss(warp, x)
            pred_next_loss = F.mse_loss(pred, x)
            hidden_next_loss = F.mse_loss(hidden, x)

        bpp_y = torch.sum(-torch.log2(y_prob)) / batch / (h * w)
        bpp_z = torch.sum(-torch.log2(z_prob)) / batch / (h * w)
        bpp = bpp_y + bpp_z + bpp_mv + bpp_mv_hyper

        if return_state:
            return x_refine, last_state, recon_loss, warp_next_loss, pred_next_loss, hidden_next_loss, bpp, bpp_y, bpp_z, bpp_mv, bpp_mv_hyper
        else:
            return x_refine, recon_loss, warp_next_loss, pred_next_loss, hidden_next_loss, bpp, bpp_y, bpp_z, bpp_mv, bpp_mv_hyper
