__author__ = 'yunbo'

import torch
import torch.nn as nn
from models.SpatioTemporalLSTMCell import SpatioTemporalLSTMCell


class PredRNN(nn.Module):
    def __init__(self, num_layers, num_hidden, configs):
        super(PredRNN, self).__init__()

        self.configs = configs
        self.frame_channel = configs.patch_size * configs.patch_size * 3
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        cell_list = []

        width = configs.img_width // configs.patch_size

        for i in range(num_layers):
            in_channel = self.frame_channel if i == 0 else num_hidden[i - 1]
            cell_list.append(
                SpatioTemporalLSTMCell(in_channel, num_hidden[i], width, configs.rnn_filter_size, configs.rnn_stride, configs.layer_norm)
            )
        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(num_hidden[num_layers - 1], self.frame_channel,
                                   kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, frames, is_training, h_t = None, c_t = None, memory = None):
        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        #frames = frames_tensor.permute(0, 1, 4, 2, 3).contiguous()
        #mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()

        frames = frames.permute(0, 1, 4, 2, 3).contiguous()
        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]
        total_length = frames.shape[1]

        next_frames = []

        if is_training:
            h_t = []
            c_t = []
            for i in range(self.num_layers):
                zeros = torch.zeros([batch, self.num_hidden[i], height, width]).to(self.configs.device)
                h_t.append(zeros)
                c_t.append(zeros)

            memory = torch.zeros([batch, self.num_hidden[0], height, width]).to(self.configs.device)

        for t in range(total_length):
            net = frames[:, t]

            h_t[0], c_t[0], memory = self.cell_list[0](net, h_t[0], c_t[0], memory)

            for i in range(1, self.num_layers):
                h_t[i], c_t[i], memory = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i], memory)

            x_gen = self.conv_last(h_t[self.num_layers - 1])
            next_frames.append(x_gen)

        # [batch, length, channel, height, width]
        next_frames = torch.stack(next_frames, dim = 0).permute(1, 0, 3, 4, 2).contiguous()
        #return next_frames.clamp(0., 1.)

        if is_training:
            return next_frames
        else:
            return next_frames, h_t, c_t, memory
