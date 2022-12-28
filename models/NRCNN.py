import torch
import torch.nn.functional as F

class eca_layer(torch.nn.Module):
    """Constructs a ECA module.  Args:
    channel: Number of channels of the input feature map
    k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=5):
        super(eca_layer, self).__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.conv = torch.nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class ResBlock(torch.nn.Module):
    def __init__(self, n_c, channel_attention):
        super(ResBlock, self).__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(n_c, n_c, 3, 1, 1),
            torch.nn.ReLU(inplace = True),
            torch.nn.Conv2d(n_c, n_c, 3, 1, 1),
        )
        self.channel_attention = channel_attention
        if channel_attention == 1:
            self.eca = eca_layer(n_c)


    def forward(self, x):
        out = self.block(x)
        if self.channel_attention == 1:
            out = self.eca(out)
        out = out + x
        return out

class ResBlock_Large(torch.nn.Module):
    def __init__(self, n_c, channel_attention):
        super(ResBlock_Large, self).__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(n_c, n_c * 2, 1),
            torch.nn.ReLU(inplace = True),
            torch.nn.Conv2d(n_c * 2, n_c * 2, 3, 1, 1),
            torch.nn.ReLU(inplace = True),
            torch.nn.Conv2d(n_c * 2, n_c * 2, 3, 1, 1),
            torch.nn.ReLU(inplace = True),
            torch.nn.Conv2d(n_c * 2, n_c, 1),
        )
        self.channel_attention = channel_attention
        if channel_attention == 1:
            self.eca = eca_layer(n_c)

    def forward(self, x):
        out = self.block(x)
        if self.channel_attention == 1:
            out = self.eca(out)
        out = out + x
        return out

class NRCNN(torch.nn.Module):
    def __init__(self, n_block, n_i, n_c, channel_attention = 0, resblock_type = 1):
        super(NRCNN, self).__init__()
        self.part1 = torch.nn.Sequential(
            torch.nn.Conv2d(n_i, n_c, 3, 1, 1), 
            torch.nn.ReLU(inplace = True),
        )
        self.part2 = torch.nn.ModuleList()
        for i in range(n_block):
            if resblock_type == 1:
                self.part2.append(ResBlock(n_c, channel_attention))
            elif resblock_type == 2:
                self.part2.append(ResBlock_Large(n_c, channel_attention))
            else:
                raise Exception("resblock type error:", resblock_type)

        self.part3 = torch.nn.Sequential(
            torch.nn.Conv2d(n_c, n_c, 3, 1, 1),
            torch.nn.ReLU(inplace = True),
            torch.nn.Conv2d(n_c, 1, 3, 1, 1),
        )

    def forward(self, x):
        out = self.part1(x)
        for layer in self.part2:
            out = layer(out)
        out = self.part3(out)
        return out
        
class EXNRCNN(torch.nn.Module):
    def __init__(self, n_block, n_i, n_c, resblock_type, channel_attention):
        super(EXNRCNN, self).__init__()
        self.part1_1 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(n_i, n_c // 2, 4, 2, 1),
            torch.nn.ReLU(inplace = True),
            torch.nn.Conv2d(n_c // 2, n_c // 2, 3, 1, 1), 
            torch.nn.ReLU(inplace = True),
        )
        self.part1_2 = torch.nn.Sequential(
            torch.nn.Conv2d(n_i, n_c // 2, 3, 1, 1), 
            torch.nn.ReLU(inplace = True),
        )
        self.part2 = torch.nn.ModuleList()
        for i in range(n_block):
            if resblock_type == 1:
                self.part2.append(ResBlock(n_c, channel_attention))
            elif resblock_type == 2:
                self.part2.append(ResBlock_Large(n_c, channel_attention))
            else:
                raise Exception("resblock type error:", resblock_type)

        self.part3 = torch.nn.Sequential(
            torch.nn.Conv2d(n_c, n_c, 3, 1, 1),
            torch.nn.ReLU(inplace = True),
            torch.nn.Conv2d(n_c, 1, 3, 1, 1),
            torch.nn.AvgPool2d(2, 2),
        )

    def forward(self, luma, chroma):
        chroma_out = self.part1_1(chroma)
        luma_out = self.part1_2(luma)
        out = torch.cat([chroma_out, luma_out], 1)
        for layer in self.part2:
            out = layer(out)
        out = self.part3(out)
        return out


if __name__ == '__main__':
    rbn = RBN(8, 64)
    rbn.cuda()
    a = torch.ones(16, 1, 64, 64)
    output = rbn(a.cuda())
    print(output.size(), output)
