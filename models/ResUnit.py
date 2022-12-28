import torch
import torch.nn as nn


class EncResUnit(nn.Module):
    def __init__(self, channels, features, stride):
        super().__init__()
        self._c = channels
        self._f = features
        self._s = stride

        self.conv1 = nn.Conv2d(self._c, self._f, self._s+1*2, self._s, 1, padding_mode="replicate")
        self.relu = nn.PReLU(self._f)
        self.conv2 = nn.Conv2d(self._f, self._c, 3, 1, 1, padding_mode="replicate")
        if self._s > 1:
            self.down = nn.AvgPool2d(self._s)

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu(y)
        y = self.conv2(y)

        if self._s == 1:
            z = x
        else:
            z = self.down(x)

        return y + z


class DecResUnit(nn.Module):
    def __init__(self, channels, features, stride):
        super().__init__()
        self._c = channels
        self._f = features
        self._s = stride

        self.conv1 = nn.ConvTranspose2d(self._c, self._f, 3, 1, 1, padding_mode="zeros")
        self.relu = nn.PReLU(self._f)
        self.conv2 = nn.ConvTranspose2d(self._f, self._c, self._s+1*2, self._s, 1, padding_mode="zeros")
        if self._s > 1:
            self.up = nn.Upsample(scale_factor=self._s, mode="nearest")

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu(y)
        y = self.conv2(y)

        if self._s == 1:
            z = x
        else:
            z = self.up(x)

        return y + z


if __name__ == "__main__":
    types = {"device": torch.device("cuda:0"), "dtype": torch.float32}
    model = DecResUnit(64, 64, 2).to(**types)

    inputs = torch.zeros(2, 64, 64, 64, **types)
    outputs = model(inputs)

    print(outputs.shape)
