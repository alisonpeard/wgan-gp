# %%
import torch
import numpy as np
import torch.nn.functional as F
import warnings
from typing import Tuple, Union


def tuple_to_numpy(tup):
    tup = tuple([x.item() for x in tup])
    return tup


def tuple_to_torch(tup):
    if isinstance(tup[0], torch.Tensor):
        return tup
    tup = tuple([torch.tensor(x) for x in tup])
    return tup


def downsize(insize, k, s, p) -> tuple:
    h = torch.floor(((insize[0]-k[0]+2*p)/s) + 1).int()
    w = torch.floor(((insize[1]-k[1]+2*p)/s) + 1).int()
    return (h, w)


def upsize(insize, k, s, p) -> tuple:
   h = torch.floor((insize[0] - 1) * s - 2 * p + (k[0] - 1) + 1).int()
   w = torch.floor((insize[1] - 1) * s - 2 * p + (k[1] - 1) + 1).int()
   return (h, w)
   

def upsample(x, kernel_size:tuple, stride:int=1, padding:int=0, mode='bilinear') -> torch.Tensor:
    insize = x.size()[2:]
    kernel_size = tuple_to_torch(kernel_size)
    outsize = upsize(insize, kernel_size, stride, padding)
    upsampled = F.interpolate(x, size=outsize, mode=mode)
    return upsampled


def downsample(x, kernel_size:tuple, stride:int=1, padding:int=0) -> torch.Tensor:
    kernel_size = tuple_to_torch(kernel_size)
    downsampled = F.avg_pool2d(x, kernel_size=kernel_size, stride=stride, padding=padding)
    return downsampled


def batchrank(x:torch.Tensor) -> torch.Tensor:
    """Differentiable ranking."""
    # from torchsort import soft_rank
    from ranking.pytorch_ops import soft_rank
    b, h, w, c, = x.size()
    x = x.reshape(b, h*w*c)
    x = x.t()
    x = soft_rank(x)
    x = x.t()
    x = x.reshape(b, h, w, c)
    return x


class GumbelEsque(torch.nn.Module):
    """Total experiment, NOTE: not using."""
    def __init__(self):
        super().__init__()
        raise NotImplementedError("Function to slow to use in training.")

    def forward(self, x):
        n = x.size(0)
        x = batchrank(x) / (n + 1)
        x = -torch.log(-torch.log(x))
        return x


class ResidualUpBlock(torch.nn.Module):
    """Single residual block for upsampling (increasing resolution)."""
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Tuple,
                 stride: int = 1,
                 padding: int = 0,
                 upsample_mode: str = 'bilinear',
                 **kwargs
                 ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.deconv = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        self.activation = torch.nn.ReLU() # GELU(), SiLU()
        self.norm = torch.nn.BatchNorm2d(out_channels)
        self.upsample = lambda x: upsample(x, kernel_size, stride, padding, upsample_mode)
        self.project = torch.nn.Conv2d(self.in_channels, self.out_channels, 1, 1)
    
    def forward(self, x) -> torch.Tensor:
        identity = self.upsample(x)
        identity = self.project(identity)
        x = self.deconv(x)
        x = self.activation(x)
        x = self.norm(x)
        return x + identity


class ResidualDownBlock(torch.nn.Module):
    """Single residual block for downsampling (decreasing resolution)."""
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 0,
                 upsample_mode: str = 'bilinear',
                 **kwargs) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.activation = torch.nn.ReLU() # GELU()
        self.downsample = lambda x: downsample(x, kernel_size, stride, padding)
        self.project = torch.nn.Conv2d(self.in_channels, self.out_channels, 1, 1)

    def forward(self, x) -> torch.Tensor:
        identity = self.project(self.downsample(x))
        x = self.conv(x)
        x = self.activation(x)
        return x + identity


# %% tests
if __name__ == "__main__":
    print('Running tests...')
    x = np.arange(0, 25, 1).reshape((1,1,5,5))
    x = torch.Tensor(x)
    xsize = x.shape[2:]

    kernel_sizes = [(2, 2), (2, 3)]
    strides = [1, 2]
    paddings = [0, 1]

    for kernel_size in kernel_sizes:
        for stride in strides:
            for padding in paddings:
                # test upsampling
                kernel_size = tuple_to_torch(kernel_size)
                up_expected = upsize(xsize, kernel_size, stride, padding)
                up_expected = tuple_to_numpy(up_expected)
                up_calculated = upsample(x, kernel_size, stride, padding).size()[2:]
                up_calculated = tuple(up_calculated)
                assert up_calculated == up_expected

                # # test downsampling
                down_expected = downsize(xsize, kernel_size, stride, padding)
                down_expected = tuple_to_numpy(down_expected)
                down_calculated = downsample(x, kernel_size, stride, padding).size()[2:]
                down_calculated = tuple(down_calculated)
                assert down_calculated == down_expected

                # # test upblock
                upblock = ResidualUpBlock(1, 2, kernel_size, stride, padding)
                y = upblock(x)
                y2 = torch.nn.ConvTranspose2d(1, 2, kernel_size, stride, padding)(x)
                assert y.shape == y2.shape

                # test downblock
                downblock = ResidualDownBlock(1, 2, kernel_size, stride, padding)
                y = downblock(x)
                y2 = torch.nn.Conv2d(1, 2, kernel_size, stride, padding)(x)
                assert y.shape == y2.shape

                # test batchranking
                x = np.array([[3, 3], [4, 4], [1,2], [0, 5]]).reshape(4, 2, 1, 1)
                x = torch.tensor(x)
                assert np.array_equal(batchrank(x)[:, :, 0, 0], torch.tensor([[3., 2.],[4., 3.], [2., 1.], [1., 4.]]))
    
    print('All tests passed!')


# %%