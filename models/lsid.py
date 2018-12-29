import torch
import torch.nn as nn
import math
import utils

__all__ = ['LSID', 'lsid']

def pixel_shuffle(input, upscale_factor, depth_first=False):
    r"""Rearranges elements in a tensor of shape :math:`[*, C*r^2, H, W]` to a
    tensor of shape :math:`[C, H*r, W*r]`.

    See :class:`~torch.nn.PixelShuffle` for details.

    Args:
        input (Tensor): Input
        upscale_factor (int): factor to increase spatial resolution by

    Examples::

        >>> ps = nn.PixelShuffle(3)
        >>> input = torch.empty(1, 9, 4, 4)
        >>> output = ps(input)
        >>> print(output.size())
        torch.Size([1, 1, 12, 12])
    """
    batch_size, channels, in_height, in_width = input.size()
    channels //= upscale_factor ** 2

    out_height = in_height * upscale_factor
    out_width = in_width * upscale_factor

    if not depth_first:
        input_view = input.contiguous().view(
            batch_size, channels, upscale_factor, upscale_factor,
            in_height, in_width)
        shuffle_out = input_view.permute(0, 1, 4, 2, 5, 3).contiguous()
        return shuffle_out.view(batch_size, channels, out_height, out_width)
    else:
        input_view = input.contiguous().view(batch_size, upscale_factor, upscale_factor, channels, in_height, in_width)
        shuffle_out = input_view.permute(0, 4, 1, 5, 2, 3).contiguous().view(batch_size, out_height, out_width,
                                                                             channels)
        return shuffle_out.permute(0, 3, 1, 2)


class LSID(nn.Module):
    def __init__(self, inchannel=4, block_size=2):
        super(LSID, self).__init__()
        self.block_size = block_size

        self.conv1_1 = nn.Conv2d(inchannel, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.lrelu =  nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True)

        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv4_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv4_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv5_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True)

        self.up6 = nn.ConvTranspose2d(512, 256, 2, stride=2, bias=False)
        self.conv6_1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv6_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)

        self.up7 = nn.ConvTranspose2d(256, 128, 2, stride=2, bias=False)
        self.conv7_1 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv7_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True)

        self.up8 = nn.ConvTranspose2d(128, 64, 2, stride=2, bias=False)
        self.conv8_1 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv8_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)

        self.up9 = nn.ConvTranspose2d(64, 32, 2, stride=2, bias=False)
        self.conv9_1 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv9_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True)

        out_channel = 3 * self.block_size * self.block_size
        self.conv10 = nn.Conv2d(32, out_channel, kernel_size=1, stride=1, padding=0, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.lrelu(x)
        x = self.conv1_2(x)
        x = self.lrelu(x)
        conv1 = x
        x = self.maxpool(x)

        x = self.conv2_1(x)
        x = self.lrelu(x)
        x = self.conv2_2(x)
        x = self.lrelu(x)
        conv2 = x
        x = self.maxpool(x)

        x = self.conv3_1(x)
        x = self.lrelu(x)
        x = self.conv3_2(x)
        x = self.lrelu(x)
        conv3 = x
        x = self.maxpool(x)

        x = self.conv4_1(x)
        x = self.lrelu(x)
        x = self.conv4_2(x)
        x = self.lrelu(x)
        conv4 = x
        x = self.maxpool(x)

        x = self.conv5_1(x)
        x = self.lrelu(x)
        x = self.conv5_2(x)
        x = self.lrelu(x)

        x = self.up6(x)
        x = torch.cat((x[:, :, :conv4.size(2), :conv4.size(3)], conv4), 1)
        x = self.conv6_1(x)
        x = self.lrelu(x)
        x = self.conv6_2(x)
        x = self.lrelu(x)

        x = self.up7(x)
        x = torch.cat((x[:, :, :conv3.size(2), :conv3.size(3)], conv3), 1)

        x = self.conv7_1(x)
        x = self.lrelu(x)
        x = self.conv7_2(x)
        x = self.lrelu(x)

        x = self.up8(x)
        x = torch.cat((x[:, :, :conv2.size(2), :conv2.size(3)], conv2), 1)

        x = self.conv8_1(x)
        x = self.lrelu(x)
        x = self.conv8_2(x)
        x = self.lrelu(x)

        x = self.up9(x)
        x = torch.cat((x[:, :, :conv1.size(2), :conv1.size(3)], conv1), 1)

        x = self.conv9_1(x)
        x = self.lrelu(x)
        x = self.conv9_2(x)
        x = self.lrelu(x)

        x = self.conv10(x)

        depth_to_space_conv = pixel_shuffle(x, upscale_factor=self.block_size, depth_first=True)

        return depth_to_space_conv

def lsid(**kwargs):
    model = LSID(**kwargs)
    return model

