import torch
import torch.nn as nn
import torch.nn.functional as F

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        nn.init.orthogonal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Embedding') != -1:
         nn.init.orthogonal_(m.weight.data)

class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.embed = nn.Embedding(num_classes, num_features * 2)
        self.embed.weight.data[:, :num_features].fill_(1.0)
        self.embed.weight.data[:, num_features:].zero_()

    def forward(self, x, y):
        out = self.bn(x)
        gamma_beta = self.embed(y)          # (N, 2C)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta  = beta.unsqueeze(-1).unsqueeze(-1)
        return gamma * out + beta

class ResBlockG(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False)
        self.cbn1   = ConditionalBatchNorm2d(in_ch, 10)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.cbn2   = ConditionalBatchNorm2d(out_ch, 10)

        # skip path
        self.conv_skip = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False) if in_ch != out_ch else None

    def forward(self, x,y):
        
        h = self.cbn1(x,y)
        h = F.relu(h, inplace=True)
        h = self.upsample(h)
        h = self.conv1(h)
        h = self.cbn2(h,y)
        h = F.relu(h, inplace=True)
        h = self.conv2(h)

        s = self.upsample(x)
        if self.conv_skip is not None:
            s = self.conv_skip(s)

        return h + s
class Generator(nn.Module):
    def __init__(self, z_dim=128, base_channels=256, img_channels=3):
        super().__init__()
        self.fc = nn.Linear(z_dim, base_channels * 4 * 4)  # 4x4 feature map

        self.block1 = ResBlockG(base_channels, base_channels)      # 4 -> 8
        self.block2 = ResBlockG(base_channels, base_channels // 2) # 8 -> 16
        self.block3 = ResBlockG(base_channels // 2, base_channels // 4)  # 16 -> 32

        self.conv_out = nn.Conv2d(base_channels // 4, img_channels, 3, 1, 1)
        self.tanh = nn.Tanh()

    def forward(self, z, y):
        h = self.fc(z).view(z.size(0), -1, 4, 4)
        h = self.block1(h,y)
        h = self.block2(h,y)
        h = self.block3(h,y)
        x = self.tanh(self.conv_out(h))
        return x
class ResBlockD(nn.Module):
    def __init__(self, in_ch, out_ch, downsample=True):
        super().__init__()
        self.downsample = downsample

        self.conv1 = nn.utils.spectral_norm(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False)
        )
        self.conv2 = nn.utils.spectral_norm(
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        )

        self.conv_skip = nn.utils.spectral_norm(
            nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)
        ) if in_ch != out_ch or downsample else None

    def forward(self, x):
        h = F.relu(x)
        h = self.conv1(h)
        h = F.relu(h)
        h = self.conv2(h)
        
        if self.downsample:
            h = F.avg_pool2d(h, 2)

        s = x
        if self.downsample:
            s = F.avg_pool2d(s, 2)
        if self.conv_skip is not None:
            s = self.conv_skip(s)

        return h + s
class Discriminator(nn.Module):
    def __init__(self, base_channels=256, img_channels=3):
        super().__init__()
        self.conv_in = nn.utils.spectral_norm(
            nn.Conv2d(img_channels, base_channels // 2, 3, 1, 1, bias=False)
        )

        self.block1 = ResBlockD(base_channels // 2, base_channels,   downsample=True)  # 32 -> 16
        self.block2 = ResBlockD(base_channels,     base_channels,   downsample=True)  # 16 -> 8
        self.block3 = ResBlockD(base_channels,     base_channels,   downsample=True)  # 8 -> 4

        self.fc = nn.utils.spectral_norm(nn.Linear(base_channels * 4 * 4, 1))
        self.embed = nn.utils.spectral_norm(nn.Embedding(10, base_channels*4*4))
    def forward(self, x,y):
        h = F.leaky_relu(self.conv_in(x), 0.2, inplace=True)
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = h.view(h.size(0), -1)
        label_proj = (h*self.embed(y)).sum(dim=1)
        out = self.fc(h)
        return out.view(-1) + label_proj
    
