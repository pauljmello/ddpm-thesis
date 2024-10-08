import math
import numpy as np

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch import nn, fft
from abc import abstractmethod

device = "cuda" if torch.cuda.is_available() else "cpu"

#  Set Seed for Reproducibility
#  seed = 3407  # https://arxiv.org/abs/2109.08203
seed = np.random.randint(3047)
torch.manual_seed(seed)

# MNIST Unet Taken From https://github.com/ForestsKing/DDPM/blob/master/ddpm_mnist.ipynb
# Minor Alterations Done

class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


# use GN for norm layer
def norm_layer(channels):
    return nn.GroupNorm(num_groups=channels, num_channels=channels)


# use sinusoidal position embedding to encode time step (https://arxiv.org/abs/1706.03762)
def timestep_embedding(timesteps, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10_000) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device=device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class ResidualBlock(TimestepBlock):
    def __init__(self, in_channels, out_channels, time_channels, dropout):
        super().__init__()
        self.conv1 = nn.Sequential(
            norm_layer(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )

        self.time_emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_channels, out_channels)
        )

        self.conv2 = nn.Sequential(
            norm_layer(out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t):
        """
        `x` has shape `[batch_size, in_dim, height, width]`
        `t` has shape `[batch_size, time_dim]`
        """
        h = self.conv1(x)
        h += self.time_emb(t)[:, :, None, None]
        h = self.conv2(h)
        return h + self.shortcut(x)


# Attention block with shortcut
class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=1):
        super().__init__()
        self.num_heads = num_heads
        assert channels % num_heads == 0

        self.norm = norm_layer(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.reshape(B * self.num_heads, -1, H * W).chunk(3, dim=1)
        scale = 1. / math.sqrt(math.sqrt(C // self.num_heads))
        attn = torch.einsum("bct,bcs->bts", q * scale, k * scale)
        attn = attn.softmax(dim=-1)
        h = torch.einsum("bts,bcs->bct", attn, v)
        h = h.reshape(B, -1, H, W)
        h = self.proj(h)
        return h + x


class Upsample(nn.Module):
    def __init__(self, channels, use_conv):
        super().__init__()
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, channels, use_conv):
        super().__init__()
        self.use_conv = use_conv
        if use_conv:
            self.op = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
        else:
            self.op = nn.AvgPool2d(stride=2, kernel_size=3)

    def forward(self, x):
        return self.op(x)


def Fourier_filter(x, threshold, scale):
    s = x.shape[-2:]
    padding = [2 ** np.ceil(np.log2(dim)) - dim for dim in s]
    x_padded = F.pad(x, (0, int(padding[1]), 0, int(padding[0])))
    x_padded = x_padded.to(torch.complex64)  # convert to complex (scalar requires float32, fft.fft2 requires float64)
    x_freq = fft.fftn(x_padded, dim=(-2, -1))
    B, C, H, W = x_freq.shape
    crow, ccol = H // 2, W // 2
    mask = torch.full((B, C, H, W), scale, device=x.device)
    mask[..., crow - threshold:crow + threshold, ccol - threshold:ccol + threshold] = 1.0
    x_freq.mul_(mask)
    x_filtered_padded = fft.ifftn(x_freq, dim=(-2, -1)).real
    x_filtered = x_filtered_padded[..., :s[0], :s[1]]
    return x_filtered


class unet(nn.Module):
    def __init__(
            self,
            in_channels,
            model_channels,
            out_channels,
            num_res_blocks,
            attention_resolutions,
            dropout,
            channel_mult,
            conv_resample,
            num_heads,
            FreeU,
            b1,
            b2,
            s1,
            s2
    ):
        super().__init__()

        self.empowerment_values = []
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_heads = num_heads
        self.FreeU = FreeU
        self.b1 = b1
        self.b2 = b2
        self.s1 = s1
        self.s2 = s2

        # time embedding
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SELU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # down blocks
        self.down_blocks = nn.ModuleList([
            TimestepEmbedSequential(nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1))
        ])
        down_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResidualBlock(ch, mult * model_channels, time_embed_dim, dropout)
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads=num_heads))
                self.down_blocks.append(TimestepEmbedSequential(*layers))
                down_block_chans.append(ch)
            if level != len(channel_mult) - 1:  # don't use downsample for the last stage
                self.down_blocks.append(TimestepEmbedSequential(Downsample(ch, self.conv_resample)))
                down_block_chans.append(ch)
                ds *= 2

        # middle block
        self.middle_block = TimestepEmbedSequential(
            ResidualBlock(ch, ch, time_embed_dim, dropout),
            AttentionBlock(ch, num_heads=num_heads),
            ResidualBlock(ch, ch, time_embed_dim, dropout)
        )

        # up blocks
        self.up_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = [
                    ResidualBlock(
                        ch + down_block_chans.pop(),
                        model_channels * mult,
                        time_embed_dim,
                        dropout
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads=num_heads))
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch, conv_resample))
                    ds //= 2
                self.up_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            norm_layer(ch),
            nn.SiLU(),
            nn.Conv2d(model_channels, out_channels, kernel_size=3, padding=1),
        )

    def resetEmpowermentValues(self):
        self.empowerment_values = []

    def getEmpowermentValues(self):
        return self.empowerment_values

    # Entropy Computation
    def compute_empowerment(self, x):
        assert len(x.shape) == 4
        probs = torch.softmax(x, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=1)
        empowerment = entropy.mean()
        return empowerment.item()

    def plot_empowerment_dynamics(self, savePath):
        plt.figure(figsize=(12, 8))  # Increase the size of the figure
        plt.plot(self.empowerment_values, '-o', label="Mean Empowerment", markersize=1)  # Plot mean empowerment
        plt.xlabel("Model Layer")
        plt.ylabel("Empowerment Value")
        plt.title("Mean Empowerment Across Model Layers")
        plt.grid(True)
        plt.legend()
        plt.savefig(savePath)
        plt.close()


    def forward(self, x, timesteps):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """

        hs = []

        # time step embedding
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        # down stage
        h = x
        for module in self.down_blocks:
            h = module(h, emb)
            #self.empowerment_values.append(self.compute_empowerment(h))
            hs.append(h)

        # middle stage
        h = self.middle_block(h, emb)

        # up stage
        for module in self.up_blocks:
            hs_ = hs.pop()
            if self.FreeU:
                # -------------------- FreeU code ------------------------
                if h.shape[1] == 64 and hs_.shape[1] == 32:
                    h[:, :512] = h[:, :512] * self.b2
                    hs_ = Fourier_filter(hs_, threshold=1, scale=self.s2)
                elif h.shape[1] == 128 and hs_.shape[1] == 64:
                    h[:, :512] = h[:, :512] * self.b1
                    hs_ = Fourier_filter(hs_, threshold=1, scale=self.s1)
                # ---------------------------------------------------------
            cat_in = torch.cat([h, hs_], dim=1)
            h = module(cat_in, emb)
            #self.empowerment_values.append(self.compute_empowerment(h))
        return self.out(h)