import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange


class Codebook(nn.Module):
    def __init__(self, size: int, emb_dim: int, beta: float = 0.25):
        super().__init__()

        self.embedding = nn.Embedding(size, emb_dim)
        nn.init.kaiming_uniform_(self.embedding.weight)

        self.hidden_dim = emb_dim
        self.beta = beta

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = rearrange(x, "b c h w -> b h w c").contiguous()
        x_flattened = x.view(-1, self.hidden_dim)

        distances = (
            x_flattened.square().sum(dim=1, keepdim=True)
            + self.embedding.weight.square().sum(dim=1)
            - 2 * x_flattened @ self.embedding.weight.t()
        )
        indices = distances.argmin(dim=1)

        x_q: torch.Tensor = self.embedding(indices).view(x.shape)

        loss = (
            self.beta * (x_q.detach() - x).square().mean()
            + (x_q - x.detach()).square().mean()
        )

        x_q = x + (x_q - x).detach()  # STE
        x_q = rearrange(x_q, "b h w c -> b c h w").contiguous()

        return x_q, loss, indices


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim**0.5
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim=1) * self.g * self.scale


class Attention(nn.Module):
    def __init__(self, in_channels: int, num_heads: int, head_channels: int):
        super().__init__()

        self.num_heads = num_heads

        self.qkv_proj = nn.Conv2d(
            in_channels, 3 * num_heads * head_channels, kernel_size=1, bias=False
        )
        self.out_proj = nn.Conv2d(
            num_heads * head_channels, in_channels, kernel_size=1, bias=False
        )

        self.norm = RMSNorm(in_channels)

        nn.init.zeros_(self.out_proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2], x.shape[-1]

        res = x
        x = self.norm(x)

        QKV = self.qkv_proj(x)

        Q, K, V = rearrange(
            QKV, "b (r heads d) h w -> r b heads (h w) d", r=3, heads=self.num_heads
        )

        A = F.scaled_dot_product_attention(Q, K, V)
        A = rearrange(A, "b heads (h w) d -> b (heads d) h w", h=h, w=w)

        return self.out_proj(A) + res


class Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()

        self.proj = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.norm = RMSNorm(out_channels)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = self.norm(x)
        x = self.act(x)

        return self.dropout(x)


class ResNetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.block_1 = Block(in_channels, out_channels, dropout=dropout)
        self.block_2 = Block(out_channels, out_channels)

        self.res_conv = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        h = self.block_1(x)
        h = self.block_2(h)

        return h + self.res_conv(x)


def Upsample(dim, dim_out):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(dim, dim_out, 3, padding=1),
    )


def Downsample(dim, dim_out):
    return nn.Sequential(
        Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
        nn.Conv2d(dim * 4, dim_out, 1),
    )


class Encoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        in_channels: int,
        z_channels: int,
        num_heads: int = 4,
        head_channels: int = 64,
        multipliers: tuple[int, ...] = (1, 2, 4, 8),
        attention_levels: tuple[int, ...] = (-1,),
        dropout: float = 0.0,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.conv_in = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)

        dimensions = [hidden_dim, *map(lambda mult: hidden_dim * mult, multipliers)]
        in_out = list(zip(dimensions[:-1], dimensions[1:]))

        self.hidden_dim = hidden_dim

        self.down_blocks: nn.ModuleList = nn.ModuleList([])

        n_levels = len(in_out)
        attn_levels = {i % n_levels for i in attention_levels}

        for idx, (dim_in, dim_out) in enumerate(in_out):
            is_last = idx >= (n_levels - 1)

            self.down_blocks.append(
                nn.ModuleList(
                    [
                        ResNetBlock(dim_in, dim_in, dropout),
                        ResNetBlock(dim_in, dim_in, dropout),
                        Attention(dim_in, num_heads, head_channels)
                        if idx in attn_levels
                        else nn.Identity(),
                        Downsample(dim_in, dim_out)
                        if not is_last
                        else nn.Conv2d(dim_in, dim_out, kernel_size=3, padding=1),
                    ]
                )
            )

        mid_dim = dimensions[-1]

        self.mid_block_1 = ResNetBlock(mid_dim, mid_dim, dropout)
        self.mid_attn = Attention(mid_dim, num_heads, head_channels)
        self.mid_block_2 = ResNetBlock(mid_dim, mid_dim, dropout)

        self.norm_out = RMSNorm(mid_dim)
        self.conv_out = nn.Conv2d(mid_dim, z_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)

        for block1, block2, attn, downsample in self.down_blocks:  # type: ignore
            x = block1(x)

            x = block2(x)
            x = attn(x)

            x = downsample(x)

        x = self.mid_block_1(x)
        x = self.mid_attn(x)
        x = self.mid_block_2(x)

        x = self.norm_out(x)
        x = F.silu(x)
        return self.conv_out(x)


class Decoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        out_channels: int,
        z_channels: int,
        num_heads: int = 4,
        head_channels: int = 64,
        multipliers: tuple[int, ...] = (1, 2, 4, 8),
        attention_levels: tuple[int, ...] = (-1,),
        dropout: float = 0.0,
    ):
        super().__init__()

        dimensions = [hidden_dim, *map(lambda mult: hidden_dim * mult, multipliers)]
        in_out = list(zip(dimensions[:-1], dimensions[1:]))

        self.hidden_dim = hidden_dim

        self.up_blocks: nn.ModuleList = nn.ModuleList([])

        n_levels = len(in_out)
        attn_levels = {i % n_levels for i in attention_levels}

        mid_dim = dimensions[-1]

        self.conv_in = nn.Conv2d(z_channels, mid_dim, kernel_size=3, padding=1)
        self.mid_block_1 = ResNetBlock(mid_dim, mid_dim, dropout)
        self.mid_attn = Attention(mid_dim, num_heads, head_channels)
        self.mid_block_2 = ResNetBlock(mid_dim, mid_dim, dropout)

        for idx, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = idx == n_levels - 1
            down_idx = n_levels - 1 - idx

            self.up_blocks.append(
                nn.ModuleList(
                    [
                        ResNetBlock(dim_out, dim_out, dropout),
                        ResNetBlock(dim_out, dim_out, dropout),
                        Attention(dim_out, num_heads, head_channels)
                        if down_idx in attn_levels
                        else nn.Identity(),
                        Upsample(dim_out, dim_in)
                        if not is_last
                        else nn.Conv2d(dim_out, dim_in, kernel_size=3, padding=1),
                    ]
                )
            )

        self.final_res_block = ResNetBlock(hidden_dim, hidden_dim)
        self.final_conv = nn.Conv2d(hidden_dim, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        x = self.mid_block_1(x)
        x = self.mid_attn(x)
        x = self.mid_block_2(x)

        for block1, block2, attn, upsample in self.up_blocks:  # type: ignore
            x = block1(x)
            x = block2(x)
            x = attn(x)

            x = upsample(x)

        x = self.final_res_block(x)
        return self.final_conv(x).tanh()


class NLayerDiscriminator(nn.Module):
    def __init__(self, in_channels: int = 3, num_layers: int = 3, hidden_dim: int = 64):
        super().__init__()

        sequence = [
            nn.Conv2d(in_channels, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
        ]

        dimensions = [
            hidden_dim,
        ] + [min(2**n, 8) * hidden_dim for n in range(1, num_layers + 1)]
        in_out = list(zip(dimensions[:-1], dimensions[1:]))

        for dim_in, dim_out in in_out:
            sequence.extend(
                [
                    nn.Conv2d(
                        dim_in, dim_out, kernel_size=4, stride=2, padding=1, bias=False
                    ),
                    nn.BatchNorm2d(dim_out),
                    nn.LeakyReLU(0.2, True),
                ]
            )

        sequence.append(
            nn.Conv2d(dimensions[-1], 1, kernel_size=4, stride=1, padding=1)
        )

        self.main = nn.Sequential(*sequence)

        self.main.apply(self._init_weights)

    def _init_weights(self, layer):
        if isinstance(layer, nn.Conv2d):
            nn.init.normal_(layer.weight.data, 0.0, 0.02)
        elif isinstance(layer, nn.BatchNorm2d):
            nn.init.normal_(layer.weight.data, 1.0, 0.02)
            layer.bias.data.fill_(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)
