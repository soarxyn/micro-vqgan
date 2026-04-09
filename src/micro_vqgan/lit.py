from typing import Self

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from micro_vqgan.model import Codebook, Decoder, Encoder, NLayerDiscriminator

# Model was based off the implementation from:
# Taming Transformers (https://github.com/CompVis/taming-transformers/blob/master/taming/modules/losses/vqperceptual.py)
# lucidrains' diffusion UNet (https://codeberg.org/lucidrains/denoising-diffusion-pytorch/src/branch/main/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py)


class LitVQGan(L.LightningModule):
    def __init__(
        self,
        hidden_dim: int,
        z_channels: int,
        emb_dim: int,
        codebook_size: int,
        in_channels: int,
        out_channels: int,
        num_heads: int = 4,
        head_channels: int = 64,
        multipliers: tuple[int, ...] = (1, 2, 4, 8),
        attention_levels: tuple[int, ...] = (-1,),
        dropout: float = 0.0,
        lr: float = 1e-3,
        perceptual_weight: float = 1.0,
        codebook_weight: float = 1.0,
        discriminator_weight: float = 1.0,
        discriminator_starting_step: int = 10000,
        image_size: int = 256,
    ):
        super().__init__()

        self.automatic_optimization = False
        self.save_hyperparameters(ignore=["model"])
        self.example_input_array = torch.randn(1, in_channels, image_size, image_size)

        self.encoder = Encoder(
            hidden_dim,
            in_channels,
            z_channels,
            num_heads,
            head_channels,
            multipliers,
            attention_levels,
            dropout,
        )

        self.pre_quant_conv = nn.Conv2d(z_channels, emb_dim, 1)
        self.codebook = Codebook(codebook_size, emb_dim)
        self.post_quant_conv = nn.Conv2d(emb_dim, z_channels, 1)

        self.decoder = Decoder(
            hidden_dim,
            out_channels,
            z_channels,
            num_heads,
            head_channels,
            multipliers,
            attention_levels,
            dropout,
        )

        self.discriminator = NLayerDiscriminator(in_channels=in_channels)

        self.lr = lr
        self.perceptual_weight = perceptual_weight
        self.codebook_weight = codebook_weight
        self.discriminator_weight = discriminator_weight
        self.discriminator_starting_step = discriminator_starting_step
        self.image_size = image_size

        self.perceptual_loss = LearnedPerceptualImagePatchSimilarity(
            net_type="vgg", normalize=False, reduction="mean"
        )
        self._lpips_shift: torch.Tensor
        self._lpips_scale: torch.Tensor
        self.register_buffer('_lpips_shift', torch.tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer('_lpips_scale', torch.tensor([.458, .448, .450])[None, :, None, None])

    def _scale_for_lpips(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self._lpips_shift) / self._lpips_scale

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        h = self.pre_quant_conv(self.encoder(x))
        x_q, loss, indices = self.codebook(h)

        return x_q, loss, indices

    def decode(self, x_q: torch.Tensor) -> torch.Tensor:
        h = self.post_quant_conv(x_q)
        return self.decoder(h)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        h = self.pre_quant_conv(self.encoder(x))
        x_q, codebook_loss, indices = self.codebook(h)
        h = self.decoder(self.post_quant_conv(x_q))

        return h, codebook_loss, indices

    def _get_lambda_weight(
        self, nll: torch.Tensor, gan_loss: torch.Tensor
    ) -> torch.Tensor:
        nll_grad = torch.autograd.grad(
            nll, self.decoder.final_conv.weight, retain_graph=True
        )[0]
        gan_grad = torch.autograd.grad(
            gan_loss, self.decoder.final_conv.weight, retain_graph=True
        )[0]

        return self.discriminator_weight * (
            (nll_grad.norm() / (gan_grad.norm() + 1e-4)).clamp(0, 1e4).detach()
        )

    def training_step(self, batch, _):
        opts = self.optimizers()
        opt_vae, opt_disc = opts[0], opts[1]  # type: ignore[index]

        x = batch["pixel_values"]
        y, codebook_loss, indices = self.forward(x)

        l_rec = (x - y).abs().mean()
        l_perceptual = self.perceptual_loss(self._scale_for_lpips(x), self._scale_for_lpips(y))
        nll = l_rec + self.perceptual_weight * l_perceptual

        # Generator step
        opt_vae.zero_grad()

        gan_loss = -self.discriminator(y).mean()
        adaptive_weight = (
            self._get_lambda_weight(nll, gan_loss)
            if self.global_step >= self.discriminator_starting_step
            else 0.0
        )
        vae_loss = (
            nll
            + self.codebook_weight * codebook_loss.mean()
            + adaptive_weight * gan_loss
        )

        self.log_dict(
            {
                "train/l_rec": l_rec,
                "train/l_perceptual": l_perceptual,
                "train/nll": nll,
                "train/vae_loss": vae_loss,
                "train/codebook_loss": codebook_loss.mean(),
                "train/gan_loss": gan_loss,
                "train/adaptive_weight": adaptive_weight,
            },
            on_step=True,
            on_epoch=False,
        )

        encodings = F.one_hot(indices, self.codebook.embedding.num_embeddings).float()
        avg_probs = encodings.mean(0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        usage = (avg_probs > 0).sum()

        self.log_dict(
            {
                "train/perplexity": perplexity,
                "train/codebook_usage": usage,
            },
            on_step=True,
            on_epoch=False,
        )

        self.manual_backward(vae_loss)
        opt_vae.step()

        # Discriminator step
        opt_disc.zero_grad()

        if self.global_step >= self.discriminator_starting_step:
            real_loss = F.relu(1.0 - self.discriminator(x)).mean()
            fake_loss = F.relu(1.0 + self.discriminator(y.detach())).mean()
            disc_loss = 0.5 * (real_loss + fake_loss)
            self.manual_backward(disc_loss)
            self.log("train/disc_loss", disc_loss, on_step=True, on_epoch=False)

        opt_disc.step()

    def validation_step(self, batch, _):
        x = batch["pixel_values"]
        y, codebook_loss, indices = self.forward(x)

        l_rec = (x - y).abs().mean()
        l_perceptual = self.perceptual_loss(self._scale_for_lpips(x), self._scale_for_lpips(y))
        nll = l_rec + self.perceptual_weight * l_perceptual

        encodings = F.one_hot(indices, self.codebook.embedding.num_embeddings).float()
        avg_probs = encodings.mean(0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        self.log_dict(
            {
                "val/l_rec": l_rec,
                "val/l_perceptual": l_perceptual,
                "val/nll": nll,
                "val/codebook_loss": codebook_loss.mean(),
                "val/perplexity": perplexity,
            },
            on_step=False,
            on_epoch=True,
        )

    def train(self, mode: bool = True) -> Self:
        super().train(mode)
        self.perceptual_loss.eval()
        return self

    def configure_optimizers(self):
        vae_params = [
            p
            for name, p in self.named_parameters()
            if not name.startswith("discriminator.")
            and not name.startswith("perceptual_loss.")
        ]

        return [
            AdamW(vae_params, lr=self.lr, betas=(0.5, 0.9)),
            AdamW(self.discriminator.parameters(), lr=self.lr, betas=(0.5, 0.9)),
        ]
