import torch
from lightning.pytorch.cli import LightningCLI

from micro_vqgan.lit import LitVQGan


def cli():
    torch.set_float32_matmul_precision("medium")
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    LightningCLI(
        model_class=LitVQGan,
        save_config_callback=None,
    )


if __name__ == "__main__":
    cli()
