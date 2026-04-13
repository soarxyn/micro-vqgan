import os

import lightning as L
import torch
import wandb
from lightning.pytorch.callbacks import BasePredictionWriter
from torchvision.utils import make_grid
from lightning.pytorch.loggers import WandbLogger


from micro_vqgan.lit import LitVQGan


class LatentWriter(BasePredictionWriter):
    def __init__(self, output_dir: str) -> None:
        super().__init__(write_interval="batch")
        self.output_dir = output_dir

    def write_on_batch_end(self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx):
        os.makedirs(self.output_dir, exist_ok=True)
        torch.save(prediction.cpu(), os.path.join(self.output_dir, f"batch_{batch_idx:06d}.pt"))


class SampleCallback(L.Callback):
    def __init__(self, every_n_steps: int = 10000, num_samples: int = 8) -> None:
        self.num_samples = num_samples
        self.every_n_steps = every_n_steps
        self._fixed_batch: torch.Tensor | None = None

    def on_validation_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx=0
    ):
        if batch_idx == 0 and self._fixed_batch is None:
            self._fixed_batch = batch["pixel_values"][: self.num_samples].clone()

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs,
        batch,
        batch_idx,
    ):
        if (
            trainer.global_step == 0
            or trainer.global_step % self.every_n_steps != 0
            or not isinstance(pl_module, LitVQGan)
            or self._fixed_batch is None
        ):
            return

        pl_module.eval()

        with torch.no_grad():
            x = self._fixed_batch.to(pl_module.device)
            y, _, _ = pl_module(x)

        pl_module.train()

        x = (x * 0.5 + 0.5).clamp(0, 1)
        y = (y * 0.5 + 0.5).clamp(0, 1)

        grid = make_grid(torch.cat([x, y], dim=0), nrow=self.num_samples)

        if isinstance(trainer.logger, WandbLogger):
            trainer.logger.experiment.log(
                {
                    "samples": wandb.Image(
                        grid, caption="top: real | bottom: reconstructed"
                    )
                },
                step=trainer.global_step,
            )
