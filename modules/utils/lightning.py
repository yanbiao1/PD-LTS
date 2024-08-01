

from typing import Dict, Union
import pytorch_lightning as pl

from pytorch_lightning.callbacks import ProgressBar

class LightningProgressBar(ProgressBar):

    def get_metrics(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        # don't show the version number
        items = super().get_metrics(trainer, pl_module)
        items.pop("v_num", None)
        return items

# class LightningModule(pl.LightningModule):

    # # -----------------------------------------------------------------
    # def get_progress_bar_dict(self) -> Dict[str, Union[int, str]]:
    #     r"""
    #     Override progress_bar. See the following link for detail.
    #     https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pytorch_lightning/core/lightning.py
    #     """
    #     running_train_loss = self.trainer.fit_loop.running_loss.mean()
    #     avg_training_loss = (running_train_loss.cpu().item() if running_train_loss is not None else float("NaN"))
    #     tqdm_dict = {"loss": "{:.3f}".format(avg_training_loss)}

    #     if self.trainer.truncated_bptt_steps is not None:
    #         tqdm_dict["split_idx"] = self.trainer.split_idx
    #     return tqdm_dict
