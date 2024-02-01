import os
from os import path
from typing import Literal, List, Callable
import numpy as np

from lightning.pytorch.callbacks import Callback

from .models.MLP import NeuralImplicitMLP
from .utils import save_imgs, apply_transforms
from . import ABCDataset, ABCTransform


class NeuralImplicitImageLogger(Callback):
    def __init__(
        self,
        view_transforms: List[ABCTransform] = [],
        save_freq: int = 1,
        best_only=True,
        disabled=False,
        **kwargs,
    ):
        super().__init__()
        self.view_transforms = view_transforms
        self.disabled = disabled
        self.save_freq = save_freq
        self.val_idx = 0
        self.image_shape = None

        self.best_only = best_only
        self.min_loss = np.inf

    def _save_imgs(
        self,
        imgs: List[np.ndarray],
        save_path: os.PathLike,
        labels: List[str] = [],
        metrics: dict = {},
        method: Literal["pil", "plt", "cv2"] = "pil",
        **plotting_kwargs,
    ):
        save_imgs(
            imgs,
            save_path,
            labels=labels,
            metrics=metrics,
            method=method,
            **plotting_kwargs,
        )

    def _apply_transforms(self, imgs: List[np.ndarray]):
        return apply_transforms(imgs, self.view_transforms)

    def setup(self, trainer, pl_module: NeuralImplicitMLP, stage):
        """Called when fit, validate, test, or predict begins"""

        dataset = trainer.datamodule.datasets[stage]
        assert isinstance(
            dataset, ABCDataset
        ), "Dataset must be derived from ABCDataset"

        self.orig_img = dataset.image  # MUST OCCUR FOR VALIDATION
        self.image_shape = dataset.input_shape  # MUST OCCUR FOR VALIDATION

        transf_orig = self._apply_transforms([self.orig_img])
        for transf_name in transf_orig:
            save_path = path.join(
                trainer.logger.log_dir, f"original_image_{transf_name}.png"
            )

            transf_imgs, plotting_kwargs = transf_orig[transf_name]
            save_imgs = [img for img in transf_imgs]

            self._save_imgs(save_imgs, save_path, method="pil", **plotting_kwargs)

    def on_validation_end(self, trainer, pl_module: NeuralImplicitMLP):
        assert self.image_shape is not None, "Image shape must be set"

        state_fn = trainer.state.fn

        if self.disabled or self.val_idx % self.save_freq != 0:
            self.val_idx += 1
            return

        # get images
        out = pl_module.reconstruct(self.image_shape)
        imgs = list(out.values())
        labels = list(out.keys())

        # compute metrics
        scores = pl_module.scores
        loss = scores["val_loss"]

        # save image data
        if not self.best_only or (self.best_only and loss < self.min_loss):
            self.min_loss = loss
            for img, label in zip(imgs, labels):
                fname = label
                if not self.best_only:
                    fname += f"_e={trainer.current_epoch}_v={self.val_idx}"
                img_path = path.join(trainer.logger.log_dir, fname)
                np.save(img_path, np.squeeze(img))
            
        # transform for view and comparison
        transf_out = self._apply_transforms(imgs)
        del transf_out['raw']
        
        for transf_name in transf_out:
            transf_imgs, plotting_kwargs = transf_out[transf_name]

            # save images
            save_imgs = [img for img in transf_imgs]
            

            fname = (
                "_".join(
                    [
                        state_fn,
                        transf_name,
                        f"e={trainer.current_epoch}_v={self.val_idx}",
                    ]
                )
                + ".png"
            )
            img_path = path.join(trainer.logger.log_dir, fname)
            self._save_imgs(
                save_imgs,
                img_path,
                labels=labels,
                metrics=scores,
                method="plt",
                **plotting_kwargs,
            )

        self.val_idx += 1
