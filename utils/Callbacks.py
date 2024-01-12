import os
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.cli import LightningCLI, SaveConfigCallback


class ConfigCallback(SaveConfigCallback):
    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
      if self.already_saved:
          return

      assert trainer.log_dir is not None
      log_dir = os.path.join(
          trainer.log_dir, trainer.logger.name, 'version_'+str(trainer.logger.version)
      )  # this broadcasts the directory
      if trainer.is_global_zero and not os.path.exists(log_dir):
        os.makedirs(log_dir)
      config_path = os.path.join(log_dir, self.config_filename)
      conf = str(self.config.as_dict())
      conf = OmegaConf.create(conf)
      with open(config_path, 'w') as f:
        OmegaConf.save(config=conf, f=f.name)