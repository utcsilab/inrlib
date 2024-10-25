import torch
from lightning.pytorch.cli import LightningCLI 
from inrlib.inrlib.callbacks.ConfigCallback import ConfigCallback

def run_from_config():
    torch.set_float32_matmul_precision('medium') # default is 'highest'
    cli = LightningCLI(save_config_callback=ConfigCallback)
    # cli = LightningCLI(save_config_kwargs={'overwrite': True})

if __name__ == "__main__":
    run_from_config()