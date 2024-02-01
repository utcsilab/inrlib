# inrlib

Zach Stoebner

This library is intended for anyone seeking to train an implicit neural representation (INR) on discretized signal measurements. It contains basic implementations of a neural implicit MLP, losses, constraints, and logging for complex-valued data. Refer to the [demo](/demo.ipynb) to get started.  

## Setup

Create a dedicated conda environment called `inr`:

```bash
conda env create -f environment.yml
```

This environment assumes Linux and CUDA>=12.1. 

If the file doesn't work out of the box, the key requirements are: 
- python>=3.10 
- [pytorch](https://pytorch.org/get-started/locally/)
- [lightning](https://lightning.ai/docs/pytorch/stable/starter/installation.html)

For the [demo](/demo.ipynb), additionally run: 

```bash
pip install gdown phantominator omegaconf
```

## Usage

```bash
sh run_from_config.sh <STAGE=[train, val, test, pred]> <CONFIG=path/to/config> <GPUID=int>
```

Refer to the [demo](/inrlib/demo.ipynb) for example implementation and how to extend to new models, losses, constraints, etc.  

## References

```{bibliography}
@article{sitzmann2020implicit,
  title={Implicit neural representations with periodic activation functions},
  author={Sitzmann, Vincent and Martel, Julien and Bergman, Alexander and Lindell, David and Wetzstein, Gordon},
  journal={Advances in neural information processing systems},
  volume={33},
  pages={7462--7473},
  year={2020}
}

@article{tancik2020fourier,
  title={Fourier features let networks learn high frequency functions in low dimensional domains},
  author={Tancik, Matthew and Srinivasan, Pratul and Mildenhall, Ben and Fridovich-Keil, Sara and Raghavan, Nithin and Singhal, Utkarsh and Ramamoorthi, Ravi and Barron, Jonathan and Ng, Ren},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  pages={7537--7547},
  year={2020}
}

@article{mildenhall2021nerf,
  title={Nerf: Representing scenes as neural radiance fields for view synthesis},
  author={Mildenhall, Ben and Srinivasan, Pratul P and Tancik, Matthew and Barron, Jonathan T and Ramamoorthi, Ravi and Ng, Ren},
  journal={Communications of the ACM},
  volume={65},
  number={1},
  pages={99--106},
  year={2021},
  publisher={ACM New York, NY, USA}
}
```
