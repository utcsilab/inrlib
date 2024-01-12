# inrlib

Zach Stoebner

This library is intended for anyone seeking to train an implicit neural representation (INR) on a discretized signal measurements. It contains basic implementations of the neural implicit MLP, losses, constraints, and logging for complex-valued data. Refer to the [demo](/inrlib/demo.ipynb) to get started.  

## Usage

```bash
sh run_from_config.sh <STAGE=[train, val, test, pred]> <CONFIG=path/to/config> <GPUID=int>
```

Refer to the [demo](/inrlib/demo.ipynb) for example implementation and how to extend to new models, losses, constraints, etc.  

## References

TODO
