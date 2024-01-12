# Conformal Credal Self-Supervised Learning

This repository contains an implementation of _Conformal Credal Self-Supervised Learning_ (C^2S^2L). Please cite this work as follows:

```
@inproceedings{DBLP:conf/copa/LienenDH23,
  author       = {Julian Lienen and
                  Caglar Demir and
                  Eyke H{\"{u}}llermeier},
  editor       = {Harris Papadopoulos and
                  Khuong An Nguyen and
                  Henrik Bostr{\"{o}}m and
                  Lars Carlsson},
  title        = {Conformal Credal Self-Supervised Learning},
  booktitle    = {Conformal and Probabilistic Prediction with Applications, 13-15 September
                  2023, Limassol, Cyprus},
  series       = {Proceedings of Machine Learning Research},
  volume       = {204},
  pages        = {214--233},
  publisher    = {{PMLR}},
  year         = {2023}
}
```

The papers' **appendix** for COPA 2023 can be found separately here: [CCSSL_Appendix.pdf](CCSSL_Appendix.pdf)

## Foreword

Our implementation re-uses parts of an unofficial PyTorch implementation of FixMatch [1], namely from [this repository](https://github.com/kekmodel/FixMatch-pytorch) (MIT License). Thus, we overtake large parts of their code, for which we gratefully thank the authors for their work. While some parts were added from scratch (e.g., the conformal prediction implementation), we also augmented existing code (e.g., the data loaders by adding a calibration split).

## Requirements

To install all required packages, you need to run
```
pip install -r requirements.txt
```

The code has been tested using Python 3.6, 3.8 and 3.9 on Ubuntu 18.* and Ubuntu 20.* systems. We trained our models on machines with Nvidia GPUs (we tested CUDA 10.1, 11.1 and 11.6). We recommend to use [Python virtual environments](https://docs.python.org/3/tutorial/venv.html) to get a clean Python environment for the execution without any dependency problems.

To track the results, we justed the service [wandb.ai](wandb.ai). We can recommend this service, which allows for an easy yet scalable tracking of all experiments. The project name can be set via the `wandb_project` parameter.

## Datasets

All datasets are downloaded automatically. To set the location, just specify the parameter `dataset_dir`.

## Training and Evaluation

For the training and evaluation (e.g., of C^2S^2L on CIFAR-10 with 250 labels and seed 2), you have to call the following function:

```
CUDA_VISIBLE_DEVICES=<the numeric ID(s) of your CUDA device(s)> python train_cp.py --dataset=cifar10  --num-labeled 250 --seed
```

`--help` allows for printing out all parameter options. All results presented in the paper were computed based the training scripts `train_*.py`. The conformal prediction implementation can be found in `conf_pred.py`.

Let us further note that we recommend to use [mixed precision](https://github.com/NVIDIA/apex), as it can drastically speed up the execution on newer Nvidia devices. We could notify a speedup also when used for this project. Additional, we support CPU-only execution by the parameterization `--gpu-id -1`.

## License

Our code uses the Apache 2.0 License, which we attached as `LICENSE` file in this repository. 

Feel free to re-use our code. We would be happy to see our ideas put into practice.

## References

[1]: Sohn, K., _et al_. FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence. NeurIPS, 2020.
