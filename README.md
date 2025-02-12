# Understanding & Improving Stochastic Weight Averaging (SWA)
This repository was made as part of the final project in the 'Introduction to Deep Learning' course in the department of Industrial Engineering and Management in Ben Gurion University of the Negev, Israel.

This repository is a fork of the original repository that contains a PyTorch implementation of the Stochastic Weight Averaging (SWA) training method for DNNs from the paper

[Averaging Weights Leads to Wider Optima and Better Generalization](https://arxiv.org/abs/1803.05407)

by Pavel Izmailov, Dmitrii Podoprikhin, Timur Garipov, Dmitry Vetrov and Andrew Gordon Wilson.



# Introduction

we added further options to train.py that would allow us to experiment with our proposed improvements to the paper.

--val_size controls the size of the validation set used for evaluation during training.

--type_of_average controls the type of averaging method to train and compare to the original SWA method ('weighted_moving_average' or 'exponential_smoothing').

--weight_from_data controls where te weights are taken from for the weighted_moving_average ( 'train' or 'validation').

--scale_weights controls whether to scale the weights using MinMax scaling.

--smoothing_factor is the alpha passed to  exponential_smoothing.

--eval_base_model controls whether to evaluate the pre SWA model during training.


the utils.py script contains our implementation of of the weighted_moving_average and exponential_smoothing functions.

visualizations of accuracy and loss during training for experiment 1 are implemented in exp_1_visuals.py .

further visualizations for experiment 1 are implemented in vizualization.py

visualizations of accuracy and loss during training for experiment 2 are implemented in exp_2_visuals.py .

the training of the Iterative SWA method from experiment 2 is implemented in exp_2_iswa_train.py .

the comperative SGD training for experiment 2 is implemented in exp_2_sgd_train.py .

The commands we used for running experiment 1:

initial pre training: 
```bash
python3 swa/train.py --dir=training_dir --dataset=CIFAR10 --data_path=data  --model=PreResNet14 --epochs=150 --lr_init=0.1 --wd=3e-4 --save_freq=1
```

different averaging techniques:

```bash
python3 swa/train.py --resume swa/checkpoints/checkpoint-150.pt --dir=training_dir --dataset=CIFAR10 --data_path=data  --model=PreResNet14 --epochs=200 --lr_init=0.1 --wd=3e-4  --save_freq=1 --swa --swa_start=151 --swa_lr=0.05 --type_of_average=exponential_smoothing --smoothing_factor=0.1
python3 swa/train.py --resume swa/checkpoints/checkpoint-150.pt --dir=training_dir --dataset=CIFAR10 --data_path=data  --model=PreResNet14 --epochs=200 --lr_init=0.1 --wd=3e-4  --save_freq=1 --swa --swa_start=151 --swa_lr=0.05 --weight_from_data=validation --type_of_weight=accuracy --type_of_average=weighted_moving_average --scale_weights=True
python3 swa/train.py --resume swa/checkpoints/checkpoint-150.pt --dir=training_dir --dataset=CIFAR10 --data_path=data  --model=PreResNet14 --epochs=200 --lr_init=0.1 --wd=3e-4  --save_freq=1 --swa --swa_start=151 --swa_lr=0.05 --weight_from_data=validation --type_of_weight=accuracy --type_of_average=weighted_moving_average --scale_weights=False
python3 swa/train.py --resume swa/checkpoints/checkpoint-150.pt --dir=training_dir --dataset=CIFAR10 --data_path=data  --model=PreResNet14 --epochs=200 --lr_init=0.1 --wd=3e-4  --save_freq=1 --swa --swa_start=151 --swa_lr=0.05 --weight_from_data=validation --type_of_weight=loss --type_of_average=weighted_moving_average --scale_weights=False
python3 swa/train.py --resume swa/checkpoints/checkpoint-150.pt --dir=training_dir --dataset=CIFAR10 --data_path=data  --model=PreResNet14 --epochs=200 --lr_init=0.1 --wd=3e-4  --save_freq=1 --swa --swa_start=151 --swa_lr=0.05 --weight_from_data=validation --type_of_weight=loss --type_of_average=weighted_moving_average --scale_weights=True
python3 swa/train.py --resume swa/checkpoints/checkpoint-150.pt --dir=training_dir --dataset=CIFAR10 --data_path=data  --model=PreResNet14 --epochs=200 --lr_init=0.1 --wd=3e-4  --save_freq=1 --swa --swa_start=151 --swa_lr=0.05 --type_of_average=exponential_smoothing  --smoothing_factor=0.9
python3 swa/train.py --resume swa/checkpoints/checkpoint-150.pt --dir=training_dir --dataset=CIFAR10 --data_path=data  --model=PreResNet14 --epochs=200 --lr_init=0.1 --wd=3e-4  --save_freq=1 --swa --swa_start=151 --swa_lr=0.05 --type_of_average=exponential_smoothing  --smoothing_factor=0.5
```

The commands we used for running experiment 2:

regular SGD:
```bash
python3 swa/experiment_2_sgd.py --sgd_duration=200  --dir=training_dir --model=PreResNet14 --dataset=CIFAR10 --data_path=data  --epochs=150 --lr_init=0.1 --wd=3e-4  --save_freq=1 --device=cuda
```
Iterative SWA:
```bash
python3 swa/exp_2_iswa_train.py --resume=swa/checkpoints/checkpoint-9.pt --dir=training_dir --model=PreResNet14 --dataset=CIFAR10 --data_path=data  --epochs=150 --lr_init=0.1 --wd=3e-4  --save_freq=1 --device=cuda
```


# From Original Repository
SWA is a simple DNN training method that can be used as a drop-in replacement for SGD with improved generalization, faster convergence, and essentially no overhead. The key idea of SWA is to average multiple samples produced by SGD with a modified learning rate schedule. We use a constant or cyclical learning rate schedule that causes SGD to _explore_ the set of points in the weight space corresponding to high-performing networks. We observe that SWA converges more quickly than SGD, and to wider optima that provide higher test accuracy. 

In this repo we implement the constant learning rate schedule that we found to be most practical on CIFAR datasets.

<p align="center">
  <img src="https://user-images.githubusercontent.com/14368801/37633888-89fdc05a-2bca-11e8-88aa-dd3661a44c3f.png" width=250>
  <img src="https://user-images.githubusercontent.com/14368801/37633885-89d809a0-2bca-11e8-8d57-3bd78734cea3.png" width=250>
  <img src="https://user-images.githubusercontent.com/14368801/37633887-89e93784-2bca-11e8-9d71-a385ea72ff7c.png" width=250>
</p>

Please cite our work if you find this approach useful in your research:
```bibtex
@article{izmailov2018averaging,
  title={Averaging Weights Leads to Wider Optima and Better Generalization},
  author={Izmailov, Pavel and Podoprikhin, Dmitrii and Garipov, Timur and Vetrov, Dmitry and Wilson, Andrew Gordon},
  journal={arXiv preprint arXiv:1803.05407},
  year={2018}
}
```


# Dependencies
* [PyTorch](http://pytorch.org/)
* [torchvision](https://github.com/pytorch/vision/)
* [tabulate](https://pypi.python.org/pypi/tabulate/)

# Usage

The code in this repository implements both SWA and conventional SGD training, with examples on the CIFAR-10 and CIFAR-100 datasets.

To run SWA use the following command:

```bash
python3 train.py --dir=<DIR> \
                 --dataset=<DATASET> \
                 --data_path=<PATH> \
                 --model=<MODEL> \
                 --epochs=<EPOCHS> \
                 --lr_init=<LR_INIT> \
                 --wd=<WD> \
                 --swa \
                 --swa_start=<SWA_START> \
                 --swa_lr=<SWA_LR>
```

Parameters:

* ```DIR``` &mdash; path to training directory where checkpoints will be stored
* ```DATASET``` &mdash; dataset name [CIFAR10/CIFAR100] (default: CIFAR10)
* ```PATH``` &mdash; path to the data directory
* ```MODEL``` &mdash; DNN model name:
    - VGG16/VGG16BN/VGG19/VGG19BN
    - PreResNet110/PreResNet164
    - WideResNet28x10
* ```EPOCHS``` &mdash; number of training epochs (default: 200)
* ```LR_INIT``` &mdash; initial learning rate (default: 0.1)
* ```WD``` &mdash; weight decay (default: 1e-4)
* ```SWA_START``` &mdash; the number of epoch after which SWA will start to average models (default: 161)
* ```SWA_LR``` &mdash; SWA learning rate (default: 0.05)


To run conventional SGD training use the following command:
```bash
python3 train.py --dir=<DIR> \
                 --dataset=<DATASET> \
                 --data_path=<PATH> \
                 --model=<MODEL> \
                 --epochs=<EPOCHS> \
                 --lr_init=<LR_INIT> \
                 --wd=<WD> 
```

## Examples

To reproduce the results from the paper run (we use same parameters for both CIFAR-10 and CIFAR-100 except for PreResNet):
```bash
#VGG16
python3 train.py --dir=<DIR> --dataset=CIFAR100 --data_path=<PATH> --model=VGG16 --epochs=200 --lr_init=0.05 --wd=5e-4 # SGD
python3 train.py --dir=<DIR> --dataset=CIFAR100 --data_path=<PATH> --model=VGG16 --epochs=300 --lr_init=0.05 --wd=5e-4 --swa --swa_start=161 --swa_lr=0.01 # SWA 1.5 Budgets

#PreResNet
python3 train.py --dir=<DIR> --dataset=CIFAR100 --data_path=<PATH>  --model=[PreResNet110 or PreResNet164] --epochs=150  --lr_init=0.1 --wd=3e-4 # SGD
#CIFAR100
python3 train.py --dir=<DIR> --dataset=CIFAR100 --data_path=<PATH>  --model=[PreResNet110 or PreResNet164] --epochs=225 --lr_init=0.1 --wd=3e-4 --swa --swa_start=126 --swa_lr=0.05 # SWA 1.5 Budgets
#CIFAR10
python3 train.py --dir=<DIR> --dataset=CIFAR10 --data_path=<PATH>  --model=[PreResNet110 or PreResNet164] --epochs=225 --lr_init=0.1 --wd=3e-4 --swa --swa_start=126 --swa_lr=0.01 # SWA 1.5 Budgets

#WideResNet28x10 
python3 train.py --dir=<DIR> --dataset=CIFAR100 --data_path=<PATH> --model=WideResNet28x10 --epochs=200 --lr_init=0.1 --wd=5e-4 # SGD
python3 train.py --dir=<DIR> --dataset=CIFAR100 --data_path=<PATH> --model=WideResNet28x10 --epochs=300 --lr_init=0.1 --wd=5e-4 --swa --swa_start=161 --swa_lr=0.05 # SWA 1.5 Budgets
```

# Results

## CIFAR-100

Test accuracy (%) of SGD and SWA on CIFAR-100 for different training budgets. For each model the _Budget_ is defined as the number of epochs required to train the model with the conventional SGD procedure.

| DNN (Budget)              |  SGD         | SWA 1 Budget | SWA 1.25 Budgets | SWA 1.5 Budgets |
| ------------------------- |:------------:|:------------:|:----------------:|:---------------:|
| VGG16 (200)               | 72.55 ± 0.10 | 73.91 ± 0.12 | 74.17 ± 0.15     | 74.27 ± 0.25    |
| PreResNet110 (150)        | 76.77 ± 0.38 | 78.75 ± 0.16 | 78.91 ± 0.29     | 79.10 ± 0.21    |
| PreResNet164 (150)        | 78.49 ± 0.36 | 79.77 ± 0.17 | 80.18 ± 0.23     | 80.35 ± 0.16    |
| WideResNet28x10 (200)     | 80.82 ± 0.23 | 81.46 ± 0.23 | 81.91 ± 0.27     | 82.15 ± 0.27    |

Below we show the convergence plot for SWA and SGD with PreResNet164 on CIFAR-100 and the corresponding learning rates. The dashed line illustrates the accuracy of individual models averaged by SWA.

<p align="center">
<img src="https://user-images.githubusercontent.com/14368801/37633527-226bb2d6-2bc9-11e8-9be6-097c0dfe64ab.png" width=500>
</p>


## CIFAR-10

Test accuracy (%) of SGD and SWA on CIFAR-10 for different training budgets.

| DNN (Budget)              |  SGD         | SWA 1 Budget | SWA 1.25 Budgets | SWA 1.5 Budgets |
| ------------------------- |:------------:|:------------:|:----------------:|:---------------:|
| VGG16 (200)               | 93.25 ± 0.16 | 93.59 ± 0.16 | 93.70 ± 0.22     | 93.64 ± 0.18    |
| PreResNet110 (150)        | 95.03 ± 0.05 | 95.51 ± 0.10 | 95.65 ± 0.03     | 95.82 ± 0.03    |
| PreResNet164 (150)        | 95.28 ± 0.10 | 95.56 ± 0.11 | 95.77 ± 0.04     | 95.83 ± 0.03    |
| WideResNet28x10 (200)     | 96.18 ± 0.11 | 96.45 ± 0.11 | 96.64 ± 0.08     | 96.79 ± 0.05    |
 
# Other Implementations

* Chainer Implementation: [github.com/chainer/models/tree/master/swa](https://github.com/chainer/models/tree/master/swa)
* Keras/Tensorflow-Keras Implementation: [github.com/simon-larsson/keras-swa](https://github.com/simon-larsson/keras-swa)
* PyTorch Contrib: [github.com/pytorch/contrib](https://github.com/pytorch/contrib)
 
# References
 
 Provided model implementations were adapted from
 * VGG: [github.com/pytorch/vision/](https://github.com/pytorch/vision/)
 * PreResNet: [github.com/bearpaw/pytorch-classification](https://github.com/bearpaw/pytorch-classification)
 * WideResNet: [github.com/meliketoy/wide-resnet.pytorch](https://github.com/meliketoy/wide-resnet.pytorch)

