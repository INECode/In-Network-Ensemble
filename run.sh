#!/bin/sh

#to run INE use the following commands(to run SWA, you can simply set ine_start equal to epochs, eg. set ine_start=250 in the VGG16 setting): 

#VGG16 
python3 train.py --lr_init=0.05 --wd=5e-4 --swa --swa_start=161 --swa_lr=0.01 --dataset=CIFAR100 --model=VGG16 --resume=./pretrained_weights/checkpoint-200.VGG16.CIFAR100.pt --epochs=250 --ine_start=220 --ine_lr_init=0.005 --ine_noise=0.01 --data_path=./data --dir=./ckpts
python3 train.py --lr_init=0.05 --wd=5e-4 --swa --swa_start=161 --swa_lr=0.01 --dataset=CIFAR10 --model=VGG16 --resume=./pretrained_weights/checkpoint-200.VGG16.CIFAR10.pt --epochs=250 --ine_start=220 --ine_lr_init=0.005 --ine_noise=0.01 --data_path=./data --dir=./ckpts

#PreResNet164
python3 train.py --lr_init=0.1 --wd=3e-4 --swa --swa_start=126 --swa_lr=0.05 --dataset=CIFAR100 --model=PreResNet164 --resume=./pretrained_weights/checkpoint-150.PreResNet164.CIFAR100.pt --epochs=185 --ine_start=170 --ine_lr_init=0.01 --ine_noise=0.001 --data_path=./data --dir=./ckpts
python3 train.py --lr_init=0.1 --wd=3e-4 --swa --swa_start=126 --swa_lr=0.01 --dataset=CIFAR10 --model=PreResNet164 --resume=./pretrained_weights/checkpoint-150.PreResNet164.CIFAR10.pt --epochs=185 --ine_start=170 --ine_lr_init=0.01 --ine_noise=0.01 --data_path=./data --dir=./ckpts

#WideResNet
python3 train.py --lr_init=0.1 --wd=5e-4 --swa --swa_start=161 --swa_lr=0.05 --dataset=CIFAR100 --model=WideResNet28x10 --resume=./pretrained_weights/checkpoint-200.WideResNet28x10.CIFAR100.pt --epochs=250 --ine_start=210 --ine_lr_init=0.01 --ine_noise=0.001 --data_path=./data --dir=./ckpts
python3 train.py --lr_init=0.1 --wd=5e-4 --swa --swa_start=161 --swa_lr=0.05 --dataset=CIFAR10 --model=WideResNet28x10 --resume=./pretrained_weights/checkpoint-200.WideResNet28x10.CIFAR10.pt --epochs=250 --ine_start=210 --ine_lr_init=0.01 --ine_noise=0.01 --data_path=./data --dir=./ckpts

