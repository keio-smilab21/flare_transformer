#!/bin/sh
python train.py --params params/params_without_pretrain_2017.json --wandb
python train.py --params params/params_without_pretrain_2016.json --wandb
python train.py --params params/params_without_pretrain_2015.json --wandb
python train.py --params params/params_without_pretrain_2014.json --wandb
