#!/bin/sh
# python train.py --params params/params_without_pretrain_2017.json --wandb
# python train.py --params params/params_without_pretrain_2016.json --wandb
# python train.py --params params/params_without_pretrain_2015.json --wandb
# python train.py --params params/params_without_pretrain_2014.json --wandb

python train.py --params params/params_k=2.json --wandb
python train.py --params params/params_k=4.json --wandb
python train.py --params params/params_k=8.json --wandb
# python train.py --params params/params_k=12.json --wandb
# python train.py --params params/params_k=24.json --wandb
