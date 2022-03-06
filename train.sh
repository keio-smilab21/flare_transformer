#!/bin/sh
python train.py --params params/params_2014.json --wandb
python train.py --params params/params_2017.json --wandb
python train.py --params params/params_2016.json --wandb
python train.py --params params/params_2015.json --wandb