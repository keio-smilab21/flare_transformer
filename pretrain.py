import sys
import json
import argparse

from torchinfo import summary

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from src.Dataloader import PretrainDataloader
from src.eval_utils import calc_score
from src.model import CNNModel
from src.BalancedBatchSampler import BalancedBatchSampler


def gmgs_loss_function(y_pred, y_true, score_matrix):
    """[GMGS loss function]

    Args:
        y_pred ([type]): [description]
        y_true ([type]): [description]
        score_matrix ([type]): [description]

    Returns:
        [type]: [description]
    """
    score_matrix = torch.tensor(score_matrix).cuda()
    y_truel = torch.argmax(y_true, dim=1)
    weight = score_matrix[y_truel]
    py = torch.log(y_pred)
    output = torch.mul(y_true, py)
    output[torch.isnan(output)] = 0
    output = torch.mul(output, weight)
    output = torch.mean(output)

    return -output


def label_smoothing(y_true, epsilon):
    """[summary]

    Args:
        y_true ([type]): [description]
        epsilon ([type]): [description]

    Returns:
        [type]: [description]
    """
    x = y_true + epsilon
    x = x / (1+epsilon*4)
    return x


def train_epoch(model, train_dl):
    """Return train loss and score for train set"""
    model.train()
    predictions = []
    observations = []
    train_loss = 0
    n = 0
    for _, (x, y) in enumerate(tqdm(train_dl)):
        optimizer.zero_grad()
        output = model(x.cuda().to(torch.float))
        bce_loss = criterion(output, torch.max(y, 1)[
                             1].cuda().to(torch.long))
        loss = bce_loss
        # gmgs_loss = \
        #     gmgs_criterion(output,
        #                    y.cuda().to(torch.float),
        #                    params["dataset"]["GMGS_score_matrix"])
        # loss = gmgs_loss
        loss.backward()
        optimizer.step()
        train_loss += (loss.detach().cpu().item() * x.shape[0])
        n += x.shape[0]

        for pred, o in zip(output.cpu().detach().numpy().tolist(),
                           y.detach().numpy().tolist()):
            predictions.append(pred)
            observations.append(np.argmax(o))

    score = calc_score(predictions, observations)
    score = calc_test_score(score, "train")
    return score, train_loss/n


def eval_epoch(model, validation_dl):
    """Return val loss and score for val set"""
    model.eval()
    predictions = []
    observations = []
    valid_loss = 0
    n = 0
    with torch.no_grad():
        for _, (x, y) in enumerate(tqdm(validation_dl)):
            output = model(x.cuda().to(torch.float))
            bce_loss = criterion(output, torch.max(y, 1)[
                                 1].cuda().to(torch.long))
            loss = bce_loss
            # gmgs_loss = \
            #     gmgs_criterion(output,
            #                    y.cuda().to(torch.float),
            #                    params["dataset"]["GMGS_score_matrix"])
            # loss = gmgs_loss
            valid_loss += (loss.detach().cpu().item() * x.shape[0])
            n += x.shape[0]
            for pred, o in zip(output.cpu().numpy().tolist(),
                               y.numpy().tolist()):
                predictions.append(pred)
                observations.append(np.argmax(o))
        score = calc_score(predictions, observations)
        score = calc_test_score(score, "valid")
    return score, valid_loss/n


def test_epoch(model, test_dl):
    """Return test loss and score for test set"""
    model.eval()
    predictions = []
    observations = []
    test_loss = 0
    n = 0
    with torch.no_grad():
        for _, (x, y) in enumerate(tqdm(test_dl)):
            output = model(x.cuda().to(torch.float))
            bce_loss = criterion(output, torch.max(y, 1)[
                                 1].cuda().to(torch.long))
            loss = bce_loss
            # gmgs_loss = \
            #     gmgs_criterion(output,
            #                    y.cuda().to(torch.float),
            #                    params["dataset"]["GMGS_score_matrix"])
            # loss = gmgs_loss
            test_loss += (loss.detach().cpu().item() * x.shape[0])
            n += x.shape[0]
            for pred, o in zip(output.cpu().numpy().tolist(), y.tolist()):
                predictions.append(pred)
                observations.append(np.argmax(o))
        score = calc_score(predictions, observations)
        score = calc_test_score(score, "test")
    return score, test_loss/n


def calc_test_score(score, label):
    """Return dict with key of label"""
    test_score = {}
    for k, v in score.items():
        test_score[label+"_"+k] = v
    return test_score


seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


def p(x):
    print(x, x.shape)
    sys.exit()


if __name__ == "__main__":
    # fix seed value
    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--params', default='params/pretrain_params2017.json')
    parser.add_argument('--project_name', default='flare_transformer_pretrain')
    args = parser.parse_args()
    wandb_flag = args.wandb

    # read params/params.json
    params = json.loads(open(args.params).read())

    # Initialize W&B
    if wandb_flag is True:
        wandb.init(project=args.project_name, name=params["wandb_name"])

    print("==========================================")
    print(params)
    print("==========================================")

    # Initialize Dataset
    train_dataset = PretrainDataloader("train", params["dataset"])

    if params["dataset"]["mean"] != 0:
        mean = params["dataset"]["mean"]
        std = params["dataset"]["std"]
    else:
        mean, std = train_dataset.calc_mean()
        print(mean, std)
    train_dataset.set_mean(mean, std)
    validation_dataset = PretrainDataloader("valid", params["dataset"])
    validation_dataset.set_mean(mean, std)
    test_dataset = PretrainDataloader("test", params["dataset"])
    test_dataset.set_mean(mean, std)
    print("Batch Sampling")
    # train_dl = DataLoader(train_dataset, batch_size=params["bs"], shuffle=True)
    train_dl = DataLoader(train_dataset, batch_sampler=BalancedBatchSampler(
        train_dataset, params["output_channel"], params["bs"]//params["output_channel"]))
    validation_dl = DataLoader(validation_dataset,
                               batch_size=params["bs"], shuffle=False)
    test_dl = DataLoader(test_dataset, batch_size=params["bs"], shuffle=False)

    # Initialize Loss Function
    criterion = nn.CrossEntropyLoss().cuda()
    gmgs_criterion = gmgs_loss_function

    # model = MagnetogramFeatureExtractor(k=16, r=16, pretrain=True).to("cuda")
    model = CNNModel(
        output_channel=params["output_channel"], pretrain=True).to("cuda")
    pretrain_main_metric = "GMGS"

    summary(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])

    # Start Training
    best_score = {}
    best_score["valid_"+params["main_metric"]] = -10
    best_epoch = 0
    for e, epoch in enumerate(range(params["epochs"])):
        print("====== Epoch ", e, " ======")
        train_score, train_loss = train_epoch(model, train_dl)
        valid_score, valid_loss = eval_epoch(model, validation_dl)
        # test_score, test_loss = test_epoch(model, test_dl)
        test_score, test_loss = valid_score, valid_loss  # skip version

        if best_score["valid_"+params["main_metric"]] < \
                valid_score["valid_"+params["main_metric"]]:
            torch.save(model.state_dict(), params["save_model_path"])
            best_score = valid_score
            best_epoch = e

        log = {'epoch': epoch, 'train_loss': np.mean(train_loss),
               'valid_loss': np.mean(valid_loss),
               'test_loss': np.mean(test_loss)}
        log.update(train_score)
        log.update(valid_score)
        log.update(test_score)

        if wandb_flag is True:
            wandb.log(log)

        print("Epoch {}: Train loss:{:.4f}  Valid loss:{:.4f}".format(
              e, train_loss, valid_loss), test_score)

    # Output Test Score
    print("========== TEST ===========")
    model.load_state_dict(torch.load(params["save_model_path"]))
    test_score, _ = test_epoch(model, test_dl)
    print("epoch : ", best_epoch, test_score)
    if wandb_flag is True:
        wandb.log(calc_test_score(test_score, "final"))
