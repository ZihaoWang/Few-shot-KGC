# -*- coding: utf-8 -*-
import sys
import logging
from logging import info, warn, error
from sys import exit
import numpy as np
import torch as T
from collections import defaultdict

from util import *
from hyp_reptile import init_hyp, log_hyp
from data_reptile import load_raw_pkl, TrainDataset, EvalDataset
from reptile import REPTILE

def predict(tag, eval_dataset, model, hyp):
    info("\n------------------------------------------\nstart evaluating on {} dataset with {}".format(tag, hyp.model))
    model.eval()
    model.predict(eval_dataset)
    if hyp.is_training:
        model.train()
    info("\n------------------------------------------\nfinish evaluating on {} dataset with {}\n".format(tag, hyp.model))

def training(train_dataset, dev_dataset, test_dataset, model, hyp):
    info("\n------------------------------------------\nstart training")
    model.train()
    loss_items = defaultdict(float)

    for epoch in range(hyp.existing_epoch + 1, hyp.max_epoch + 1):
        loss = model.meta_train(train_dataset)
        for name, val in loss.items():
            loss_items[name] += val 
        if epoch % hyp.training_print_freq == 0:
            #info("{}/{} epoches".format(epoch, hyp.max_epoch))
            total_loss = 0.0
            msg = "{}/{} epoches, ".format(epoch, hyp.max_epoch)
            for name, val in loss_items.items():
                msg += "{} = {}, ".format(name, val)
                total_loss += val
            msg += "total_loss = {}".format(total_loss)
            info(msg)

            if dev_dataset is not None:
                predict("dev", dev_dataset, model, hyp)
            if test_dataset is not None:
                predict("test", test_dataset, model, hyp)
            loss_items = defaultdict(float)

        if not hyp.is_debugging and epoch % hyp.save_freq == 0:
            save_model(epoch, model, hyp)

def run(hyp):
    if hyp.seed is not None:
        fix_random_seeds(int(hyp.seed))

    if hyp.idx_device == -1 or not T.cuda.is_available():
        hyp.device = T.device("cpu")
    else:
        hyp.device = T.device("cuda:{}".format(hyp.idx_device))

    train_task, dev_task, test_task, ent_dscps, rel_dscps, i2r, w2i, i2w, rel2cand, e1rel_e2 = load_raw_pkl(hyp)

    hyp.dict_size = len(i2w)
    #hyp.char_size = len(i2c)
    hyp.idx_word_pad = w2i[hyp.WORD_PAD]

    train_dataset = TrainDataset("train", train_task, ent_dscps, rel_dscps, rel2cand, e1rel_e2, hyp)
    dev_dataset = EvalDataset("dev", dev_task, ent_dscps, rel_dscps, rel2cand, e1rel_e2, hyp)
    test_dataset = EvalDataset("test", test_task, ent_dscps, rel_dscps, rel2cand, e1rel_e2, hyp)

    model = eval(hyp.model)(hyp)

    if hyp.load_existing_model:
        model = load_model(model, hyp)

    if hyp.is_training:
        training(train_dataset, dev_dataset, test_dataset, model, hyp)
    else:
        predict("dev", dev_dataset, model, hyp)
        predict("test", test_dataset, model, hyp)

if __name__ == "__main__":
    np.set_printoptions(threshold = np.inf)
    hyp = init_hyp()

    logger = init_logger(hyp)
    log_hyp(hyp)
    run(hyp)
