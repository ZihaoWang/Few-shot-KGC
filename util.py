# -*- coding: utf-8 -*-
from sys import exit
import logging
from logging import info, warn, error
import random
import numpy as np
import torch as T

def init_logger(hyp):
    log_path = hyp.log_root + hyp.prefix
    if hyp.is_training:
        log_path += "_train_"
        if hyp.postfix != "":
            log_path += hyp.postfix + "_"
        log_path += hyp.existing_timestamp
    else:
        log_path += "_predict_"
        log_path += hyp.existing_timestamp
        log_path += "_epoch" + str(hyp.existing_epoch)
    log_path += ".log"

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(levelname)s: - %(message)s", datefmt = "%Y-%m-%d %H:%M:%S")
    if not hyp.is_debugging:
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger

def fix_random_seeds(seed):
    warn("random seed is fixed: {}\n".format(seed))
    random.seed(seed)
    np.random.seed(seed)
    T.manual_seed(seed)
    if T.cuda.is_available():
        T.cuda.manual_seed_all(seed)

def save_model(epoch, model, hyp):
    path = "{}{}_{}_epoch{}.pt".format(hyp.model_root, hyp.prefix, hyp.existing_timestamp, epoch)
    checkpoint = {
            "epoch" : epoch,
            "idx_device" : hyp.idx_device,
            }
    state_dict = model.get_state_dict()
    checkpoint = {**checkpoint, **state_dict}
    T.save(checkpoint, path)
    info("finish saving model to {}".format(path))

def load_model(model, hyp):
    path = "{}{}_{}_epoch{}.pt".format(hyp.model_root, hyp.prefix, hyp.existing_timestamp, hyp.existing_epoch)
    checkpoint = T.load(path)
    assert hyp.existing_epoch == checkpoint["epoch"]
    model.set_state_dict(checkpoint)
    info("finish loading model from {}".format(path))
    return model

def build_glove_emb(w2i, hyp):
    glove_emb = pickle.load(open(hyp.glove_pkl_path, "rb"))

    if glove_emb.shape[0] != len(w2i) or glove_emb.shape[1] != hyp.dim_emb:
        info("rebuild glove word embedding")
        glove_path = "{}glove.6B.{}d.txt".format(hyp.glove_root, hyp.dim_emb)
        glove_emb = T.empty((len(w2i), hyp.dim_emb))
        with open(glove_path, "r") as f_src:
            pass
    else:
        return glove_emb

def extract_top_level_dict(current_dict):
    """
    Builds a graph dictionary from the passed depth_keys, value pair. Useful for dynamically passing external params
    :param depth_keys: A list of strings making up the name of a variable. Used to make a graph for that params tree.
    :param value: Param value
    :param key_exists: If none then assume new dict, else load existing dict and add new key->value pairs to it.
    :return: A dictionary graph of the params already added to the graph.
    """
    output_dict = {}
    for key in current_dict.keys():
        name = key.replace("layer_dict.", "")
        top_level = name.split(".")[0]
        sub_level = ".".join(name.split(".")[1:])

        if top_level not in output_dict:
            if sub_level == "":
                output_dict[top_level] = current_dict[key]
            else:
                output_dict[top_level] = {sub_level: current_dict[key]}
        else:
            new_item = {key: value for key, value in output_dict[top_level].items()}
            new_item[sub_level] = current_dict[key]
            output_dict[top_level] = new_item

    #print(current_dict.keys(), output_dict.keys())
    return output_dict



