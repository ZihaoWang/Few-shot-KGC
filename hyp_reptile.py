from sys import exit
import argparse
import logging
from logging import info
import time
from dataset_config import get_dataset_config

def init_hyp(dataset = None):
    parser = argparse.ArgumentParser()

    parser.add_argument("--UNK", default = "UNK", type = str)
    parser.add_argument("--NUM", default = "$", type = str) # for both character and word embeddings
    parser.add_argument("--WORD_PAD", default = "PAD", type = str)

    if dataset is None:
        parser.add_argument("--dataset", default = "", type = str)

    parser.add_argument("--data_root", default = "./data/", type = str)
    parser.add_argument("--model_root", default = "./model/", type = str)
    parser.add_argument("--log_root", default = "./log/", type = str)
    parser.add_argument("--raw_glove_root", default = "/gds/zhwang/zhwang/data/general_data/glove/", type = str)
    parser.add_argument("--glove_pkl_path", default = "./data/glove_emb.pkl", type = str)

    parser.add_argument("--postfix", default = "", type = str)
    parser.add_argument("--idx_device", default = -1, type = int) # -1: CPU, 0, 1, ...: GPU
    parser.add_argument("--model", default = "reptile", type = str)

    parser.add_argument("--debug", dest = "is_debugging", action = "store_true")
    parser.set_defaults(is_debugging = False)
    parser.add_argument("--train", dest = "is_training", action = "store_true")
    parser.add_argument("--test", dest = "is_training", action = "store_false")
    parser.set_defaults(is_training = None)
    parser.add_argument("--max_epoch", default = 1000, type = int)
    parser.add_argument("--training_print_freq", default = 100, type = int)
    parser.add_argument("--save_freq", default = 100000, type = int)

    parser.add_argument("--existing_epoch", default = 0, type = int)
    parser.add_argument("--existing_timestamp", default = "", type = str)

    parser.add_argument("--cand_bucket_size", default = 100, type = int)

    parser.add_argument("--meta_batch_size", default = 8, type = int)
    parser.add_argument("--num_shot", default = 1, type = int)
    parser.add_argument("--num_train_inner_iter", default = 5, type = int)
    parser.add_argument("--num_test_inner_iter", default = 5, type = int)

    parser.add_argument("--inner_lr", default = 1e-3, type = float)
    parser.add_argument("--meta_lr", default = 1e-3, type = float)

    parser.add_argument("--dim_emb", default = 100, type = int)

    parser.add_argument("--encoder_num_cnn_layer", default = 3, type = int)
    parser.add_argument("--encoder_dim_conv_filter", default = 3, type = int)
    parser.add_argument("--encoder_num_conv_filter", nargs = "+", default = [100, 100, 100], type = int)
    parser.add_argument("--encoder_normalization", default = "instance_norm", type = str)
    parser.add_argument("--encoder_num_memory", default = 128, type = int)
    parser.add_argument("--no_encoder_self_atten", dest = "encoder_self_atten", action = "store_false")
    parser.set_defaults(encoder_self_atten = True)
    parser.add_argument("--encoder_num_head", nargs = "+", default = [5, 5], type = int)
    parser.add_argument("--encoder_act_func", default = "tanh", type = str)

    parser.add_argument("--num_aug", default = 0, type = int)
    parser.add_argument("--no_vae", dest = "use_vae", action = "store_false")
    parser.set_defaults(use_vae = True)
    parser.add_argument("--vae_num_conv_filter", default = 100, type = int)
    parser.add_argument("--vae_dim_hidden", default = 100, type = int)

    parser.add_argument("--vae_prior_nn_num_layer", default = 2, type = int)
    parser.add_argument("--vae_prior_nn_dim_hidden", nargs = "+", default = [100, 100], type = int)

    parser.add_argument("--vae_dim_latent", default = 50, type = int)
    parser.add_argument("--no_vae_prior", dest = "vae_use_prior", action = "store_false")
    parser.set_defaults(vae_use_prior = True)
    parser.add_argument("--prior_sigma_m", default = 1e4, type = float)
    parser.add_argument("--prior_sigma_s", default = 1e-4, type = float)
    parser.add_argument("--vae_lambda_kld", default = 1.0, type = float)
    parser.add_argument("--vae_lambda_reg", default = 1.0, type = float)
    parser.add_argument("--vae_normalization", default = "instance_norm", type = str)
    parser.add_argument("--vae_act_func", default = "tanh", type = str)

    parser.add_argument("--cnn_encoder", dest = "memory", action = "store_false")
    parser.set_defaults(memory = True)

    parser.add_argument("--sim_func", default = 'TransE', type = str)

    parser.add_argument("--seed", default = 1550148948, type = int)
    hyp = parser.parse_args()

    if dataset is None: # main()
        dataset_config = get_dataset_config(hyp.dataset)
    else: # prepare_data()
        dataset_config = get_dataset_config(dataset)
    for k, v in dataset_config.items():
        if k == "num_aug":
            if not hyp.use_vae or hyp.num_aug != 0:
                continue
        setattr(hyp, k, v)

    if dataset is None:
        if hyp.is_training is None:
            print("--train or --test must be specified")
            exit(-1)
        elif hyp.is_training:
            if hyp.existing_timestamp != "" and hyp.existing_epoch != 0:
                hyp.load_existing_model = True
            else:
                hyp.load_existing_model = False
                new_timestamp = str(time.time()).split(".")[0]
                hyp.existing_timestamp = new_timestamp
        else:
            hyp.load_existing_model = True
            if hyp.existing_timestamp == "":
                print("existing_timestamp must be specified")
                exit(-1)
            if hyp.existing_epoch == 0:
                print("existing_epoch must be specified")
                exit(-1)

        if dataset is None and hyp.seed == 0:
            hyp.seed = hyp.existing_timestamp

        hyp.model = hyp.model.upper()
        hyp.prefix = hyp.model

    return hyp

def log_hyp(hyp):
    info("----------------------------\n\nhyperparameters:\n")
    for k, v in vars(hyp).items():
        info("{} = {}".format(k, v))
    info("------------------------------------------\n")

if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)

    hyp = init_hyp()
    hyp.aa = 1 # add extra hyper-parameters after parsing
    log_hyp(hyp)
