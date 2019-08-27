from sys import exit
import pickle
import logging
from logging import info, error
from random import sample, choices
import torch as T

from hyp_reptile import *

def load_raw_pkl(hyp):
    pkl_path = hyp.raw_pkl_path + ".debug" if hyp.is_debugging else hyp.raw_pkl_path
    info("start loading raw pickle from {}".format(pkl_path))
    all_data = pickle.load(open(pkl_path, "rb"))
    train_task, dev_task, test_task, ent_dscps, rel_dscps, i2r, w2i, i2w, rel_cand, e1rel_e2 = all_data

    info("#task in each split: #train = {}, #dev = {}, #test = {}".format(
        len(train_task), len(dev_task), len(test_task)))
    info("#ent = {}, #rel = {}, #word = {}".format(
        len(ent_dscps), len(i2r), len(w2i)))

    return all_data

class BaseDataset(object):
    def __init__(self, dataset, ent_dscps, rel_dscps, rel2cand, e1rel_e2, hyp,
            task_type, num_shot, num_inner_iter):
        self.dataset = dataset
        self.ent_dscps = ent_dscps
        self.rel_dscps = rel_dscps
        self.rel2cand = rel2cand
        self.e1rel_e2 = e1rel_e2
        self.hyp = hyp

        self.idx_word_pad = hyp.idx_word_pad
        self.device = hyp.device
        self.task_type = task_type
        self.num_shot = hyp.num_shot
        self.num_inner_iter = num_inner_iter

        self.dataset_rels = list(self.dataset.keys())
        self.num_all_data = 0
        for rel in self.dataset_rels:
            self.num_all_data += len(self.dataset[rel])

        self.task_size = self.num_shot if task_type == "train" else 1
        self.idx_sup_inner = 0
        self.sup_task = None
        self.idx_sup = list(range(self.task_size))
        self.update_len = lambda new_len, exist_len : new_len if new_len > exist_len else exist_len

    def sample_sup_noise(self, ent1, rel, ent2):
        existing_ent2 = self.e1rel_e2[(ent1, rel)]
        while 1:
            noise = sample(self.rel2cand[rel], 1)[0]
            if noise not in existing_ent2 and noise != ent2:
                break

        return noise

    def init_sup_tensors(self, task_src):
        self.sup_task = []
        for i, (ent1, ent2, entn, rel) in enumerate(task_src):
            dscp1, dscp2, dscpn, dscpr = self.ent_dscps[ent1], self.ent_dscps[ent2], self.ent_dscps[entn], self.rel_dscps[rel]
            self.sup_task.append((dscp1, dscp2, dscpn, dscpr))

    def get_inner_data(self):
        batch = [e.to(self.device) for e in self.sup_task[self.idx_sup_inner]]

        self.idx_sup_inner += 1
        if self.idx_sup_inner >= self.num_shot:
            self.sup_task = None

        return batch

class TrainDataset(BaseDataset):
    def __init__(self, task_type, dataset, ent_dscps, rel_dscps, rel2cand, e1rel_e2, hyp):
        BaseDataset.__init__(self, dataset, ent_dscps, rel_dscps, rel2cand, e1rel_e2, hyp,
                task_type, hyp.num_shot, hyp.num_train_inner_iter)
        info("finish initializing {} dataset".format(task_type))

    def get_sup_inner_data(self):
        if self.sup_task is None:
            self.__init_sup_task()
        return self.get_inner_data()

    def __init_sup_task(self):
        self.idx_sup_inner = 0
        task_src = []

        rel = sample(self.dataset_rels, 1)[0]
        rel_duos = self.dataset[rel]
        if self.num_shot > len(rel_duos):
            task_duo = [choice(rel_duos) for i in range(self.num_shot)]
        else:
            task_duo = sample(rel_duos, self.num_shot)
        for ent1, ent2 in task_duo:
            noise = self.sample_sup_noise(ent1, rel, ent2)
            task_src.append((ent1, ent2, noise, rel))

        self.init_sup_tensors(task_src)

class EvalDataset(BaseDataset):
    def __init__(self, task_type, dataset, ent_dscps, rel_dscps, rel2cand, e1rel_e2, hyp):
        BaseDataset.__init__(self, dataset, ent_dscps, rel_dscps, rel2cand, e1rel_e2, hyp,
                task_type, hyp.num_shot, hyp.num_test_inner_iter)
        self.cand_bucket_size = hyp.cand_bucket_size

        self.__reset()
        info("finish initializing {} dataset".format(task_type))

    def next_rel(self):
        self.idx_ent = self.num_shot
        if self.idx_rel >= len(self.dataset_rels):
            self.__reset()
            return False
        else:
            self.cur_rel = self.dataset_rels[self.idx_rel]
            self.cur_task = self.dataset[self.cur_rel]
            self.idx_rel += 1
            return True

    def get_sup_inner_data(self):
        if self.sup_task is None:
            self.__init_sup_task()
        return self.get_inner_data()

    def get_qur_task_cand_rel(self):
        cand_buckets = []
        bucket = []
        for cand in self.rel2cand[self.cur_rel]: # cands in rel2cand have been sorted aescendingly
            if len(bucket) >= self.cand_bucket_size:
                cand_buckets.append(bucket)
                bucket = []
            bucket.append(cand)
        if len(bucket) != 0:
            cand_buckets.append(bucket)

        batch_dscp_cand = []
        for bucket in cand_buckets:
            max_bucket_dscp_len = len(self.ent_dscps[bucket[-1]])
            t_dscp_cand = T.full((len(bucket), max_bucket_dscp_len), self.idx_word_pad, dtype = T.long)
            for i, cand in enumerate(bucket):
                cand_dscp = self.ent_dscps[cand]
                t_dscp_cand[i, :len(cand_dscp)] = cand_dscp
            batch_dscp_cand.append(t_dscp_cand.to(self.device))

        t_dscpr = self.rel_dscps[self.cur_rel].long().to(self.device)
        return batch_dscp_cand, t_dscpr

    # this method only iterate through entities within a relation
    def get_qur_inner_data(self):
        if self.idx_ent >= len(self.cur_task):
            return None

        ent1, ent_true = self.cur_task[self.idx_ent]
        self.idx_ent += 1

        t_dscp1 = self.ent_dscps[ent1].to(self.device)
        t_dscp2 = self.ent_dscps[ent_true].to(self.device)

        cands = []
        for idx_cand, cand in enumerate(self.rel2cand[self.cur_rel]):
            if cand not in self.e1rel_e2[(ent1, self.cur_rel)] and cand != ent_true:
                cands.append(idx_cand) # local index of candidate entity in rel2cand[cur_rel]
        t_idx_cands = T.tensor(cands, dtype = T.long, device = self.device)

        qur_task = [t_dscp1, t_dscp2, t_idx_cands]
        return qur_task

    def __init_sup_task(self):
        self.idx_sup_inner = 0
        task_src = []

        for ent1, ent2 in self.cur_task[:self.num_shot]:
            noise = self.sample_sup_noise(ent1, self.cur_rel, ent2)
            task_src.append((ent1, ent2, noise, self.cur_rel))
        self.init_sup_tensors(task_src)

    def __reset(self):
        self.idx_rel = 0
        self.idx_ent = self.num_shot
        self.cur_rel = self.dataset_rels[self.idx_rel]
        self.cur_task = self.dataset[self.cur_rel]

if __name__ == "__main__":
    hyp = init_hyp()
    hyp.device = T.device("cpu")

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(levelname)s: - %(message)s', datefmt = '%Y-%m-%d %H:%M:%S')
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    train_task, dev_task, test_task, ent_dscps, rel_dscps, i2r, w2i, i2w, rel_cand, e1rel_e2 = all_data 
