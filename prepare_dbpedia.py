import string
import random
import json
from sys import exit
import os
import re
from collections import defaultdict, Counter
import pickle
import torch as T
from hyp_reptile import *

def build_dscp(hyp):
    ents = set()
    rels = set()
    #ent_names = {}
    rel_names = {}
    ent_dscps = {}
    w_cnt = Counter()

    rm_aster = lambda s : re.sub(r'\*', '', s) # remove *
    tokenize = lambda s : re.sub('[{}]'.format(re.escape(string.punctuation)), r' ', s) # replace punctuations with space
    replace_num = lambda s, num_token : re.sub(r'\d+', num_token, s) # replace continuous numbers with '$'

    '''
    with open(hyp.dataset_root + "entity_names.txt", "r", encoding = "utf-8") as f_ent:
        for ii, l in enumerate(f_ent):
            l = l.strip().encode("ascii", "ignore").decode("ascii").lower().split("\t")
            if len(l) != 3:
                continue
            token, name = l[0], l[2]
            name = replace_num(tokenize(rm_aster(name)), hyp.NUM) if hyp.rm_num_in_name else tokenize(rm_aster(name))
            name = name.split()
            for e in name:
                w_cnt[e] += 1
            ent_names[token] = name
            ents.add(token)
    '''

    with open(hyp.dataset_root + "relation_names.txt", "r", encoding = "utf-8") as f_ent:
        for l in f_ent:
            l = l.strip().encode("ascii", "ignore").decode("ascii").lower().split("\t")
            if len(l) != 3:
                continue
            token, name = l[0], l[2]
            name = replace_num(tokenize(rm_aster(name)), hyp.NUM) if hyp.rm_num_in_name else tokenize(rm_aster(name))
            name = name.split()
            for e in name:
                w_cnt[e] += 1
            rel_names[token] = name
            rels.add(token)

    with open(hyp.dataset_root + "descriptions.txt", "r", encoding = "utf-8") as f_dscp:
        for ii, l in enumerate(f_dscp):
            l = l.strip().encode("ascii", "ignore").decode("ascii").lower().split("\t")
            if len(l) != 3:
                continue
            token, dscp = l[0], l[2]
            dscp = replace_num(tokenize(rm_aster(dscp)), hyp.NUM)
            dscp = dscp.split()
            dscp = dscp[:hyp.max_len_dscp]
            for e in dscp:
                w_cnt[e] += 1
            ent_dscps[token] = dscp
            ents.add(token)

    pkl_path = hyp.tmp_root + "dscp.pkl.tmp"
    with open(pkl_path, 'wb') as f_dump:
        pickle.dump((ents, rels, rel_names, ent_dscps, w_cnt), f_dump, protocol = pickle.HIGHEST_PROTOCOL)
    return ents, rels, rel_names, ent_dscps, w_cnt

def load_dscp(hyp):
    pkl_path = hyp.tmp_root + "dscp.pkl.tmp"
    with open(pkl_path, 'rb') as f_dump:
        ents, rels, rel_names, ent_dscps, w_cnt = pickle.load(f_dump)
    return ents, rels, rel_names, ent_dscps, w_cnt

def build_dataset(dscp_ents, dscp_rels, hyp):
    def read_dataset(src_path, all_triplets, dscp_ents, dscp_rels):
        with open(src_path, "r", encoding = "utf-8") as f_dataset:
            for l in f_dataset:
                l = l.strip().encode("ascii", "ignore").decode("ascii").lower().split()
                if len(l) != 3:
                    #print(l)
                    continue

                ent1, ent2, rel = l
                if ent1 not in dscp_ents:
                    #print("ent1 missing: {}".format(ent1))
                    continue
                if ent2 not in dscp_ents:
                    #print("ent2 missing: {}".format(ent2))
                    continue
                if rel not in dscp_rels:
                    #print("rel missing: {}".format(rel))
                    continue
                all_triplets[rel].append((ent1, ent2))

    all_triplets = defaultdict(list)
    print(len(all_triplets))
    read_dataset(hyp.dataset_root + "train.txt", all_triplets, dscp_ents, dscp_rels)
    print(len(all_triplets))
    read_dataset(hyp.dataset_root + "valid.txt", all_triplets, dscp_ents, dscp_rels)
    print(len(all_triplets))
    read_dataset(hyp.dataset_root + "test.txt", all_triplets, dscp_ents, dscp_rels)
    print(len(all_triplets))

    valid_all_triplets = {}
    for rel, duos in all_triplets.items():
        if len(duos) >= hyp.min_task_size and len(duos) <= hyp.max_task_size:
            valid_all_triplets[rel] = duos
    print("after filtering small and large tasks, #tasks =" ,len(valid_all_triplets))

    if hyp.is_debugging:
        train_size = dev_size = test_size = len(valid_all_triplets) // 3
    else:
        train_size, dev_size, test_size = hyp.tasks_split_size
    train_dataset = defaultdict(list)
    dev_dataset = defaultdict(list)
    test_dataset = defaultdict(list)

    task_ents = set()
    task_rels = set()
    for idx_rel, (rel, duos) in enumerate(valid_all_triplets.items()):
        if idx_rel < train_size:
            dst_dataset = train_dataset
        elif idx_rel >= train_size and idx_rel < train_size + dev_size:
            dst_dataset = dev_dataset
        else:
            dst_dataset = test_dataset

        if hyp.is_debugging:
            duos = duos[:10]
        dst_dataset[rel] = duos
        for ent1, ent2 in duos:
            task_ents.add(ent1)
            task_ents.add(ent2)
        task_rels.add(rel)

    print("after building few shot dataset, #train = {}, #dev = {}, #test = {}".format(len(train_dataset), len(dev_dataset), len(test_dataset)))
    
    pkl_path = hyp.tmp_root + "dataset.pkl.tmp.debug" if hyp.is_debugging else hyp.tmp_root + "dataset.pkl.tmp"
    with open(pkl_path, 'wb') as f_dump:
        pickle.dump((train_dataset, dev_dataset, test_dataset, task_ents, task_rels), f_dump, protocol = pickle.HIGHEST_PROTOCOL)

    return train_dataset, dev_dataset, test_dataset, task_ents, task_rels

def load_dataset(hyp):
    pkl_name = "dataset.pkl.tmp"
    if hyp.is_debugging:
        pkl_name += ".debug"
    pkl_path = hyp.tmp_root + pkl_name

    with open(pkl_path, 'rb') as f_dump:
        train_dataset, dev_dataset, test_dataset, task_ents, task_rels = pickle.load(f_dump)
    return train_dataset, dev_dataset, test_dataset, task_ents, task_rels

def build_map(ents, rels, w_cnt, hyp):
    w2i = {}
    i2w = {}
    for w, cnt in w_cnt.items():
        if cnt < hyp.min_word_cnt:
            continue
        i = len(w2i)
        w2i[w] = i
        i2w[i] = w
    if hyp.UNK not in w2i:
        i = len(w2i)
        w2i[hyp.UNK] = i
        i2w[i] = hyp.UNK
    if hyp.NUM not in w2i:
        i = len(w2i)
        w2i[hyp.NUM] = i
        i2w[i] = hyp.NUM
    if hyp.WORD_PAD not in w2i:
        i = len(w2i)
        w2i[hyp.WORD_PAD] = i
        i2w[i] = hyp.WORD_PAD
    print("min_word_cnt = {}, after filtering, #word = {}".format(hyp.min_word_cnt, len(i2w)))

    e2i = {}
    i2e = {}
    r2i = {}
    i2r = {}
    for rel in rels:
        i = len(r2i)
        r2i[rel] = i
        i2r[i] = rel
    for ent in ents:
        i = len(e2i)
        e2i[ent] = i
        i2e[i] = ent

    print("#ent = {}, #rel = {}".format(len(ents), len(rels)))
    pkl_path = hyp.tmp_root + "map.pkl.tmp.debug" if hyp.is_debugging else hyp.tmp_root + "map.pkl.tmp"
    with open(pkl_path, 'wb') as f_dump:
        pickle.dump((e2i, i2e, r2i, i2r, w2i, i2w), f_dump, protocol = pickle.HIGHEST_PROTOCOL)
    return e2i, i2e, r2i, i2r, w2i, i2w

def load_map(hyp):
    pkl_path = hyp.tmp_root + "map.pkl.tmp.debug" if hyp.is_debugging else hyp.tmp_root + "map.pkl.tmp"
    with open(pkl_path, 'rb') as f_dump:
        e2i, i2e, r2i, i2r, w2i, i2w = pickle.load(f_dump)
    return e2i, i2e, r2i, i2r, w2i, i2w

def token2idx(train_dataset, dev_dataset, test_dataset, rel_names, ent_dscps, e2i, r2i, w2i):
    new_train_dataset = defaultdict(list)
    new_dev_dataset = defaultdict(list)
    new_test_dataset = defaultdict(list)
    new_rel_names = {}
    new_ent_dscps = {}

    for src, dst in zip([train_dataset, dev_dataset, test_dataset], [new_train_dataset, new_dev_dataset, new_test_dataset]):
        for rel, duos in src.items():
            for ent1, ent2 in duos:
                dst[r2i[rel]].append((e2i[ent1], e2i[ent2]))

    for rel, name in rel_names.items():
        if rel in r2i:
            idx_rel = r2i[rel]
            new_rel_names[idx_rel] = T.tensor([w2i[w] if w in w2i else w2i[hyp.UNK] for w in name], dtype = T.long)
    for ent, dscp in ent_dscps.items():
        if ent in e2i:
            idx_ent = e2i[ent]
            new_ent_dscps[idx_ent] = T.tensor([w2i[w] if w in w2i else w2i[hyp.UNK] for w in dscp], dtype = T.long)

    print("finish transforming symbols to indices")
    pkl_path = "idx.pkl.tmp.debug" if hyp.is_debugging else "idx.pkl.tmp"
    with open(hyp.tmp_root + pkl_path, 'wb') as f_dump:
        pickle.dump((new_train_dataset, new_dev_dataset, new_test_dataset, new_rel_names, new_ent_dscps), f_dump, protocol = pickle.HIGHEST_PROTOCOL)
    return new_train_dataset, new_dev_dataset, new_test_dataset, new_rel_names, new_ent_dscps

def load_idx(hyp):
    pkl_path = "idx.pkl.tmp.debug" if hyp.is_debugging else "idx.pkl.tmp"
    with open(hyp.tmp_root + pkl_path, 'rb') as f_dump:
        train_dataset, dev_dataset, test_dataset, rel_names, ent_dscps = pickle.load(f_dump)
    return train_dataset, dev_dataset, test_dataset, rel_names, ent_dscps

def build_aux(train_dataset, dev_dataset, test_dataset, ent_dscps, e2i, i2r, hyp):
    e1rel_e2 = defaultdict(set)
    for src in [train_dataset, dev_dataset]:
        for idx_rel, duos in src.items():
            for idx_ent1, idx_ent2 in duos:
                e1rel_e2[(idx_ent1, idx_rel)].add(idx_ent2)

    rel_cand = defaultdict(set)
    for src in [train_dataset, dev_dataset, test_dataset]:
        for idx_rel, duos in src.items():
            for idx_ent1, idx_ent2 in duos:
                if idx_ent2 not in rel_cand[idx_rel]:
                    rel_cand[idx_rel].add(idx_ent2)
                    if len(rel_cand[idx_rel]) >= hyp.max_num_cand:
                        break

    print("remaining candidates / origin candidates in dataset:")
    for idx_rel, cands in rel_cand.items():
        purposed_num_cand = len(cands) if len(cands) > hyp.min_num_cand else hyp.min_num_cand
        if len(cands) < purposed_num_cand:
            num_left = purposed_num_cand - len(cands)
            while num_left > 0:
                while 1:
                    idx_ent = random.randint(0, len(e2i) - 1)
                    if idx_ent not in cands:
                        break
                cands.add(idx_ent)
                num_left -= 1
        cands = list(cands)
        cands.sort(key = lambda idx_ent, ent_dscps = ent_dscps : len(ent_dscps[idx_ent]))
        #print("{}: {}".format(i2r[idx_rel], len(cands)))
        rel_cand[idx_rel] = cands

    print("finish building e1rel_e2, rel_cand")
    pkl_path = "aux.pkl.tmp.debug" if hyp.is_debugging else "aux.pkl.tmp"
    with open(hyp.tmp_root + pkl_path, 'wb') as f_dump:
        pickle.dump((e1rel_e2, rel_cand), f_dump, protocol = pickle.HIGHEST_PROTOCOL)
    return e1rel_e2, rel_cand

def load_aux(hyp):
    pkl_path = "aux.pkl.tmp.debug" if hyp.is_debugging else "aux.pkl.tmp"
    with open(hyp.tmp_root + pkl_path, 'rb') as f_dump:
        e1rel_e2, rel_cand = pickle.load(f_dump)
    return e1rel_e2, rel_cand

def prepare_task(hyp):
    dscp_ents, dscp_rels, rel_names, ent_dscps, w_cnt = build_dscp(hyp)
    #dscp_ents, dscp_rels, rel_names, ent_dscps, w_cnt = load_dscp(hyp)
    print(1)

    raw_train_dataset, raw_dev_dataset, raw_test_dataset, task_ents, task_rels = build_dataset(dscp_ents, dscp_rels, hyp)
    #raw_train_dataset, raw_dev_dataset, raw_test_dataset, task_ents, task_rels = load_dataset(hyp)
    print(2)

    e2i, i2e, r2i, i2r, w2i, i2w = build_map(task_ents, task_rels, w_cnt, hyp)
    #e2i, i2e, r2i, i2r, w2i, i2w = load_map(hyp)
    print(3)

    train_dataset, dev_dataset, test_dataset, rel_names, ent_dscps = token2idx(raw_train_dataset, raw_dev_dataset, raw_test_dataset, rel_names, ent_dscps, e2i, r2i, w2i)
    #train_dataset, dev_dataset, test_dataset, rel_names, ent_dscps = load_idx(hyp)
    print(4)

    e1rel_e2, rel_cand = build_aux(train_dataset, dev_dataset, test_dataset, ent_dscps, e2i, i2r, hyp)
    #e1rel_e2, rel_cand = load_aux(hyp)
    print(5)

    all_data = [train_dataset, dev_dataset, test_dataset, ent_dscps, rel_names, i2r, w2i, i2w, rel_cand, e1rel_e2]
    pkl_path = hyp.raw_pkl_path + ".debug" if hyp.is_debugging else hyp.raw_pkl_path
    print("start dumping to {0}".format(pkl_path))
    with open(pkl_path, 'wb') as f_dump:
        pickle.dump(all_data, f_dump, protocol = pickle.HIGHEST_PROTOCOL)
    print("finish")

    return all_data

if __name__ == "__main__":
    hyp = init_hyp("dbpedia")
    prepare_task(hyp)
