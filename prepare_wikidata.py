import string
import random
import json
from sys import exit
import os
import re
from collections import defaultdict, Counter, namedtuple
import pickle
import torch as T
from hyp_reptile import *

def build_dscp(hyp):
    w_cnt = Counter()
    chars = set()
    ents = set()
    rels = set()
    ent_dscps = {}
    rel_dscps = {}
    ent_parents = {}
    rel_parents = {}
    parent_ents = set()
    wiki_dscp_path = hyp.proj_root + "data/wikidata/dscp_all.txt"

    rm_aster = lambda s : re.sub(r'\*', '', s) # remove *
    tokenize = lambda s : re.sub('([{}])'.format(re.escape(string.punctuation)), r' \1 ', s) # separate punctuations
    replace_num = lambda s, num_token : re.sub(r'\d+', num_token, s) # replace continuous numbers with '$'

    with open(wiki_dscp_path, "r", encoding = "utf-8") as f_src:
        for ii, l in enumerate(f_src):
            l = l.strip().split("####")
            if len(l) != 4:
                continue
            token, name, dscp, parents = l[0], l[1].lower(), l[2].lower(), l[3].split()
            if name == "none" or dscp == "none":
                continue
            name = replace_num(rm_aster(name), hyp.NUM) if hyp.rm_num_in_name else rm_aster(name)
            dscp = replace_num(tokenize(rm_aster(dscp)), hyp.NUM)
            words = dscp.split()
            for e in words:
                w_cnt[e] += 1
            for c in name:
                chars.add(c)
            if token[0] == "P":
                rels.add(token)
                rel_dscps[token] = (name, words)
                rel_parents[token] = parents
            elif token[0] == "Q":
                ents.add(token)
                ent_dscps[token] = (name, words)
                ent_parents[token] = parents
            else:
                raise RuntimeError(token)
            for e in parents:
                parent_ents.add(e)

    pkl_path = hyp.tmp_root + "dscp.pkl.tmp"
    with open(pkl_path, 'wb') as f_dump:
        pickle.dump((ents, rels, ent_dscps, rel_dscps, w_cnt, chars, ent_parents, rel_parents, parent_ents), f_dump, protocol = pickle.HIGHEST_PROTOCOL)
    return ents, rels, ent_dscps, rel_dscps, w_cnt, chars, ent_parents, rel_parents, parent_ents

def load_dscp(hyp):
    pkl_path = hyp.tmp_root + "dscp.pkl.tmp"
    with open(pkl_path, 'rb') as f_dump:
        dscp_ents, dscp_rels, ent_dscps, rel_dscps, w_cnt, chars, ent_parents, rel_parents, parent_ents = pickle.load(f_dump)
    return dscp_ents, dscp_rels, ent_dscps, rel_dscps, w_cnt, chars, ent_parents, rel_parents, parent_ents

def build_task(dscp_ents, dscp_rels, hyp):
    src_root = hyp.json_dataset_root
    raw_train_task = json.load(open(src_root + "train_tasks.json", "r"))
    raw_dev_task = json.load(open(src_root + "dev_tasks.json", "r"))
    raw_test_task = json.load(open(src_root + "test_tasks.json", "r"))
    task_ents = set()
    task_rels = set()
    train_task = defaultdict(list)
    dev_task = defaultdict(list)
    test_task = defaultdict(list)

    dev_size1 = 0
    dev_size2 = 0
    for rel, tups in raw_dev_task.items():
        dev_size1 += len(tups)
        if rel not in dscp_rels:
            print("dev task {} has no dscp".format(rel))
            continue

        valid_tups = []
        for ii, tup in enumerate(tups):
            if hyp.is_debugging and ii >= 10:
                break
            if tup[0] in dscp_ents and tup[2] in dscp_ents:
                valid_tups.append((tup[0], tup[2]))
        if len(valid_tups) < hyp.min_task_size:
            print("dev task {} is too small: size = {}".format(rel, len(valid_tups)))
            continue

        dev_task[rel] += valid_tups
        dev_size2 += len(valid_tups)
        task_rels.add(rel)
        for e1, e2 in valid_tups:
            task_ents.add(e1)
            task_ents.add(e2)
    print("origin dev task: #task = {}, #tuple = {}, after filtering entities and relations without description, #task = {}, #tuple = {}".format(len(raw_dev_task), dev_size1, len(dev_task), dev_size2))

    test_size1 = 0
    test_size2 = 0
    for rel, tups in raw_test_task.items():
        test_size1 += len(tups)
        if rel not in dscp_rels:
            print("test task {} has no dscp".format(rel))
            continue

        valid_tups = []
        for ii, tup in enumerate(tups):
            if hyp.is_debugging and ii >= 10:
                break
            if tup[0] in dscp_ents and tup[2] in dscp_ents:
                valid_tups.append((tup[0], tup[2]))
        if len(valid_tups) < hyp.min_task_size:
            print("test task {} is too small: size = {}".format(rel, len(valid_tups)))
            continue

        test_task[rel] += valid_tups
        test_size2 += len(valid_tups)
        task_rels.add(rel)
        for e1, e2 in valid_tups:
            task_ents.add(e1)
            task_ents.add(e2)
    print("origin test task: #task = {}, #tuple = {}, after filtering entities and relations without description, #task = {}, #tuple = {}".format(len(raw_test_task), test_size1, len(test_task), test_size2))

    train_size1 = 0
    train_size2 = 0
    train_size3 = 0
    for rel, tups in raw_train_task.items():
        train_size1 += len(tups)
        if rel not in dscp_rels:
            print("train task {} has no dscp".format(rel))
            continue

        possible_tups = []
        for ii, tup in enumerate(tups):
            if hyp.is_debugging and ii >= 10:
                break
            if tup[0] in dscp_ents and tup[2] in dscp_ents:
                possible_tups.append((tup[0], tup[2]))
        train_size2 += len(possible_tups)
        valid_tups = []
        for e1, e2 in possible_tups:
            valid_tups.append((e1, e2))
        if len(valid_tups) < hyp.min_task_size:
            print("train task {} is too small: size = {}".format(rel, len(valid_tups)))
            continue

        train_task[rel] += valid_tups
        train_size3 += len(valid_tups)
        task_rels.add(rel)
        for e1, e2 in valid_tups:
            task_ents.add(e1)
            task_ents.add(e2)
    print("origin training task: #task = {}, #tuple = {}, after filtering by e2i, r2i, #task = {}, #tuple = {}, after filtering dev_test entities, #tuple = {}".format(len(raw_train_task), train_size1, len(train_task), train_size2, train_size3))

    pkl_path = hyp.tmp_root
    pkl_path += "task.pkl.tmp.debug" if hyp.is_debugging else "task.pkl.tmp"
    with open(pkl_path, 'wb') as f_dump:
        pickle.dump((train_task, dev_task, test_task, task_ents, task_rels), f_dump, protocol = pickle.HIGHEST_PROTOCOL)
    return train_task, dev_task, test_task, task_ents, task_rels

def load_task(hyp):
    pkl_path = hyp.tmp_root
    pkl_path += "task.pkl.tmp.debug" if hyp.is_debugging else "task.pkl.tmp"
    with open(pkl_path, 'rb') as f_dump:
        train_task, dev_task, test_task, task_ents, task_rels = pickle.load(f_dump)
    return train_task, dev_task, test_task, task_ents, task_rels

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
    pkl_path = hyp.tmp_root
    pkl_path += "map.pkl.tmp.debug" if hyp.is_debugging else "map.pkl.tmp"
    with open(pkl_path, 'wb') as f_dump:
        pickle.dump((e2i, i2e, r2i, i2r, w2i, i2w), f_dump, protocol = pickle.HIGHEST_PROTOCOL)
    return e2i, i2e, r2i, i2r, w2i, i2w

def load_map(hyp):
    pkl_path = hyp.tmp_root
    pkl_path += "map.pkl.tmp.debug" if hyp.is_debugging else "map.pkl.tmp"
    with open(pkl_path, 'rb') as f_dump:
        e2i, i2e, r2i, i2r, w2i, i2w = pickle.load(f_dump)
    return e2i, i2e, r2i, i2r, w2i, i2w

def token2idx(train_task, dev_task, test_task, ent_dscps, rel_dscps, e2i, r2i, w2i):
    new_train_task = defaultdict(list)
    new_dev_task = defaultdict(list)
    new_test_task = defaultdict(list)
    new_ent_dscps = {}
    new_rel_dscps = {}

    for src, dst in zip([train_task, dev_task, test_task], [new_train_task, new_dev_task, new_test_task]):
        for rel, duos in src.items():
            for duo in duos:
                dst[r2i[rel]].append((e2i[duo[0]], e2i[duo[1]]))

    for ent, (name, dscp) in ent_dscps.items():
        if ent in e2i:
            idx_ent = e2i[ent]
            new_ent_dscps[idx_ent] = T.tensor([w2i[w] if w in w2i else w2i[hyp.UNK] for w in dscp], dtype = T.long)
    for rel, (name, dscp) in rel_dscps.items():
        if rel in r2i:
            idx_rel = r2i[rel]
            new_rel_dscps[idx_rel] = T.tensor([w2i[w] if w in w2i else w2i[hyp.UNK] for w in dscp], dtype = T.long)

    print("finish transforming symbols to indices")
    pkl_path = hyp.tmp_root
    pkl_path += "idx.pkl.tmp.debug" if hyp.is_debugging else "idx.pkl.tmp"
    with open(pkl_path, 'wb') as f_dump:
        pickle.dump((new_train_task, new_dev_task, new_test_task, new_ent_dscps, new_rel_dscps), f_dump, protocol = pickle.HIGHEST_PROTOCOL)
    return new_train_task, new_dev_task, new_test_task, new_ent_dscps, new_rel_dscps

def load_idx(hyp):
    pkl_path = hyp.tmp_root
    pkl_path += "idx.pkl.tmp.debug" if hyp.is_debugging else "idx.pkl.tmp"
    with open(pkl_path, 'rb') as f_dump:
        train_task, dev_task, test_task, ent_dscps, rel_dscps = pickle.load(f_dump)
    return train_task, dev_task, test_task, ent_dscps, rel_dscps

def build_aux(train_task, dev_task, test_task, e2i, i2r, ent_dscps, hyp):
    raw_rel_cand = json.load(open(hyp.json_dataset_root + "rel2candidates.json", "r"))
    new_rel_cand = {}
    valid_rels = list(train_task) + list(dev_task) + list(test_task)

    e1rel_e2 = defaultdict(set)
    for tasks in [train_task, dev_task]:
        for idx_rel, duos in tasks.items():
            for idx_e1, idx_e2 in duos:
                e1rel_e2[(idx_e1, idx_rel)].add(idx_e2)

    print("remaining candidates / origin candidates in dataset:")
    for idx_rel in valid_rels:
        ents = raw_rel_cand[i2r[idx_rel]]
        cands = []
        for ent in ents:
            if ent in e2i:
                idx_ent = e2i[ent]
                if idx_ent in ent_dscps:
                    cands.append(idx_ent)

        purposed_num_cand = len(ents) if len(ents) > hyp.min_num_cand else hyp.min_num_cand
        if len(cands) < purposed_num_cand:
            existing_cand = set(cands)
            num_left = purposed_num_cand - len(cands)
            while num_left > 0:
                while 1:
                    idx_ent = random.randint(0, len(e2i) - 1)
                    if idx_ent not in ent_dscps:
                        continue
                    if idx_ent not in existing_cand:
                        break
                cands.append(idx_ent)
                num_left -= 1
        cands.sort(key = lambda idx_ent, ent_dscps = ent_dscps : len(ent_dscps[idx_ent]))

        print("{}: {}/{}".format(i2r[idx_rel], len(cands), len(ents)))
        new_rel_cand[idx_rel] = cands

    print("finish building rel_cand and e1rel_e2")
    pkl_path = hyp.tmp_root
    pkl_path += "aux.pkl.tmp.debug" if hyp.is_debugging else "aux.pkl.tmp"
    with open(pkl_path, 'wb') as f_dump:
        pickle.dump((new_rel_cand, e1rel_e2), f_dump, protocol = pickle.HIGHEST_PROTOCOL)
    return new_rel_cand, e1rel_e2

def load_aux(hyp):
    pkl_path = hyp.tmp_root
    pkl_path += "aux.pkl.tmp.debug" if hyp.is_debugging else "aux.pkl.tmp"
    with open(pkl_path, 'rb') as f_dump:
        rel_cand, e1rel_e2 = pickle.load(f_dump)
    return rel_cand, e1rel_e2

def prepare_task(hyp):
    print("start preparing wikidata pickles")
    #dscp_ents, dscp_rels, ent_dscps, rel_dscps, w_cnt, chars, ent_parents, rel_parents, parent_ents = build_dscp(hyp)
    dscp_ents, dscp_rels, ent_dscps, rel_dscps, w_cnt, chars, ent_parents, rel_parents, parent_ents = load_dscp(hyp)
    print(1)

    #train_task, dev_task, test_task, task_ents, task_rels = build_task(dscp_ents, dscp_rels, hyp)
    train_task, dev_task, test_task, task_ents, task_rels = load_task(hyp)
    '''
    print("train")
    for k in train_task:
        print(k, len(train_task[k]))
    print("dev")
    for k in dev_task:
        print(k, len(dev_task[k]))
    print("test")
    for k in test_task:
        print(k, len(test_task[k]))
    exit(0)
    '''
    print(2)

    #e2i, i2e, r2i, i2r, w2i, i2w = build_map(task_ents, task_rels, w_cnt, hyp)
    e2i, i2e, r2i, i2r, w2i, i2w = load_map(hyp)
    print(3)

    #train_task, dev_task, test_task, ent_dscps, rel_dscps = token2idx(train_task, dev_task, test_task, ent_dscps, rel_dscps, e2i, r2i, w2i)
    train_task, dev_task, test_task, ent_dscps, rel_dscps = load_idx(hyp)

    print(len(task_ents))
    print(len(e2i))
    print(len(ent_dscps))
    #rel_cand, e1rel_e2 = build_aux(train_task, dev_task, test_task, e2i, i2r, ent_dscps, hyp)
    rel_cand, e1rel_e2 = load_aux(hyp)

    all_data = [train_task, dev_task, test_task, ent_dscps, rel_dscps, i2r, w2i, i2w, rel_cand, e1rel_e2]
    pkl_path = hyp.raw_pkl_path + ".debug" if hyp.is_debugging else hyp.raw_pkl_path
    print("start dumping to {0}".format(pkl_path))
    with open(pkl_path, 'wb') as f_dump:
        pickle.dump(all_data, f_dump, protocol = pickle.HIGHEST_PROTOCOL)
    print("finish")

    return all_data

if __name__ == "__main__":
    hyp = init_hyp("wikidata")
    prepare_task(hyp)
