# -*- coding: utf-8 -*-

def get_dataset_config(dataset):
    config = {}
    if dataset == "wikidata":
        config["tmp_root"] = "./tmp/wikidata/"
        config["json_dataset_root"] = "/gds/zhwang/zhwang/data/knowledge_graph/wikidata/"
        config["pkl_dataset_root"] = "/gds/zhwang/zhwang/workspace/knowledge_graph/zero_shot/"
        config["raw_pkl_path"] = "./data/wikidata/all_data.pkl"

        config["min_word_cnt"] = 1
        config["rm_num_in_name"] = True
        config["max_len_dscp"] = 32
        config["min_task_size"] = 5
        config["min_num_cand"] = 100

        config["encoder_dim_pool_kernel"] = 2
        config["num_aug"] = 8
        config["margin"] = 1.0
    elif dataset == "dbpedia":
        config["tmp_root"] = "./tmp/dbpedia/"
        config["dataset_root"] = "/gds/zhwang/zhwang/data/knowledge_graph/dbpedia/dbpedia500/"
        config["raw_pkl_path"] = "./data/dbpedia/all_data.pkl"

        config["min_word_cnt"] = 10
        config["rm_num_in_name"] = True
        config["max_len_dscp"] = 200
        config["tasks_split_size"] = [220, 30, 69] # 319
        config["min_task_size"] = 5
        config["max_task_size"] = 1000
        config["min_num_cand"] = 100
        config["max_num_cand"] = 1000

        config["encoder_dim_pool_kernel"] = 4
        config["num_aug"] = 128
        config["margin"] = 2.0
    else:
        raise RuntimeError("wrong dataset in get_dataset_config()")

    return config
