# -*- coding: utf-8 -*-
import sys
import logging
from logging import info, warn, error
from sys import exit

from util import *

class Metrics(object):
    def __init__(self):
        self.hits10 = 0
        self.hits5 = 0
        self.hits1 = 0
        self.mrr = 0
        self.task_hits10 = 0
        self.task_hits5 = 0
        self.task_hits1 = 0
        self.task_mrr = 0
        self.num_task_data = 0
        self.num_all_data = 0

    def reset_task_metrics(self, task):
        self.cur_task = task
        self.task_hits10 = 0
        self.task_hits5 = 0
        self.task_hits1 = 0
        self.task_mrr = 0
        self.task_rank = []
        self.num_task_data = 0

    def add(self, data):
        idx_sorted = list(np.argsort(data)) # ascending
        rank = idx_sorted.index(0) + 1
        if rank <= 10:
            self.hits10 += 1
            self.task_hits10 += 1
        if rank <= 5:
            self.hits5 += 1
            self.task_hits5 += 1
        if rank <= 1:
            self.hits1 += 1
            self.task_hits1 += 1
        self.mrr += 1 / rank
        self.task_mrr += 1 / rank
        self.num_task_data += 1
        self.num_all_data += 1
        self.task_rank.append((rank, data.shape[0]))

    def log_task_metric(self):
        task_metrics = {}
        task_metrics["hits10"] = self.task_hits10 / self.num_task_data
        task_metrics["hits5"] = self.task_hits5 / self.num_task_data
        task_metrics["hits1"] = self.task_hits1 / self.num_task_data
        task_metrics["mrr"] = self.task_mrr / self.num_task_data
        prefix = "task {}: {} data, ".format(self.cur_task, self.num_task_data)
        '''
        print("task rank: ", end = " ")
        for duo in self.task_rank:
            print("{}/{} ".format(duo[0], duo[1]), end = " ")
        print("")
        '''
        self.__log_metric(task_metrics, prefix)


    def log_overall_metric(self):
        metrics = {}
        metrics["hits10"] = self.hits10 / self.num_all_data
        metrics["hits5"] = self.hits5 / self.num_all_data
        metrics["hits1"] = self.hits1 / self.num_all_data
        metrics["mrr"] = self.mrr / self.num_all_data
        prefix = "overall dataset: {} data, ".format(self.num_all_data)
        self.__log_metric(metrics, prefix)


    def __log_metric(self, metrics, prefix):
        logging_info = prefix
        for name, val in metrics.items():
            logging_info += "{} = {}    ".format(name, "{:.4f}".format(val))
        info(logging_info)

