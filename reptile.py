# -*- coding: utf-8 -*-
from sys import exit
from logging import info, error
from metrics import Metrics
import torch as T
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

#from cnn_encoder import CNN_Encoder
from memory_encoder import MemoryEncoder
from triplet_vae import TripletVAE

class REPTILE(nn.Module):
    def __init__(self, hyp):
        super(REPTILE, self).__init__()
        self.hyp = hyp
        self.memory = hyp.memory
        self.device = hyp.device

        if self.memory:
            self.encoder = MemoryEncoder(hyp)
            info("finish initializing memory encoder")
        else:
            exit(-1)
            #self.encoder = CNN_Encoder(hyp)
            #info("finish initializing CNN encoder")

        self.vae = TripletVAE(hyp)
        self.modules = {
                "encoder" : self.encoder,
                "vae" : self.vae
                }

        if hyp.sim_func == "TransE":
            self.sim_func = lambda dscp1_emb, dscp2_emb, dscpr_emb: T.sum(T.abs(dscp1_emb + dscpr_emb - dscp2_emb), -1)
        else:
            error("sim_func {} has not implemented. Candidates: TransE".format(hyp.sim_func))
            exit(-1)

        if hyp.use_vae:
            info("VAE is used to augment data")
            self.optm = optim.Adam(self.parameters(), lr = hyp.inner_lr, betas = (0.0, 0.999))
        else:
            self.optm = optim.Adam(self.encoder.parameters(), lr = hyp.inner_lr, betas = (0.0, 0.999))
        info("finish initializing REPTILE")

    def get_state_dict(self):
        state_dict = {
                "encoder" : self.encoder.state_dict(),
                "vae" : self.vae.state_dict(),
                "optm" : self.optm.state_dict()
                }
        return state_dict

    def set_state_dict(self, checkpoint):
        self.encoder.load_state_dict(checkpoint["encoder"])
        self.vae.load_state_dict(checkpoint["vae"])
        self.optm.load_state_dict(checkpoint["optm"])

    def meta_train(self, train_dataset):
        weights_original = self.copy_param()
        new_weights = None
        batch_loss = defaultdict(float)

        for idx_batch in range(self.hyp.meta_batch_size):
            dscp1, dscp2, dscpn, dscpr = train_dataset.get_sup_inner_data()
            for idx_iter in range(self.hyp.num_train_inner_iter):
                loss = self.__inner_train_step(dscp1, dscp2, dscpn, dscpr, True)
                for name, val in loss:
                    batch_loss[name] += val
            if idx_batch == 0:
                new_weights = self.copy_param()
            else:
                tmp = self.copy_param()
                for module_name in tmp:
                    for param_name in tmp[module_name]:
                        new_weights[module_name][param_name] += tmp[module_name][param_name]
                self.set_param(weights_original)
        tmp = self.hyp.num_train_inner_iter * self.hyp.meta_batch_size
        for name in batch_loss:
            batch_loss[name] /= tmp

        if self.hyp.meta_batch_size > 1:
            for module_name in new_weights:
                for param_name in new_weights[module_name]:
                    new_weights[module_name][param_name] /= self.hyp.meta_batch_size

        meta_updated_weights = defaultdict(dict)
        for module_name in new_weights:
            for param_name in new_weights[module_name]:
                meta_grad = weights_original[module_name][param_name] - new_weights[module_name][param_name]
                meta_updated_weights[module_name][param_name] = weights_original[module_name][param_name] - self.hyp.meta_lr * meta_grad
        self.set_param(meta_updated_weights)

        return batch_loss

    def __inner_train_step(self, dscp1, dscp2, dscpn, dscpr, meta_training):
        if self.memory:
            dscpr_emb, head_memory_dscpr_emb, tail_memory_dscpr_emb = self.encoder.encode_rel(dscpr)
            dscp1_emb = self.encoder(dscp1, head_memory_dscpr_emb)
            dscp2_emb = self.encoder(dscp2, tail_memory_dscpr_emb)
            dscpn_emb = self.encoder(dscpn, tail_memory_dscpr_emb)
        else:
            dscpr_emb = self.encoder(dscpr)
            dscp1_emb = self.encoder(dscp1)
            dscp2_emb = self.encoder(dscp2)
            dscpn_emb = self.encoder(dscpn)

        if self.hyp.use_vae and meta_training:
            vae_loss, vae_loss_print = self.vae.learn(dscp1_emb.detach(), dscp2_emb.detach(), dscpr_emb.detach())
        else:
            vae_loss = T.zeros(1, device = self.device)

        if self.hyp.use_vae and not meta_training:
            aug1_emb, aug2_emb = self.vae.generate(dscpr_emb.detach())
            dscp1_emb = T.cat([dscp1_emb, aug1_emb.detach()], 0)
            dscp2_emb = T.cat([dscp2_emb, aug2_emb.detach()], 0)

        sim_pos = self.sim_func(dscp1_emb, dscp2_emb, dscpr_emb)
        sim_neg = self.sim_func(dscp1_emb, dscpn_emb, dscpr_emb)
        kg_loss = T.mean(F.relu(self.hyp.margin + sim_pos - sim_neg))
        loss = kg_loss + vae_loss

        self.optm.zero_grad()
        loss.backward()
        self.optm.step()

        loss_print = [["kg_loss", kg_loss.data.cpu().item()]]
        if self.hyp.use_vae and meta_training:
            loss_print += vae_loss_print
        return loss_print

    def __predict_triplet(self, dscp1, dscp2, idx_cands):
        if self.memory:
            dscp1_emb = self.encoder(dscp1, self.qur_head_memory_dscpr_emb)
            dscp2_emb = self.encoder(dscp2, self.qur_tail_memory_dscpr_emb)
        else:
            dscp1_emb = self.encoder(dscp1)
            dscp2_emb = self.encoder(dscp2)
        cands_emb = T.cat([dscp2_emb, self.qur_cands_emb[idx_cands]], 0)

        sim_cand = self.sim_func(dscp1_emb, cands_emb, self.qur_dscpr_emb)
        return sim_cand

    def predict(self, dataset):
        weights_original = self.copy_param()
        metrics = Metrics()

        while dataset.next_rel():
            metrics.reset_task_metrics(dataset.cur_rel)

            for idx_iter in range(self.hyp.num_test_inner_iter):
                for idx_data in range(self.hyp.num_shot):
                    dscp1, dscp2, dscpn, dscpr = dataset.get_sup_inner_data()
                    for idx_iter in range(self.hyp.num_test_inner_iter):
                        self.__inner_train_step(dscp1, dscp2, dscpn, dscpr, False)

            batch_dscp_cand, dscpr = dataset.get_qur_task_cand_rel()
            cands_emb = []
            if self.memory:
                self.qur_dscpr_emb, self.qur_head_memory_dscpr_emb, self.qur_tail_memory_dscpr_emb = self.encoder.encode_rel(dscpr)
                cands_emb = [self.encoder(bucket, self.qur_tail_memory_dscpr_emb) for bucket in batch_dscp_cand]
            else:
                self.qur_dscpr_emb = self.encoder(dscpr)
                cands_emb = [self.encoder(bucket) for bucket in batch_dscp_cand]
            self.qur_cands_emb = T.cat(cands_emb, 0)

            while 1:
                qur_task = dataset.get_qur_inner_data()
                if qur_task is None:
                    break
                sim_cands = self.__predict_triplet(*qur_task)
                sim_cands = sim_cands.detach().data # sim_cands[0]: y_true sim, sim_cands[1:]: y_cands sim
                sim_cands = sim_cands.cpu().numpy()
                metrics.add(sim_cands)

            self.set_param(weights_original)
            if self.memory:
                self.qur_dscpr_emb = self.qur_cands_emb = self.qur_head_memory_dscpr_emb = self.qur_tail_memory_dscpr_emb = None
            else:
                self.qur_dscpr_emb = self.qur_cands_emb = None
            metrics.log_task_metric()
        metrics.log_overall_metric()

    # saving original parameters by using deepcopy(self.encoder.state_dict()) and load_state_dict() leads to GPU memory leak
    def copy_param(self):
        new_params = {}
        for name, module in self.modules.items():
            new_params[name] = module.copy_param()

        return new_params

    def set_param(self, new_params):
        for name, module_params in new_params.items():
            self.modules[name].set_param(module_params)


