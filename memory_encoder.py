# -*- coding: utf-8 -*-
from sys import exit
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from basic_meta_learning_module import BasicMetaLearningModule

class MemoryEncoder(BasicMetaLearningModule):
    def __init__(self, hyp):
        super(MemoryEncoder, self).__init__(hyp)
        self.dict_size = hyp.dict_size
        self.dim_emb = hyp.dim_emb
        self.idx_word_pad = hyp.idx_word_pad

        self.num_cnn_layer = hyp.encoder_num_cnn_layer
        self.dim_conv_filter = hyp.encoder_dim_conv_filter
        self.num_conv_filter = hyp.encoder_num_conv_filter
        self.dim_pool_kernel = hyp.encoder_dim_pool_kernel
        self.normalization = hyp.encoder_normalization
        self.num_memory = hyp.encoder_num_memory
        self.num_head = hyp.encoder_num_head
        self.act_func = T.tanh if hyp.encoder_act_func == "tanh" else F.relu
        self.use_self_atten = hyp.encoder_self_atten

        assert self.dim_conv_filter % 2 != 0
        self.padding_size = self.dim_conv_filter // 2
        self.dim_head = []
        for i in range(len(self.num_head)):
            assert self.num_conv_filter[i] % self.num_head[i] == 0
            self.dim_head.append(self.num_conv_filter[i] // self.num_head[i])

        self.__init_params()

    def encode_rel(self, dscpr):
        r_emb = self.__rel_CNN(dscpr)

        if self.num_memory > 0:
            atten = F.cosine_similarity(self.params["w_head_memory"], r_emb.unsqueeze(0), 1) # shape: (num_memory)
            atten = atten.unsqueeze(0)
            atten = T.softmax(atten, 1).transpose(0, 1)
            head_memory_r = T.sum(atten * self.params["w_head_trans"], 0)

            atten = F.cosine_similarity(self.params["w_tail_memory"], r_emb.unsqueeze(0), 1) # shape: (num_memory)
            atten = atten.unsqueeze(0)
            atten = T.softmax(atten, 1).transpose(0, 1)
            tail_memory_r = T.sum(atten * self.params["w_tail_trans"], 0)

            return r_emb.unsqueeze(0), head_memory_r, tail_memory_r
        else:
            return r_emb.unsqueeze(0), r_emb, r_emb

    def __rel_CNN(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        x = F.embedding(x, self.params["w_emb"], self.idx_word_pad, 1.0)
        x = x.permute(0, 2, 1)
        x = F.conv1d(x, self.params["w_conv00"], self.params["b_conv00"], padding = self.padding_size)
        x = F.conv1d(x, self.params["w_conv01"], self.params["b_conv01"], padding = self.padding_size)
        self.norm(self.normalization, x, "encoder", 0)
        x = self.act_func(x)
        x = F.max_pool1d(x, self.dim_pool_kernel, self.dim_pool_kernel, ceil_mode = True) # len_x /= 2

        for i in range(1, self.num_cnn_layer):
            x = F.conv1d(x, self.params["w_conv{}0".format(i)], self.params["b_conv{}0".format(i)], padding = self.padding_size)
            x = F.conv1d(x, self.params["w_conv{}1".format(i)], self.params["b_conv{}1".format(i)], padding = self.padding_size)
            self.norm(self.normalization, x, "encoder", i)
            x = self.act_func(x)
            if i == self.num_cnn_layer - 1:
                x = T.mean(x, -1)
            else:
                x = F.max_pool1d(x, self.dim_pool_kernel, self.dim_pool_kernel, ceil_mode = True)

        x = x.squeeze()
        return x # shape: (num_conv_filter[-1])

    def forward(self, x, memory_r):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        x_emb = F.embedding(x, self.params["w_emb"], self.idx_word_pad, 1.0)
        x = x_emb.permute(0, 2, 1)
        
        x = self.__CNN(x, memory_r)
        x = self.__SelfAtten(x)

        return x # shape: (batch_size, num_conv_filter[-1])

    def __CNN(self, x, memory_r):
        x = F.conv1d(x, self.params["w_conv00"], self.params["b_conv00"], padding = self.padding_size)
        x = F.conv1d(x, self.params["w_conv01"], self.params["b_conv01"], padding = self.padding_size)
        self.norm(self.normalization, x, "encoder", 0)
        x = self.act_func(x)

        if self.num_memory > 0:
            x = x.transpose(1, 2)
            memory_atten = F.cosine_similarity(x, memory_r.unsqueeze(0).unsqueeze(0), 2)
            memory_atten = T.softmax(memory_atten, 1)
            x = x * memory_atten.unsqueeze(-1)
            x = x.transpose(1, 2)

        x = F.max_pool1d(x, self.dim_pool_kernel, self.dim_pool_kernel, ceil_mode = True) # len_x /= 2
        return x

    def __SelfAtten(self, x):
        batch_size = x.shape[0]
        for i in range(1, self.num_cnn_layer):
            x = F.conv1d(x, self.params["w_conv{}0".format(i)], self.params["b_conv{}0".format(i)], padding = self.padding_size)
            x = F.conv1d(x, self.params["w_conv{}1".format(i)], self.params["b_conv{}1".format(i)], padding = self.padding_size)
            self.norm(self.normalization, x, "encoder", i)
            x = self.act_func(x)
            
            if self.use_self_atten:
                len_x = x.shape[2]
                num_head, dim_head = self.num_head[i - 1], self.dim_head[i - 1]
                x = x.transpose(1, 2)
                x = x.view(batch_size, len_x, num_head, dim_head).transpose(1, 2)
                atten = T.matmul(x, x.transpose(-2, -1)) / (dim_head ** 0.5)
                atten = T.softmax(atten, -1) # shape: (batch_size, len_x, num_head[i - 1], num_head[i - 1])
                x = T.matmul(atten, x)
                x = x.transpose(1, 2).contiguous().view(batch_size, len_x, num_head * dim_head)
                x = x.transpose(1, 2)

            if i == self.num_cnn_layer - 1:
                x = T.mean(x, -1)
            else:
                x = F.max_pool1d(x, self.dim_pool_kernel, self.dim_pool_kernel, ceil_mode = True)

        return x

    def __init_params(self):
        w_emb = nn.init.xavier_uniform_(T.empty((self.dict_size, self.dim_emb), device = self.device))
        w_emb[self.idx_word_pad].fill_(0.0)
        self.params["w_emb"] = nn.Parameter(w_emb) # w_emb

        num_in_filter = self.dim_emb
        num_out_filter = self.num_conv_filter[0]
        self.params["w_conv00"] = self.xavier(num_out_filter, num_in_filter, self.dim_conv_filter)
        self.params["b_conv00"] = self.fill(0.0, num_out_filter)
        #'''
        num_in_filter = self.num_conv_filter[0]
        num_out_filter = self.num_conv_filter[0]
        self.params["w_conv01"] = self.xavier(num_out_filter, num_in_filter, self.dim_conv_filter)
        self.params["b_conv01"] = self.fill(0.0, num_out_filter)
        #'''

        self.add_norm_param(self.normalization, num_out_filter, "encoder", 0)

        if self.num_memory > 0:
            self.params["w_head_memory"] = self.unif(self.num_memory, num_out_filter)
            self.params["w_tail_memory"] = self.unif(self.num_memory, num_out_filter)
            self.params["w_head_trans"] = self.unif(self.num_memory, num_out_filter)
            self.params["w_tail_trans"] = self.unif(self.num_memory, num_out_filter)

        for i in range(1, self.num_cnn_layer):
            num_in_filter = self.num_conv_filter[i - 1]
            num_out_filter = self.num_conv_filter[i]
            # conv1d
            self.params["w_conv{}0".format(i)] = self.xavier(num_out_filter, num_in_filter, self.dim_conv_filter)
            self.params["b_conv{}0".format(i)] = self.fill(0.0, num_out_filter)
            #'''
            num_in_filter = self.num_conv_filter[i]
            num_out_filter = self.num_conv_filter[i]
            self.params["w_conv{}1".format(i)] = self.xavier(num_out_filter, num_in_filter, self.dim_conv_filter)
            self.params["b_conv{}1".format(i)] = self.fill(0.0, num_out_filter)
            #'''

            self.add_norm_param(self.normalization, num_out_filter, "encoder", i)


