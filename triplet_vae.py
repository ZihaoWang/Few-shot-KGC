# -*- coding: utf-8 -*-
from sys import exit
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from basic_meta_learning_module import BasicMetaLearningModule

class TripletVAE(BasicMetaLearningModule):
    def __init__(self, hyp):
        super(TripletVAE, self).__init__(hyp)
        self.dim_input = hyp.encoder_num_conv_filter[-1]
        self.num_aug = hyp.num_aug
        self.num_conv_filter = hyp.vae_num_conv_filter
        self.dim_hidden = hyp.vae_dim_hidden
        self.prior_nn_num_layer = hyp.vae_prior_nn_num_layer
        self.prior_nn_dim_hidden = hyp.vae_prior_nn_dim_hidden

        self.dim_latent = hyp.vae_dim_latent

        self.use_prior = hyp.vae_use_prior
        self.prior_sigma_m = hyp.prior_sigma_m
        self.prior_sigma_s = hyp.prior_sigma_s
        self.lambda_kld = hyp.vae_lambda_kld
        self.lambda_reg = hyp.vae_lambda_reg
        self.normalization = hyp.vae_normalization
        self.act_func = T.tanh if hyp.vae_act_func == "tanh" else F.leaky_relu

        self.__init_params()

    def learn(self, dscp1, dscp2, dscpr):
        triplet = T.stack([dscp1, dscpr, dscp2], 1).transpose(1, 2) # shape: (num_shot, dim_input, 3)
        proposal_mu, proposal_sigma = self.__proposal_network(triplet)
        prior_mu, prior_sigma = self.__prior_network(dscpr)
        eps = T.randn(proposal_mu.shape, device = self.device)
        recon_triplet = self.__generative_network(proposal_mu, proposal_sigma, eps)
        
        log_proposal_sigma, log_prior_sigma = T.log(proposal_sigma), T.log(prior_sigma)
        loss_reconstruct = T.mean((triplet - recon_triplet) ** 2)
        kld = log_prior_sigma - log_proposal_sigma + 0.5 * (proposal_sigma ** 2 + (proposal_mu - prior_mu) ** 2) / prior_sigma ** 2 - 0.5
        kld = self.lambda_kld * T.mean(kld)
        reg = 0
        if self.use_prior:
            reg = prior_mu ** 2 / (2 * self.prior_sigma_m ** 2) - self.prior_sigma_s * (log_prior_sigma - prior_sigma)
            reg = self.lambda_reg * T.mean(reg)
        loss = loss_reconstruct + kld + reg

        loss_print = [["vae_reconstruct", loss_reconstruct.data.cpu().item()],
                ["vae_kld", kld.data.cpu().item()]]
        if self.use_prior:
            loss_print.append(["vae_regularization", reg.data.cpu().item()])
        
        return loss, loss_print

    def generate(self, dscpr):
        prior_mu, prior_sigma = self.__prior_network(dscpr)
        eps = T.randn((self.num_aug, self.dim_latent), device = self.device)
        gen_triplets = self.__generative_network(prior_mu, prior_sigma, eps)
        ent1, ent2 = gen_triplets[:, :, 0], gen_triplets[:, :, 2]

        return ent1, ent2

    def __generative_network(self, mu, sigma, eps):
        x = mu + sigma * eps

        for i in range(2):
            x = T.mm(x, self.params["generative_w_hidden{}".format(i)]) + self.params["generative_b_hidden{}".format(i)]
            x = self.norm(self.normalization, x, "generative", i)
            x = self.act_func(x)
        x = x.unsqueeze(2)
        x = F.conv_transpose1d(x, self.params["generative_w_deconv0"], self.params["generative_b_deconv0"])
        x = self.norm(self.normalization, x, "generative", 2)
        x = self.act_func(x)
        x = F.conv_transpose1d(x, self.params["generative_w_deconv1"], self.params["generative_b_deconv1"])
        x = self.act_func(x)

        return x

    def __proposal_network(self, x):
        for i in range(2):
            x = F.conv1d(x, self.params["proposal_w_conv{}".format(i)], self.params["proposal_b_conv{}".format(i)])
            x = self.norm(self.normalization, x, "proposal", i)
            x = self.act_func(x)
        x = x.squeeze(2)
        x = T.mm(x, self.params["proposal_w_hidden"]) + self.params["proposal_b_hidden"]
        x = self.norm(self.normalization, x, "proposal", 2)
        x = self.act_func(x)

        mu = T.mm(x, self.params["proposal_w_mu"]) + self.params["proposal_b_mu"]
        sigma = F.softplus(T.mm(x, self.params["proposal_w_sigma"]) + self.params["proposal_b_sigma"])
        return mu, sigma

    def __prior_network(self, x):
        for i in range(self.prior_nn_num_layer):
            x = T.mm(x, self.params["prior_w_hidden{}".format(i)]) + self.params["prior_b_hidden{}".format(i)]
            x = self.norm(self.normalization, x, "prior", i)
            x = self.act_func(x)
        mu = T.mm(x, self.params["prior_w_mu"]) + self.params["prior_b_mu"]
        sigma = F.softplus(T.mm(x, self.params["prior_w_sigma"]) + self.params["prior_b_sigma"])
        return mu, sigma

    def __init_params(self):
        # proposal network
        self.params["proposal_w_conv0"] = self.xavier(self.num_conv_filter, self.dim_input, 2)
        self.params["proposal_b_conv0"] = self.fill(0.0, self.num_conv_filter)
        self.add_norm_param(self.normalization, self.num_conv_filter, "proposal", 0)
        self.params["proposal_w_conv1"] = self.xavier(self.num_conv_filter, self.num_conv_filter, 2)
        self.params["proposal_b_conv1"] = self.fill(0.0, self.num_conv_filter)
        self.add_norm_param(self.normalization, self.num_conv_filter, "proposal", 1)
        self.params["proposal_w_hidden"] = self.xavier(self.num_conv_filter, self.dim_hidden)
        self.params["proposal_b_hidden"] = self.fill(0.0, self.dim_hidden)
        self.add_norm_param(self.normalization, self.dim_hidden, "proposal", 2)
        self.params["proposal_w_mu"] = self.xavier(self.dim_hidden, self.dim_latent)
        self.params["proposal_b_mu"] = self.fill(0.0, self.dim_latent)
        self.params["proposal_w_sigma"] = self.xavier(self.dim_hidden, self.dim_latent)
        self.params["proposal_b_sigma"] = self.fill(0.0, self.dim_latent)

        # prior network
        self.params["prior_w_hidden0"] = self.xavier(self.dim_input, self.prior_nn_dim_hidden[0])
        self.params["prior_b_hidden0"] = self.fill(0.0, self.prior_nn_dim_hidden[0])
        self.add_norm_param(self.normalization, self.prior_nn_dim_hidden[0], "prior", 0)
        for i in range(1, self.prior_nn_num_layer):
            self.params["prior_w_hidden{}".format(i)] = self.xavier(self.prior_nn_dim_hidden[i - 1], self.prior_nn_dim_hidden[i])
            self.params["prior_b_hidden{}".format(i)] = self.fill(0.0, self.prior_nn_dim_hidden[i])
            self.add_norm_param(self.normalization, self.prior_nn_dim_hidden[i], "prior", i)
        self.params["prior_w_mu"] = self.xavier(self.prior_nn_dim_hidden[-1], self.dim_latent)
        self.params["prior_b_mu"] = self.fill(0.0, self.dim_latent)
        self.params["prior_w_sigma"] = self.xavier(self.prior_nn_dim_hidden[-1], self.dim_latent)
        self.params["prior_b_sigma"] = self.fill(0.0, self.dim_latent)

        # generative network
        self.params["generative_w_hidden0"] = self.xavier(self.dim_latent, self.dim_hidden)
        self.params["generative_b_hidden0"] = self.fill(0.0, self.dim_hidden)
        self.add_norm_param(self.normalization, self.dim_hidden, "generative", 0)
        self.params["generative_w_hidden1"] = self.xavier(self.dim_hidden, self.num_conv_filter)
        self.params["generative_b_hidden1"] = self.fill(0.0, self.num_conv_filter)
        self.add_norm_param(self.normalization, self.num_conv_filter, "generative", 1)
        self.params["generative_w_deconv0"] = self.xavier(self.num_conv_filter, self.num_conv_filter, 2)
        self.params["generative_b_deconv0"] = self.fill(0.0, self.num_conv_filter)
        self.add_norm_param(self.normalization, self.num_conv_filter, "generative", 2)
        self.params["generative_w_deconv1"] = self.xavier(self.num_conv_filter, self.dim_input, 2)
        self.params["generative_b_deconv1"] = self.fill(0.0, self.dim_input)
