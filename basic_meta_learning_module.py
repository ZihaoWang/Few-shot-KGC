# -*- coding: utf-8 -*-
from sys import exit
import torch as T
import torch.nn as nn
import torch.nn.functional as F

class BasicMetaLearningModule(nn.Module):
    def __init__(self, hyp):
        super(BasicMetaLearningModule, self).__init__()
        self.device = hyp.device
        self.params = nn.ParameterDict()

    # saving original parameters by using deepcopy(self.encoder.state_dict()) and load_state_dict() leads to GPU memory leak
    def copy_param(self):
        new_params = {}
        for name, param in self.params.items():
            new_params[name] = T.clone(param)

        return new_params

    def set_param(self, new_params):
        for name, param in new_params.items():
            self.params[name].data = self.params[name].data.copy_(param)

    def xavier(self, *shape):
        return nn.Parameter(nn.init.xavier_uniform_(T.empty(shape, device = self.device)))

    def unif(self, *shape):
        return nn.Parameter(nn.init.uniform_(T.empty(shape, device = self.device), -0.1, 0.1))

    def fill(self, value, *shape):
        return nn.Parameter(T.empty(shape, device = self.device).fill_(value))

    def add_norm_param(self, norm_type, dim, prefix, postfix):
        if norm_type == "instance_norm":
            self.params["{}_mean_norm{}".format(prefix, postfix)] = self.fill(0.0, dim)
            self.params["{}_var_norm{}".format(prefix, postfix)] = self.fill(1.0, dim)
            self.params["{}_mean_norm{}".format(prefix, postfix)].requires_grad = False
            self.params["{}_var_norm{}".format(prefix, postfix)].requires_grad = False
            self.params["{}_w_norm{}".format(prefix, postfix)] = self.unif(dim)
            self.params["{}_b_norm{}".format(prefix, postfix)] = self.fill(0.0, dim)
        else:
            pass

    def norm(self, norm_type, x, prefix, postfix):
        if norm_type == "instance_norm":
            return F.instance_norm(x,
                    self.params["{}_mean_norm{}".format(prefix, postfix)],
                    self.params["{}_var_norm{}".format(prefix, postfix)],
                    self.params["{}_w_norm{}".format(prefix, postfix)],
                    self.params["{}_b_norm{}".format(prefix, postfix)])
        else:
            return x
