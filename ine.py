# -*- coding: utf-8 -*-
from __future__ import print_function, division

import random
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

teacher_model = None
teacher_noises = []    
device = 'cuda'

class EnsLinear(nn.Module):
    def __init__(self, k, in_dim, out_dim, batch_size, noise_wei):
        super(EnsLinear, self).__init__()
        self.split = in_dim > 256
        self.in_dim_unit = in_dim // k if self.split else in_dim
        self.fcs = nn.ModuleList([nn.Linear(self.in_dim_unit, out_dim).to(device) for _ in range(k)])
        global teacher_noises
        for i in range(k):
            noise = torch.randn((batch_size, out_dim)).to(device) * noise_wei
            teacher_noises.append(noise)

    def forward(self, x):
        out = []
        for i in range(len(self.fcs)):
            fc = self.fcs[i]
            if self.split:
                off_b = i * self.in_dim_unit
                off_e = (i + 1) * self.in_dim_unit
            else:
                off_b = 0
                off_e = self.in_dim_unit
            y = fc(x[:, off_b:off_e])
            y = y.unsqueeze(0)
            out.append(y)
        return out

def ine_loss(inputs, target_outputs, labels, args):
    loss_kd = torch.tensor(0.).to(device)

    T = args.ine_T
    source_outputs = teacher_model(inputs).detach()
    q = F.softmax(source_outputs / T, dim = 1)

    for i, ot in enumerate(target_outputs):
        ot = ot.squeeze()
        p = F.log_softmax(ot / T, dim = 1)
        kl_divergence = F.kl_div(p, q + teacher_noises[i][:p.shape[0], :], size_average = False) * (T ** 2) / p.shape[0]
        loss_kd += args.ine_kd_alpha * kl_divergence

    output = torch.sum(torch.cat(target_outputs, 0), 0)
    loss_ce = F.cross_entropy(output, labels)
    loss_ce = (1 - args.ine_kd_alpha) * loss_ce

    loss = (loss_ce + loss_kd) / len(target_outputs)
    if args.ine_noise > 1e-8:
        loss *= 0.01 / args.ine_noise
 
    return loss 

def ine_init(model, args, fea_dim, num_classes):
    global teacher_model
    teacher_model = copy.deepcopy(model)
    if args.model == 'VGG16':
        model.classifier = EnsLinear(args.ine_K, fea_dim, num_classes, args.batch_size, args.ine_noise)
    elif args.model == 'WideResNet28x10':
        model.linear = EnsLinear(args.ine_K, fea_dim, num_classes, args.batch_size, args.ine_noise)
    else:
        model.fc = EnsLinear(args.ine_K, fea_dim, num_classes, args.batch_size, args.ine_noise)

