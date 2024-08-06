################################################################################
# MIT License
#
# Copyright (c) 2018
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#


################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

################################################################################


class VanillaRNN(nn.Module):

    def __init__(
        self, seq_length, input_dim, num_hidden, num_classes, batch_size, device="cpu"
    ):
        super(VanillaRNN, self).__init__()
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.num_hidden = num_hidden
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.device = device

        self.Whx = torch.nn.Parameter(torch.randn(self.num_hidden, self.input_dim)).to(device)
        self.Whh = torch.nn.Parameter(torch.randn(self.num_hidden, self.num_hidden)).to(device)
        self.Why = torch.nn.Parameter(torch.randn(self.num_classes, self.num_hidden)).to(device)
        self.bh = torch.nn.Parameter(torch.zeros(self.num_hidden)).to(device)
        self.by = torch.nn.Parameter(torch.zeros(self.num_classes)).to(device)

    def forward(self, input, h_prev):
        h = h_prev.to(self.device)
        outputs = []  # Initialize outputs as an empty list
        for t in range(len(input)):
            x_t = input[t].to(self.device)
            h = torch.tanh(self.Whx @ x_t + self.Whh @ h + self.bh[:, None])
            y = self.Why @ h + self.by[:, None]
            p = torch.exp(y) / torch.sum(torch.exp(y), dim=0)
            outputs.append(p.T)
            
        outputs = torch.stack(outputs).to(self.device)
        return outputs, h




        
        
