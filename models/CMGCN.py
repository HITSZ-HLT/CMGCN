# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.dynamic_rnn import DynamicLSTM

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, text, adj):
        hidden = torch.matmul(text, self.weight)
        denom = torch.sum(adj, dim=2, keepdim=True) + 1
        output = torch.matmul(adj, hidden.float()) / denom
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class CMGCN(nn.Module):
    def __init__(self, bert, opt):
        super(CMGCN, self).__init__()
        self.opt = opt
        self.bert = bert
        
        self.gc1 = GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)
        self.gc2 = GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)
        #self.gc3 = GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)
        #self.gc4 = GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)
        
        self.text_lstm = DynamicLSTM(opt.bert_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.vit_fc = nn.Linear(768, 2*opt.hidden_dim)
        self.fc = nn.Linear(2*opt.hidden_dim, opt.polarities_dim)


    def forward(self, inputs):
        bert_indices, graph,box_vit = inputs
        bert_text_len = torch.sum(bert_indices != 0, dim=-1)
        encoder_layer, pooled_output = self.bert(bert_indices,  output_all_encoded_layers=False)
        text_out, (_, _) = self.text_lstm(encoder_layer, bert_text_len)

        box_vit = self.vit_fc(box_vit)
        features = torch.cat([text_out,box_vit],dim = 1)
        
        x = F.relu(self.gc1(features, graph))
        x = F.relu(self.gc2(x, graph))

        alpha_mat = torch.matmul(features, x.transpose(1, 2))
        alpha = F.softmax(alpha_mat.sum(1, keepdim=True), dim=2)
        x = torch.matmul(alpha, x).squeeze(1)
        output = self.fc(x)
        return output
