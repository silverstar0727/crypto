import torch
from torch import nn
import numpy as np
import math

def t2v(tau, f, out_features, w, b, w0, b0, arg=None):
    if arg:
        v1 = f(torch.matmul(tau, w) + b, arg)
    else:
        #print(w.shape, t1.shape, b.shape)
        v1 = f(torch.matmul(tau, w) + b)
    v2 = torch.matmul(tau, w0) + b0
    #print(v1.shape)
    return torch.cat([v1, v2], -1)

class SineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(SineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features-1))
        self.b = nn.parameter.Parameter(torch.randn(out_features-1))
        self.f = torch.sin

    def forward(self, tau):
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)

class CosineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(CosineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features-1))
        self.b = nn.parameter.Parameter(torch.randn(out_features-1))
        self.f = torch.cos

    def forward(self, tau):
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)

class Time2Vec(nn.Module):
    def __init__(self, input_size, activation, hidden_dim, out_dim, batch_size, lstm_hidden_dim, lstm_layer):
        super(Time2Vec, self).__init__()
        self.batch_size = batch_size
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_layer = lstm_layer

        if activation == "sin":
            self.tsv = SineActivation(input_size, hidden_dim)
        elif activation == "cos":
            self.tsv = CosineActivation(input_size, hidden_dim)
        
        self.lstm = torch.nn.LSTM(
            input_size=hidden_dim, 
            hidden_size=lstm_hidden_dim, 
            num_layers=self.lstm_layer, 
            batch_first=True,
            bidirectional=False
        )
        self.out_layer = nn.Linear(lstm_hidden_dim, out_dim)
    
    def forward(self, x):
        x = self.tsv(x) # (b, w, h)
        output, (final_hidden_state, final_cell_state) = self.lstm(x)
        output = output[:,-1,:]
        x = self.out_layer(output)

        return x
