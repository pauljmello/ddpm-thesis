#  Author: Paul-Jason Mello
#  Date: June 5th, 2023

# General Libraries
import math
import torch
import torch.nn as nn


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        emb = math.log(10_000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class mlp(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, emb_dim, dropout_prob):
        super(mlp, self).__init__()

        self.time_emb = SinusoidalPosEmb(emb_dim)
        self.activation = nn.SiLU()

        self.layer_1 = nn.Linear(input_dim + emb_dim, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_3 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_4 = nn.Linear(hidden_dim, output_dim)

        self.dropout = nn.Dropout(dropout_prob)


    def forward(self, x, d):
        emb = self.time_emb(d)
        out = torch.cat([x, emb], dim=-1)

        out = self.layer_1(out)
        out = self.activation(out)
        out = self.dropout(out)

        out = self.layer_2(out)
        out = self.activation(out)
        out = self.dropout(out)

        out = self.layer_3(out)
        out = self.activation(out)
        out = self.dropout(out)

        out = self.layer_4(out)

        return out
