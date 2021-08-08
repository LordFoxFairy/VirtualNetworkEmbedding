import torch
import torch.nn as nn
import torch.nn.functional as F

from layer.gat import GraphAttentionLayer


class GATModelVAE(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout):
        super(GATModelVAE, self).__init__()
        self.ga1 = GraphAttentionLayer(input_feat_dim, hidden_dim1, dropout, act=F.relu)
        self.ga2 = GraphAttentionLayer(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.ga3 = GraphAttentionLayer(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.dc = InnerProductDecoder(dropout, act=lambda x: x)

        self.w1 = torch.nn.Linear(hidden_dim2,hidden_dim2)
        self.w2 = torch.nn.Linear(hidden_dim2,input_feat_dim)

    def encode(self, x, adj):
        hidden1 = self.ga1(x, adj)
        return self.ga2(hidden1, adj), self.ga3(hidden1, adj)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, adj):
        mu, logvar = self.encode(x, adj)
        z = self.reparameterize(mu, logvar)
        z = self.w2(z)
        return self.dc(z)


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout, act=None):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        if self.act:
            return self.act(z)
        return F.softmax(z,dim=1)
