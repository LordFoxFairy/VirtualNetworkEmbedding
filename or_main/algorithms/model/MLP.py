import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import gym
import os

from algorithms.model.utils import set_init
from torch_geometric.nn import GCNConv, ChebConv

from common import config


class MLP_Model(nn.Module):
    def __init__(self, chev_conv_state_dim, action_dim):
        super(MLP_Model, self).__init__()
        self.substrate_state = 0
        self.substrate_edge_index = 0
        self.v_cpu_demand_t = 0
        self.v_bw_demand_t = 0
        self.num_pending_v_nodes_t = 0

        self.actor_conv = nn.Linear(chev_conv_state_dim, 60)
        self.critic_conv = nn.Linear(chev_conv_state_dim, 60)

        self.actor_fc = nn.Linear(6003, action_dim)
        self.critic_fc = nn.Linear(6003, 1)

        set_init([self.actor_conv, self.critic_conv, self.actor_fc, self.critic_fc])
        self.distribution = torch.distributions.Categorical

    def forward(self, substrate_features, substrate_edge_index, vnr_features):
        gcn_embedding_actor = self.actor_conv(substrate_features)
        gcn_embedding_actor = gcn_embedding_actor.tanh()
        gcn_embedding_actor = torch.flatten(gcn_embedding_actor).unsqueeze(dim=0)

        gcn_embedding_critic = self.critic_conv(substrate_features)
        gcn_embedding_critic = gcn_embedding_critic.tanh()
        gcn_embedding_critic = torch.flatten(gcn_embedding_critic).unsqueeze(dim=0)

        # gcn_embedding_actor.shape: (1, 6000)
        # gcn_embedding_critic.shape: (1, 6000)

        concatenated_state_actor = torch.cat([gcn_embedding_actor, vnr_features], dim=1)
        concatenated_state_critic = torch.cat([gcn_embedding_critic, vnr_features], dim=1)

        # concatenated_state_actor.shape: (1, 6003)
        # concatenated_state_critic.shape: (1, 6003)

        logits = self.actor_fc(concatenated_state_actor)
        values = self.critic_fc(concatenated_state_critic)

        return logits, values

    def select_node(self, substrate_features, substrate_edge_index, vnr_features):
        self.eval()

        logits, values = self.forward(substrate_features, substrate_edge_index, vnr_features)
        probs = F.softmax(logits, dim=1).data
        m = self.distribution(probs)

        return m.sample().numpy()[0]

    def loss_func(self, substrate_features, substrate_edge_index, vnr_features, action, v_t):
        self.train()
        logits, values = self.forward(substrate_features, substrate_edge_index, vnr_features)
        td = v_t - values
        c_loss = td.pow(2)

        probs = F.softmax(logits, dim=1)
        m = self.distribution(probs)
        exp_v = m.log_prob(action) * td.detach().squeeze()
        a_loss = -exp_v

        total_loss = (c_loss + a_loss).mean()

        return total_loss, c_loss.item(), -1.0 * a_loss.item()
