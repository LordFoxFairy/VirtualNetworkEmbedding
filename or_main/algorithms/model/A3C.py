import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import gym
import os

from or_main.common import config

from or_main.algorithms.model.utils import set_init
from torch_geometric.nn import GCNConv, ChebConv

from or_main.common import config


class A3C_Model(nn.Module):
    def __init__(self, chev_conv_state_dim, action_dim):
        super(A3C_Model, self).__init__()
        self.substrate_state = 0
        self.substrate_edge_index = 0
        self.v_cpu_demand_t = 0
        self.v_bw_demand_t = 0
        self.num_pending_v_nodes_t = 0

        self.actor_conv = ChebConv(in_channels=chev_conv_state_dim, out_channels=3, K=3)
        self.critic_conv = ChebConv(in_channels=chev_conv_state_dim, out_channels=3, K=3)

        self.actor_vnr_1_fc = nn.Linear(1, 3)
        self.actor_vnr_2_fc = nn.Linear(1, 3)
        self.actor_vnr_3_fc = nn.Linear(1, 3)

        self.critic_vnr_1_fc = nn.Linear(1, 3)
        self.critic_vnr_2_fc = nn.Linear(1, 3)
        self.critic_vnr_3_fc = nn.Linear(1, 3)

        self.actor_fc = nn.Linear((config.SUBSTRATE_NODES + 3) * 3, action_dim)
        self.critic_fc = nn.Linear((config.SUBSTRATE_NODES + 3) * 3, 1)

        set_init([
                self.actor_conv, self.critic_conv,
                self.actor_vnr_1_fc, self.actor_vnr_2_fc, self.actor_vnr_2_fc,
                self.critic_vnr_1_fc, self.critic_vnr_2_fc, self.critic_vnr_3_fc,
                self.actor_fc, self.critic_fc
            ])
        self.distribution = torch.distributions.Categorical

    def forward(self, substrate_features, substrate_edge_index, vnr_features):
        # Actor
        gcn_embedding_actor = self.actor_conv(substrate_features, substrate_edge_index)
        gcn_embedding_actor = gcn_embedding_actor.tanh()

        vnr_features = torch.as_tensor(vnr_features, dtype=torch.float)
        vnr_features = vnr_features.view(-1, 3)
        vnr_output_1_actor = self.actor_vnr_1_fc(vnr_features[:, 0].view(-1, 1))
        vnr_output_2_actor = self.actor_vnr_2_fc(vnr_features[:, 1].view(-1, 1))
        vnr_output_3_actor = self.actor_vnr_3_fc(vnr_features[:, 2].view(-1, 1))
        vnr_output_1_actor = vnr_output_1_actor.view(-1, 1, 3)
        vnr_output_2_actor = vnr_output_2_actor.view(-1, 1, 3)
        vnr_output_3_actor = vnr_output_3_actor.view(-1, 1, 3)

        final_embedding_actor = torch.cat(
            [gcn_embedding_actor, vnr_output_1_actor, vnr_output_2_actor, vnr_output_3_actor], dim=1
        )
        gcn_embedding_actor = torch.flatten(final_embedding_actor).unsqueeze(dim=0)
        gcn_embedding_actor = gcn_embedding_actor.view(-1, (config.SUBSTRATE_NODES + 3) * 3)

        # Critic
        gcn_embedding_critic = self.critic_conv(substrate_features, substrate_edge_index)
        gcn_embedding_critic = gcn_embedding_critic.tanh()
        vnr_output_1_critic = self.critic_vnr_1_fc(vnr_features[:, 0].view(-1, 1))
        vnr_output_2_critic = self.critic_vnr_2_fc(vnr_features[:, 1].view(-1, 1))
        vnr_output_3_critic = self.critic_vnr_3_fc(vnr_features[:, 2].view(-1, 1))
        vnr_output_1_critic = vnr_output_1_critic.view(-1, 1, 3)
        vnr_output_2_critic = vnr_output_2_critic.view(-1, 1, 3)
        vnr_output_3_critic = vnr_output_3_critic.view(-1, 1, 3)

        final_embedding_critic = torch.cat(
            [gcn_embedding_critic, vnr_output_1_critic, vnr_output_2_critic, vnr_output_3_critic], dim=1
        )
        gcn_embedding_critic = torch.flatten(final_embedding_critic).unsqueeze(dim=0)
        gcn_embedding_critic = gcn_embedding_critic.view(-1, (config.SUBSTRATE_NODES + 3) * 3)

        logits = self.actor_fc(gcn_embedding_actor)
        values = self.critic_fc(gcn_embedding_critic)

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
        c_loss_mean = c_loss.mean()
        a_loss_mean = a_loss.mean()

        return total_loss, c_loss_mean.item(), -1.0 * a_loss_mean.item()
