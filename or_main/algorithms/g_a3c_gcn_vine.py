import os
import torch_geometric
from or_main.algorithms.a_baseline import BaselineVNEAgent
from or_main.algorithms.model.A3C import A3C_Model
from or_main.common import utils, config

import torch
import numpy as np

from or_main.common.config import model_save_path
from or_main.main.a3c_gcn_train.vne_env_a3c_train import A3C_GCN_Action


class A3C_GCN_VNEAgent(BaselineVNEAgent):
    def __init__(
            self, local_model, beta, logger, time_window_size, agent_type, type_of_virtual_node_ranking,
            allow_embedding_to_same_substrate_node, max_embedding_path_length
    ):
        super(A3C_GCN_VNEAgent, self).__init__(
            logger, time_window_size, agent_type, type_of_virtual_node_ranking,
            allow_embedding_to_same_substrate_node, max_embedding_path_length
        )
        self.local_model = local_model
        self.beta = beta
        self.initial_s_CPU = []
        self.initial_s_bandwidth = []
        self.count_node_mapping = 0
        self.action_count = 0
        self.eligibility_trace = np.zeros(shape=(config.SUBSTRATE_NODES,))
        self.a3c_gcn_agent = A3C_Model(
            chev_conv_state_dim=config.NUM_SUBSTRATE_FEATURES, action_dim=config.SUBSTRATE_NODES
        )
        # self.a3c_gcn_agent = MLP_Model(
        #     chev_conv_state_dim=config.NUM_SUBSTRATE_FEATURES, action_dim=config.SUBSTRATE_NODES
        # )
        # self.new_model_path = os.path.join(model_save_path, "A3C_model_0421.pth")
        # self.a3c_gcn_agent.load_state_dict(torch.load(self.new_model_path))


    def get_node_action(self, state):
        action = A3C_GCN_Action()
        action.v_node = state.current_v_node

        action.s_node = self.local_model.select_node(
            substrate_features=state.substrate_features,
            substrate_edge_index=state.substrate_edge_index,
            vnr_features=state.vnr_features
        )

        return action

    def get_state_information(self, copied_substrate, vnr, already_embedding_s_nodes, current_v_node, current_v_cpu_demand, v_pending):
        # Substrate Initial State
        s_cpu_capacity = copied_substrate.initial_s_cpu_capacity
        s_bandwidth_capacity = copied_substrate.initial_s_node_total_bandwidth

        s_cpu_remaining = []
        s_bandwidth_remaining = []

        # S_cpu_Free, S_bw_Free
        for s_node, s_node_data in copied_substrate.net.nodes(data=True):
            s_cpu_remaining.append(s_node_data['CPU'])

            total_node_bandwidth = 0.0
            for link_id in copied_substrate.net[s_node]:
                total_node_bandwidth += copied_substrate.net[s_node][link_id]['bandwidth']

            s_bandwidth_remaining.append(total_node_bandwidth)

        # Generate substrate feature matrix
        substrate_features = []
        substrate_features.append(s_cpu_capacity)
        substrate_features.append(s_bandwidth_capacity)
        substrate_features.append(s_cpu_remaining)
        substrate_features.append(s_bandwidth_remaining)
        substrate_features.append(already_embedding_s_nodes)

        # Convert to the torch.tensor
        substrate_features = torch.tensor(substrate_features)
        substrate_features = torch.transpose(substrate_features, 0, 1)
        substrate_features = substrate_features.view(1, config.SUBSTRATE_NODES, config.NUM_SUBSTRATE_FEATURES)
        # substrate_features.size() --> (100, 5)

        # GCN for Feature Extract
        substrate_geometric_data = torch_geometric.utils.from_networkx(copied_substrate.net)

        vnr_features = []
        vnr_features.append(current_v_cpu_demand)
        vnr_features.append(sum((vnr.net[current_v_node][link_id]['bandwidth'] for link_id in vnr.net[current_v_node])))
        vnr_features.append(v_pending)
        vnr_features = torch.tensor(vnr_features).view(1, 1, 3)
        # substrate_features.size() --> (100, 5)
        # vnr_features.size()) --> (1, 3)

        return substrate_features, substrate_geometric_data.edge_index, vnr_features

    def find_substrate_nodes(self, copied_substrate, vnr):
        subset_S_per_v_node = {}
        embedding_s_nodes = {}
        already_embedding_s_nodes = []
        current_embedding_s_nodes = [0] * len(copied_substrate.net.nodes)

        # self.config.TYPE_OF_VIRTUAL_NODE_RANKING.TYPE_2
        sorted_v_nodes_with_node_ranking = utils.get_sorted_v_nodes_with_node_ranking(
            vnr=vnr, type_of_node_ranking=self.type_of_virtual_node_ranking
        )

        num_remain_v_node = 1
        for v_node_id, v_node_data, _ in sorted_v_nodes_with_node_ranking:
            v_cpu_demand = v_node_data['CPU']
            v_node_location = v_node_data['LOCATION']
            v_pending = len(sorted_v_nodes_with_node_ranking) - num_remain_v_node

            # Find the subset S of substrate nodes that satisfy restrictions and
            # available CPU capacity (larger than that specified by the request.)
            subset_S_per_v_node[v_node_id] = utils.find_subset_S_for_virtual_node(
                copied_substrate, v_cpu_demand, v_node_location, already_embedding_s_nodes
            )

            # Generate the state
            substrate_features, edge_index, vnr_features = self.get_state_information(
                copied_substrate,
                vnr,
                current_embedding_s_nodes,
                v_node_id,
                v_cpu_demand,
                v_pending
            )

            # select the node
            selected_s_node_id = self.a3c_gcn_agent.select_node(substrate_features, edge_index, vnr_features)

            if copied_substrate.net.nodes[selected_s_node_id]['CPU'] <= v_cpu_demand or \
                    selected_s_node_id in already_embedding_s_nodes:
                self.num_node_embedding_fails += 1
                msg = "VNR {0} REJECTED ({1}): 'no suitable SUBSTRATE NODE for nodal constraints: {2}' {3}".format(
                    vnr.id, self.num_node_embedding_fails, v_cpu_demand, vnr
                )
                self.logger.info("{0} {1}".format(utils.step_prefix(self.time_step), msg))
                return None

            assert selected_s_node_id != -1
            embedding_s_nodes[v_node_id] = (selected_s_node_id, v_cpu_demand)

            if not self.allow_embedding_to_same_substrate_node:
                already_embedding_s_nodes.append(selected_s_node_id)

            assert copied_substrate.net.nodes[selected_s_node_id]['CPU'] >= v_cpu_demand
            copied_substrate.net.nodes[selected_s_node_id]['CPU'] -= v_cpu_demand
            current_embedding_s_nodes[selected_s_node_id] = 1
            num_remain_v_node += 1


        return embedding_s_nodes


