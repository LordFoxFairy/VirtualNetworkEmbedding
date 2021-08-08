import copy

import gym
import networkx as nx
import torch
import torch_geometric

from or_main.common import config, utils
from or_main.environments.vne_env import Substrate, VNR


class A3C_GCN_State:
    def __init__(self, substrate_features, substrate_edge_index, current_v_node, vnr_features):
        self.substrate_features = substrate_features
        self.substrate_edge_index = substrate_edge_index
        self.current_v_node = current_v_node
        self.vnr_features = vnr_features

    def __str__(self):
        substrate_features_str = str(self.substrate_features)
        substrate_edge_index_str = str(self.substrate_edge_index)
        current_v_node_str = str(self.current_v_node)
        vnr_features_str = str(self.vnr_features)

        state_str = " ".join([substrate_features_str, substrate_edge_index_str, current_v_node_str, vnr_features_str])

        return state_str

    def __repr__(self):
        return self.__str__()


class A3C_GCN_Action:
    def __init__(self):
        self.v_node = None
        self.s_node = None

    def __str__(self):
        action_str = "[V_NODE {0:2}] [S_NODE {1:2}]".format(
            self.v_node, self.s_node
        )

        return action_str


class A3C_GCN_TRAIN_VNEEnvironment(gym.Env):
    def __init__(self, logger):
        self.logger = logger

        self.time_step = None
        self.episode_reward = None
        self.num_reset = 0

        self.revenue = None
        self.acceptance_ratio = None
        self.rc_ratio = None
        self.link_embedding_fails_against_total_fails_ratio = None

        self.total_arrival_vnrs = 0
        self.total_embedded_vnrs = 0

        self.substrate = Substrate()
        # self.copied_substrate = copy.deepcopy(self.substrate)
        self.vnrs = []
        self.vnr_idx = 0
        self.vnr = None
        self.v_node_embedding_success = []
        self.vnr_embedding_success_count = []
        self.already_embedded_v_nodes = []
        self.embedding_s_nodes = None

        self.previous_step_revenue = None
        self.previous_step_cost = None

        self.egb_trace = None
        self.decay_factor_for_egb_trace = 0.99

        self.current_embedding = None
        self.sorted_v_nodes = None
        self.current_v_node = None
        self.current_v_cpu_demand = None

    def reset(self):
        self.time_step = 0

        self.episode_reward = 0.0

        self.revenue = 0.0
        self.acceptance_ratio = 0.0
        self.rc_ratio = 0.0
        self.link_embedding_fails_against_total_fails_ratio = 0.0

        self.num_reset += 1
        # if self.num_reset % 1 == 0:
        # self.substrate = copy.deepcopy(self.copied_substrate)
        self.substrate = Substrate()
        self.vnr_idx = 0
        self.vnrs = []
        for idx in range(config.NUM_VNR_FOR_TRAIN):
            self.vnrs.append(
                VNR(
                    id=idx,
                    vnr_duration_mean_rate=config.VNR_DURATION_MEAN_RATE,
                    delay=config.VNR_DELAY,
                    time_step_arrival=0
                )
            )
        # self.vnrs = sorted(
        #     self.vnrs, key=lambda vnr: vnr.revenue, reverse=True
        # )
        self.vnr = self.vnrs[self.vnr_idx]
        self.v_node_embedding_success = []
        self.vnr_embedding_success_count = []

        self.already_embedded_v_nodes = []

        self.embedding_s_nodes = {}
        self.num_processed_v_nodes = 0
        self.previous_step_revenue = 0.0
        self.previous_step_cost = 0.0

        self.egb_trace = [1] * len(self.substrate.net.nodes)
        self.current_embedding = [0] * len(self.substrate.net.nodes)

        self.sorted_v_nodes = utils.get_sorted_v_nodes_with_node_ranking(
            vnr=self.vnr, type_of_node_ranking=config.TYPE_OF_VIRTUAL_NODE_RANKING.TYPE_2
        )

        self.current_v_node, current_v_node_data, _ = self.sorted_v_nodes[self.num_processed_v_nodes]
        self.current_v_cpu_demand = current_v_node_data['CPU']

        substrate_features, substrate_edge_index, vnr_features = self.get_state_information(
            self.current_v_node, self.current_v_cpu_demand
        )
        initial_state = A3C_GCN_State(substrate_features, substrate_edge_index, self.current_v_node, vnr_features)

        return initial_state

    def step(self, action: A3C_GCN_Action):
        self.time_step += 1
        self.vnr = self.vnrs[self.vnr_idx]

        embedding_success = True
        v_cpu_demand = None

        node_embedding_fail_conditions = [
            self.substrate.net.nodes[action.s_node]['CPU'] < self.vnr.net.nodes[action.v_node]['CPU'],
            self.current_embedding[action.s_node] == 1
        ]

        sum_v_bandwidth_demand = 0.0  # for r_c calculation
        sum_s_bandwidth_embedded = 0.0  # for r_c calculation

        if any(node_embedding_fail_conditions):
            embedding_success = False

        else:
            # Success for node embedding
            v_cpu_demand = self.vnr.net.nodes[action.v_node]['CPU']
            self.embedding_s_nodes[action.v_node] = action.s_node

            # Start to try link embedding
            for already_embedded_v_node in self.already_embedded_v_nodes:
                if self.vnr.net.has_edge(action.v_node, already_embedded_v_node):
                    v_bandwidth_demand = self.vnr.net[action.v_node][already_embedded_v_node]['bandwidth']

                    sum_v_bandwidth_demand += v_bandwidth_demand

                    subnet = nx.subgraph_view(
                        self.substrate.net,
                        filter_edge=lambda node_1_id, node_2_id: \
                            True if self.substrate.net.edges[(node_1_id, node_2_id)]['bandwidth'] >= v_bandwidth_demand else False
                    )

                    src_s_node = self.embedding_s_nodes[already_embedded_v_node]
                    dst_s_node = self.embedding_s_nodes[action.v_node]
                    if len(subnet.edges) == 0 or not nx.has_path(subnet, source=src_s_node, target=dst_s_node):
                        embedding_success = False
                        del self.embedding_s_nodes[action.v_node]
                        break
                    else:
                        MAX_K = 10
                        shortest_s_path = utils.k_shortest_paths(subnet, source=src_s_node, target=dst_s_node, k=MAX_K)[0]
                        if len(shortest_s_path) > config.MAX_EMBEDDING_PATH_LENGTH:
                            embedding_success = False
                            break
                        else:
                            # SUCCESS --> EMBED VIRTUAL LINK!
                            s_links_in_path = []
                            for node_idx in range(len(shortest_s_path) - 1):
                                s_links_in_path.append((shortest_s_path[node_idx], shortest_s_path[node_idx + 1]))

                            for s_link in s_links_in_path:
                                assert self.substrate.net.edges[s_link]['bandwidth'] >= v_bandwidth_demand
                                self.substrate.net.edges[s_link]['bandwidth'] -= v_bandwidth_demand
                                sum_s_bandwidth_embedded += v_bandwidth_demand

        # calculate r_s
        if embedding_success:
            r_s = self.substrate.net.nodes[action.s_node]['CPU'] / self.substrate.initial_s_cpu_capacity[
                action.s_node]
        else:
            r_s = 1.0

        if embedding_success:
            # ALL SUCCESS --> EMBED VIRTUAL NODE!
            assert self.substrate.net.nodes[action.s_node]['CPU'] >= v_cpu_demand
            self.substrate.net.nodes[action.s_node]['CPU'] -= v_cpu_demand
            self.current_embedding[action.s_node] = 1
            self.already_embedded_v_nodes.append(action.v_node)
            self.v_node_embedding_success.append(embedding_success)
        else:
            self.v_node_embedding_success.append(embedding_success)
            if action.v_node in self.embedding_s_nodes:
                del self.embedding_s_nodes[action.v_node]


        # 이 지점에서 self.num_processed_v_nodes += 1 매우 중요: 이후 next_state 및 reward 계산에 영향을 줌
        self.num_processed_v_nodes += 1

        reward = self.get_reward(
            embedding_success, v_cpu_demand, sum_v_bandwidth_demand, sum_s_bandwidth_embedded, action, r_s
        )

        done = False
        # if not embedding_success or self.num_processed_v_nodes == len(self.vnr.net.nodes):
        if self.num_processed_v_nodes == len(self.vnr.net.nodes):
            if sum(self.v_node_embedding_success) == len(self.vnr.net.nodes):
                self.vnr_embedding_success_count.append(1)
            else:
                self.vnr_embedding_success_count.append(0)

            if self.vnr_idx == len(self.vnrs) - 1 or sum(self.vnr_embedding_success_count[-3:]) == 0:
            # if self.vnr_idx == len(self.vnrs) - 1:
                # print(self.vnr_embedding_success_count)
                print("The number of embedded success vnr: ", sum(self.vnr_embedding_success_count))
                done = True
                next_state = A3C_GCN_State(None, None, None, None)
            else:
                self.vnr_idx += 1
                self.vnr = self.vnrs[self.vnr_idx]
                self.sorted_v_nodes = utils.get_sorted_v_nodes_with_node_ranking(
                    vnr=self.vnr, type_of_node_ranking=config.TYPE_OF_VIRTUAL_NODE_RANKING.TYPE_2
                )
                self.num_processed_v_nodes = 0
                self.current_v_node, current_v_node_data, _ = self.sorted_v_nodes[self.num_processed_v_nodes]
                self.current_v_cpu_demand = current_v_node_data['CPU']

                substrate_features, substrate_edge_index, vnr_features = self.get_state_information(
                    self.current_v_node, self.current_v_cpu_demand
                )
                next_state = A3C_GCN_State(substrate_features, substrate_edge_index, self.current_v_node, vnr_features)
                self.already_embedded_v_nodes = []
                self.current_embedding = [0] * len(self.substrate.net.nodes)
                self.v_node_embedding_success = []

        else:
            self.current_v_node, current_v_node_data, _ = self.sorted_v_nodes[self.num_processed_v_nodes]
            self.current_v_cpu_demand = current_v_node_data['CPU']

            substrate_features, substrate_edge_index, vnr_features = self.get_state_information(
                self.current_v_node, self.current_v_cpu_demand
            )
            next_state = A3C_GCN_State(substrate_features, substrate_edge_index, self.current_v_node, vnr_features)

        info = {}

        return next_state, reward, done, info

    def get_state_information(self, current_v_node, current_v_cpu_demand):
        # Substrate Initial State
        s_cpu_capacity = self.substrate.initial_s_cpu_capacity
        s_bandwidth_capacity = self.substrate.initial_s_node_total_bandwidth

        s_cpu_remaining = []
        s_bandwidth_remaining = []

        # S_cpu_Free, S_bw_Free
        for s_node, s_node_data in self.substrate.net.nodes(data=True):
            s_cpu_remaining.append(s_node_data['CPU'])

            total_node_bandwidth = 0.0
            for link_id in self.substrate.net[s_node]:
                total_node_bandwidth += self.substrate.net[s_node][link_id]['bandwidth']

            s_bandwidth_remaining.append(total_node_bandwidth)

        assert len(s_cpu_capacity) == len(s_bandwidth_capacity) == len(s_cpu_remaining) == len(s_bandwidth_remaining) == len(self.current_embedding)

        # Generate substrate feature matrix
        substrate_features = []
        substrate_features.append(s_cpu_capacity)
        substrate_features.append(s_bandwidth_capacity)
        substrate_features.append(s_cpu_remaining)
        substrate_features.append(s_bandwidth_remaining)
        substrate_features.append(self.current_embedding)

        # Convert to the torch.tensor
        substrate_features = torch.tensor(substrate_features)
        substrate_features = torch.transpose(substrate_features, 0, 1)
        substrate_features = substrate_features.view(1, config.SUBSTRATE_NODES, config.NUM_SUBSTRATE_FEATURES)
        # substrate_features.size() --> (1, 100, 5)

        # GCN for Feature Extract
        substrate_geometric_data = torch_geometric.utils.from_networkx(self.substrate.net)

        vnr_features = []
        vnr_features.append(current_v_cpu_demand)
        vnr_features.append(sum((self.vnr.net[current_v_node][link_id]['bandwidth'] for link_id in self.vnr.net[current_v_node])))
        vnr_features.append(len(self.sorted_v_nodes) - self.num_processed_v_nodes)
        vnr_features = torch.tensor(vnr_features).view(1, 1, 3)

        # substrate_features.size() --> (1, 100, 5)
        # vnr_features.size()) --> (1, 3)

        return substrate_features, substrate_geometric_data.edge_index, vnr_features

    def get_reward(self, embedding_success, v_cpu_demand, sum_v_bandwidth_demand, sum_s_bandwidth_embedded, action, r_s):
        gamma_action = self.num_processed_v_nodes / len(self.vnr.net.nodes)
        r_a = 100 * gamma_action if embedding_success else -100 * gamma_action

        # calculate r_c
        if embedding_success:
            step_revenue = v_cpu_demand + sum_v_bandwidth_demand
            step_cost = v_cpu_demand + sum_s_bandwidth_embedded
            delta_revenue = step_revenue - self.previous_step_revenue
            delta_cost = step_cost - self.previous_step_cost
            if delta_cost == 0.0:
                r_c = 1.0
            else:
                r_c = delta_revenue / delta_cost
            self.previous_step_revenue = step_revenue
            self.previous_step_cost = step_cost
        else:
            r_c = 1.0

        # calculate eligibility trace
        for s_node in self.substrate.net.nodes:
            if action.s_node == s_node:
                self.egb_trace[s_node] = self.decay_factor_for_egb_trace * (self.egb_trace[s_node] + 1)
            else:
                self.egb_trace[s_node] = self.decay_factor_for_egb_trace * self.egb_trace[s_node]

        reward = r_a * r_c * r_s / (self.egb_trace[action.s_node] + 1e-6)

        return reward
