from or_main.algorithms.a_baseline import BaselineVNEAgent
from or_main.common import utils
import networkx as nx
from networkx.algorithms.flow import shortest_augmenting_path
import pulp as plp
from collections import defaultdict

import pandas as pd

import warnings

warnings.filterwarnings(action='ignore')


class DeterministicVNEAgent(BaselineVNEAgent):
    def __init__(
            self, logger, time_window_size, agent_type, type_of_virtual_node_ranking,
            allow_embedding_to_same_substrate_node, max_embedding_path_length, substrate_nodes
    ):
        super(DeterministicVNEAgent, self).__init__(
            logger, time_window_size, agent_type, type_of_virtual_node_ranking,
            allow_embedding_to_same_substrate_node, max_embedding_path_length
        )
        self.substrate_nodes = substrate_nodes

    @staticmethod
    def change_to_augmented_substrate(copied_substrate, vnr, substrate_nodes):
        for v_node_id, v_node_data in vnr.net.nodes(data=True):
            v_cpu_demand = v_node_data['CPU']
            v_node_location = v_node_data['LOCATION']

            # Meta node add
            meta_node_id = v_node_id + substrate_nodes
            copied_substrate.net.add_node(meta_node_id)
            copied_substrate.net.nodes[meta_node_id]['CPU'] = v_cpu_demand
            copied_substrate.net.nodes[meta_node_id]['LOCATION'] = v_node_location

            # Meta edge add
            for a_node_id, a_node_data, in copied_substrate.net.nodes(data=True):
                a_cpu_demand = a_node_data['CPU']
                a_node_location = a_node_data['LOCATION']
                if v_node_location == a_node_location and a_node_id < substrate_nodes:
                    copied_substrate.net.add_edge(meta_node_id, a_node_id)
                    copied_substrate.net.edges[meta_node_id, a_node_id].update({'bandwidth': 1000000})

    @staticmethod
    def revoke_from_augmented_substrate(copied_substrate, vnr, substrate_nodes):
        for v_node_id, v_node_data in vnr.net.nodes(data=True):
            # Meta node add
            meta_node_id = v_node_id + substrate_nodes
            copied_substrate.net.remove_node(meta_node_id)

    def find_substrate_nodes(self, copied_substrate, vnr):
        '''
        Execute Step 1
        :param copied_substrate: copied substrate network
        :param vnr: virtual network request
        :return: embedded substrate nodes
        '''
        subset_S_per_v_node = {}
        embedding_s_nodes = {}
        already_embedding_s_nodes = []

        # Generate the augmented substrate network with location info.
        self.change_to_augmented_substrate(copied_substrate, vnr, self.substrate_nodes)

        opt_lp_f_vars, opt_lp_x_vars = self.calculate_LP_variables(copied_substrate, vnr)

        for v_node_id, v_node_data in vnr.net.nodes(data=True):
            v_cpu_demand = v_node_data['CPU']
            v_node_location = v_node_data['LOCATION']

            # Find the subset S of substrate nodes that satisfy restrictions and
            # available CPU capacity (larger than that specified by the request.)
            subset_S_per_v_node[v_node_id] = utils.find_subset_S_for_virtual_node(
                copied_substrate, v_cpu_demand, v_node_location, already_embedding_s_nodes
            )

            # selected_s_node_id = max(
            #     subset_S_per_v_node[v_node_id],
            #     key=lambda s_node_id:
            #         sum(opt_lp_f_vars[(opt_lp_f_vars['u'] == s_node_id) &
            #                           (opt_lp_f_vars['v'] == v_node_id + self.substrate_nodes)]['solution_value'].values +
            #             opt_lp_f_vars[(opt_lp_f_vars['u'] == v_node_id + self.substrate_nodes) &
            #                           (opt_lp_f_vars['v'] == s_node_id)]['solution_value'].values
            #             ) *
            #         opt_lp_x_vars[(opt_lp_x_vars['u'] == s_node_id) &
            #                       (opt_lp_x_vars['v'] == v_node_id + self.substrate_nodes)]['solution_value'].values,
            #     default=None
            # )

            selected_s_node_id = max(
                subset_S_per_v_node[v_node_id],
                key=lambda s_node_id:
                sum(opt_lp_f_vars[(opt_lp_f_vars['u'] == s_node_id) &
                                  (opt_lp_f_vars['v'] == v_node_id + self.substrate_nodes)]['solution_value'].values +
                    opt_lp_f_vars[(opt_lp_f_vars['u'] == v_node_id + self.substrate_nodes) &
                                  (opt_lp_f_vars['v'] == s_node_id)]['solution_value'].values) *
                    opt_lp_x_vars[(opt_lp_x_vars['u'] == s_node_id) &
                                  (opt_lp_x_vars['v'] == v_node_id + self.substrate_nodes)]['solution_value'].values[0],
                default=None
            )

            if selected_s_node_id is None:
                self.num_node_embedding_fails += 1
                msg = "VNR {0} REJECTED ({1}): 'no suitable SUBSTRATE NODE for nodal constraints: {2}' {3}".format(
                    vnr.id, self.num_node_embedding_fails, v_cpu_demand, vnr
                )
                self.logger.info("{0} {1}".format(utils.step_prefix(self.time_step), msg))
                self.revoke_from_augmented_substrate(copied_substrate, vnr, self.substrate_nodes)
                return None

            assert selected_s_node_id != -1
            embedding_s_nodes[v_node_id] = (selected_s_node_id, v_cpu_demand)
            if not self.allow_embedding_to_same_substrate_node:
                already_embedding_s_nodes.append(selected_s_node_id)

            assert copied_substrate.net.nodes[selected_s_node_id]['CPU'] >= v_cpu_demand
            copied_substrate.net.nodes[selected_s_node_id]['CPU'] -= v_cpu_demand

        self.revoke_from_augmented_substrate(copied_substrate, vnr, self.substrate_nodes)

        return embedding_s_nodes

    def find_substrate_path(self, copied_substrate, vnr, embedding_s_nodes):
        embedding_s_paths = {}
        directed_copied_substrate = copied_substrate.net.to_directed()

        # mapping the virtual nodes and substrate_net nodes
        for src_v_node, dst_v_node, edge_data in vnr.net.edges(data=True):
            v_link = (src_v_node, dst_v_node)
            src_s_node = embedding_s_nodes[src_v_node][0]
            dst_s_node = embedding_s_nodes[dst_v_node][0]
            v_bandwidth_demand = edge_data['bandwidth']

            if src_s_node == dst_s_node:
                s_links_in_path = []
                embedding_s_paths[v_link] = (s_links_in_path, v_bandwidth_demand)
            else:
                subnet = nx.subgraph_view(
                    copied_substrate.net,
                    filter_edge=lambda node_1_id, node_2_id: \
                        True if copied_substrate.net.edges[(node_1_id, node_2_id)][
                                    'bandwidth'] >= v_bandwidth_demand else False
                )

                # Just for assertion
                # for u, v, a in subnet.edges(data=True):
                #     assert a["bandwidth"] >= v_bandwidth_demand

                if len(subnet.edges) == 0 or not nx.has_path(subnet, source=src_s_node, target=dst_s_node):
                    self.num_link_embedding_fails += 1
                    msg = "VNR {0} REJECTED ({1}): 'no suitable LINK for bandwidth demand: {2} {3}".format(
                        vnr.id, self.num_link_embedding_fails, v_bandwidth_demand, vnr
                    )
                    self.logger.info("{0} {1}".format(utils.step_prefix(self.time_step), msg))
                    return None

                MAX_K = 1

                # shortest_s_path = utils.k_shortest_paths(subnet, source=src_s_node, target=dst_s_node, k=MAX_K)[0]
                # https://networkx.org/documentation/stable//reference/algorithms/generated/networkx.algorithms.flow.shortest_augmenting_path.html
                residual_network = shortest_augmenting_path(directed_copied_substrate, src_s_node, dst_s_node,
                                                            capacity='bandwidth',
                                                            cutoff=v_bandwidth_demand)
                s_links_in_path = []
                path = []
                for src_r_node, dst_r_node, r_edge_data in residual_network.edges(data=True):
                    if r_edge_data['flow'] > 0:
                        s_links_in_path.append((src_r_node, dst_r_node))

                # s_links_in_path = []
                # for node_idx in range(len(shortest_s_path) - 1):
                #     s_links_in_path.append((shortest_s_path[node_idx], shortest_s_path[node_idx + 1]))

                for s_link in s_links_in_path:
                    # assert copied_substrate.net.edges[s_link]['bandwidth'] >= v_bandwidth_demand
                    copied_substrate.net.edges[s_link]['bandwidth'] -= v_bandwidth_demand

                embedding_s_paths[v_link] = (s_links_in_path, v_bandwidth_demand)

        return embedding_s_paths

    def calculate_LP_variables(self, augmented_substrate, vnr):
        num_nodes = len(list(augmented_substrate.net.nodes))
        edges_bandwidth = [[0] * num_nodes for _ in range(num_nodes)]
        a_nodes_id = []
        s_nodes_id = []
        meta_nodes_id = []
        nodes_CPU = []
        v_flow_id = []
        v_flow_start = []
        v_flow_end = []
        v_flow_demand = []
        location_ids = defaultdict(list)
        meta_nodes_location = {}

        for a_edge_src, a_edge_dst, a_edge_data in augmented_substrate.net.edges(data=True):
            edges_bandwidth[a_edge_src][a_edge_dst] = a_edge_data['bandwidth']
            edges_bandwidth[a_edge_dst][a_edge_src] = a_edge_data['bandwidth']

        for a_node_id, a_node_data in augmented_substrate.net.nodes(data=True):
            a_nodes_id.append(a_node_id)
            nodes_CPU.append(a_node_data['CPU'])
            if a_node_id >= self.substrate_nodes:
                meta_nodes_id.append(a_node_id)
                meta_nodes_location[a_node_id] = a_node_data['LOCATION']
            else:
                s_nodes_id.append(a_node_id)
                location_ids[a_node_data['LOCATION']].append(a_node_id)

        id_idx = 0
        for v_edge_src, v_edge_dst, v_edge_data in vnr.net.edges(data=True):
            v_flow_id.append(id_idx)
            v_flow_start.append(v_edge_src + self.substrate_nodes)
            v_flow_end.append(v_edge_dst + self.substrate_nodes)
            v_flow_demand.append(v_edge_data['bandwidth'])
            id_idx += 1

        # f_vars
        f_vars = {
            (i, u, v): plp.LpVariable(
                cat=plp.LpContinuous,
                lowBound=0,
                name="f_{0}_{1}_{2}".format(i, u, v)
            )
            for i in v_flow_id for u in a_nodes_id for v in a_nodes_id
        }

        # x_vars
        x_vars = {(u, v):
            plp.LpVariable(
                cat=plp.LpContinuous,
                lowBound=0, upBound=1,
                name="x_{0}_{1}".format(u, v)
            )
            for u in a_nodes_id for v in a_nodes_id
        }

        opt_model = plp.LpProblem(name="MIP Model", sense=plp.LpMinimize)

        # Objective function
        opt_model += sum(1 / (edges_bandwidth[u][v] + 0.000001) *
                         sum(f_vars[i, u, v] for i in v_flow_id)
                         for u in s_nodes_id for v in s_nodes_id) + \
                     sum(1 / (nodes_CPU[w] + 0.000001) *
                         sum(x_vars[m, w] * nodes_CPU[m]
                             for m in meta_nodes_id) for w in s_nodes_id)

        # Capacity constraint 1
        for u in a_nodes_id:
            for v in a_nodes_id:
                opt_model += sum(f_vars[i, u, v] + f_vars[i, v, u] for i in v_flow_id) <= edges_bandwidth[u][v]

        # Capacity constraint 2
        for m in meta_nodes_id:
            for w in s_nodes_id:
                opt_model += nodes_CPU[w] >= x_vars[m, w] * nodes_CPU[m]

        # Flow constraints 1
        for i in v_flow_id:
            for u in s_nodes_id:
                opt_model += sum(f_vars[i, u, w] for w in a_nodes_id) - \
                             sum(f_vars[i, w, u] for w in a_nodes_id) == 0

        # Flow constraints 2
        for i in v_flow_id:
            for fs in v_flow_start:
                opt_model += sum(f_vars[i, fs, w] for w in a_nodes_id) - \
                             sum(f_vars[i, w, fs] for w in a_nodes_id) == v_flow_demand[i]

        # Flow constraints 3
        for i in v_flow_id:
            for fe in v_flow_end:
                opt_model += sum(f_vars[i, fe, w] for w in a_nodes_id) - \
                             sum(f_vars[i, w, fe] for w in a_nodes_id) == -1 * v_flow_demand[i]

        # Meta constraint 1
        for w in s_nodes_id:
            opt_model += sum(x_vars[m, w] for m in meta_nodes_id) <= 1

        # Meta constraint 2
        for u in a_nodes_id:
            for v in a_nodes_id:
                opt_model += x_vars[u, v] == x_vars[v, u]

        # Meta constraint 3
        for m in meta_nodes_id:
            opt_model += sum(x_vars[m, w] for w in location_ids[meta_nodes_location[m]]) == 1

        # for minimization
        # solve VNE_LP_RELAX
        opt_model.solve(plp.PULP_CBC_CMD(msg=0))

        # for v in opt_model.variables():
        #     if v.varValue > 0:
        #         print(v.name, "=", v.varValue)

        # make the DataFrame for f_vars and x_vars
        opt_lp_f_vars = pd.DataFrame.from_dict(f_vars, orient="index", columns=['variable_object'])
        opt_lp_f_vars.index = pd.MultiIndex.from_tuples(opt_lp_f_vars.index, names=["i", "u", "v"])
        opt_lp_f_vars.reset_index(inplace=True)
        opt_lp_f_vars["solution_value"] = opt_lp_f_vars["variable_object"].apply(lambda item: item.varValue)
        opt_lp_f_vars.drop(columns=["variable_object"], inplace=True)

        opt_lp_x_vars = pd.DataFrame.from_dict(x_vars, orient="index", columns=['variable_object'])
        opt_lp_x_vars.index = pd.MultiIndex.from_tuples(opt_lp_x_vars.index, names=["u", "v"])
        opt_lp_x_vars.reset_index(inplace=True)
        opt_lp_x_vars["solution_value"] = opt_lp_x_vars["variable_object"].apply(lambda item: item.varValue)
        opt_lp_x_vars.drop(columns=["variable_object"], inplace=True)

        return opt_lp_f_vars, opt_lp_x_vars