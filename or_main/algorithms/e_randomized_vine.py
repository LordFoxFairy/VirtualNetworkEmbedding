from or_main.algorithms.d_deterministic_vine import DeterministicVNEAgent
from or_main.common import utils
import numpy as np
from scipy.special import softmax

import warnings

warnings.filterwarnings(action='ignore')


class RandomizedVNEAgent(DeterministicVNEAgent):
    def __init__(
            self, logger, time_window_size, agent_type, type_of_virtual_node_ranking,
            allow_embedding_to_same_substrate_node, max_embedding_path_length, substrate_nodes
    ):
        super(RandomizedVNEAgent, self).__init__(
            logger, time_window_size, agent_type, type_of_virtual_node_ranking,
            allow_embedding_to_same_substrate_node, max_embedding_path_length, substrate_nodes
        )

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

            selected_s_node_p_value = []
            candidate_s_node_id = []
            selected_s_node_id = max(
                subset_S_per_v_node[v_node_id],
                key=lambda s_node_id:
                    sum(opt_lp_f_vars[(opt_lp_f_vars['u'] == s_node_id) &
                                      (opt_lp_f_vars['v'] == v_node_id + self.substrate_nodes)]['solution_value'].values +
                        opt_lp_f_vars[(opt_lp_f_vars['u'] == v_node_id + self.substrate_nodes) &
                                      (opt_lp_f_vars['v'] == s_node_id)]['solution_value'].values
                        ) *
                    opt_lp_x_vars[(opt_lp_x_vars['u'] == s_node_id) &
                                  (opt_lp_x_vars['v'] == v_node_id + self.substrate_nodes)]['solution_value'].values,
                default=None
            )

            # for calculating p_value
            # selected_s_node_p_value = []
            # candidate_s_node_id = []
            # for s_node_id in subset_S_per_v_node[v_node_id]:
            #     candidate_s_node_id.append(s_node_id)
            #     selected_s_node_p_value.append(
            #         sum(opt_lp_f_vars[
            #                 (opt_lp_f_vars['u'] == s_node_id) &
            #                 (opt_lp_f_vars['v'] == v_node_id + self.substrate_nodes)]['solution_value'].values +
            #             opt_lp_f_vars[
            #                 (opt_lp_f_vars['u'] == v_node_id + self.substrate_nodes) &
            #                 (opt_lp_f_vars['v'] == s_node_id)]['solution_value'].values))

            # Calculate the probability
            # scipy softmax 추가하여 이용하기
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.softmax.html
            total_p_value = sum(selected_s_node_p_value)
            if total_p_value == 0:
                self.num_node_embedding_fails += 1
                msg = "VNR {0} REJECTED ({1}): 'no suitable SUBSTRATE NODE for nodal constraints: {2}' {3}".format(
                    vnr.id, self.num_node_embedding_fails, v_cpu_demand, vnr
                )
                self.logger.info("{0} {1}".format(utils.step_prefix(self.time_step), msg))
                self.revoke_from_augmented_substrate(copied_substrate, vnr, self.substrate_nodes)
                return None
            else:
                probability = softmax(selected_s_node_p_value)
                selected_s_node_id = np.random.choice(candidate_s_node_id, p=probability)

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
