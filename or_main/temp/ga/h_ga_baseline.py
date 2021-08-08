import copy

from algorithms.a_baseline import BaselineVNEAgent
from common import utils, config
from common.config import ALGORITHMS
from temp.ga.ga_utils import GAOperator, GAEarlyStopping


class GABaselineVNEAgent(BaselineVNEAgent):
    def __init__(self, logger):
        super(GABaselineVNEAgent, self).__init__(logger)
        self.type = ALGORITHMS.GENETIC_ALGORITHM

    def find_substrate_nodes(self, copied_substrate, vnr):
        subset_S_per_v_node = {}
        embedding_s_nodes = {}
        already_embedding_s_nodes = []

        for v_node_id, v_node_data in vnr.net.nodes(data=True):
            v_cpu_demand = v_node_data['CPU']
            v_node_location = v_node_data['LOCATION']

            # Find the subset S of substrate nodes that satisfy restrictions and
            # available CPU capacity (larger than that specified by the request.)
            subset_S_per_v_node[v_node_id] = utils.find_subset_S_for_virtual_node(
                copied_substrate, v_cpu_demand, v_node_location, already_embedding_s_nodes
            )

            selected_s_node_id = max(
                subset_S_per_v_node[v_node_id],
                key=lambda s_node_id: copied_substrate.net.nodes[s_node_id]['CPU'] - v_cpu_demand,
                default=None
            )

            if selected_s_node_id is None:
                self.num_node_embedding_fails += 1
                msg = "VNR {0} REJECTED ({1}): 'no suitable SUBSTRATE NODE for nodal constraints: {2}' {3}".format(
                    vnr.id, self.num_node_embedding_fails, v_cpu_demand, vnr
                )
                self.logger.info("{0} {1}".format(utils.step_prefix(self.time_step), msg))
                return None

            assert selected_s_node_id != -1
            embedding_s_nodes[v_node_id] = (selected_s_node_id, v_cpu_demand)

            if not config.ALLOW_EMBEDDING_TO_SAME_SUBSTRATE_NODE:
                already_embedding_s_nodes.append(selected_s_node_id)

            assert copied_substrate.net.nodes[selected_s_node_id]['CPU'] >= v_cpu_demand
            copied_substrate.net.nodes[selected_s_node_id]['CPU'] -= v_cpu_demand

        return embedding_s_nodes

    def find_substrate_path(self, copied_substrate, vnr, embedding_s_nodes):
        is_ok, results = utils.find_all_s_paths_2(copied_substrate, embedding_s_nodes, vnr)

        if is_ok:
            all_s_paths = results
        else:
            (v_link, v_bandwidth_demand) = results
            self.num_link_embedding_fails += 1

            if v_bandwidth_demand:
                msg = "VNR {0} REJECTED ({1}): 'no suitable LINK for bandwidth demand: {2} {3}".format(
                    vnr.id, self.num_link_embedding_fails, v_bandwidth_demand, vnr
                )
            else:
                msg = "VNR {0} REJECTED ({1}): 'not found for any substrate path for v_link: {2} {3}".format(
                    vnr.id, self.num_link_embedding_fails, v_link, vnr
                )

            self.logger.info("{0} {1}".format(utils.step_prefix(self.time_step), msg))
            return None

        # GENETIC ALGORITHM START: mapping the virtual nodes and substrate_net nodes
        print("[[VNR {0}] GA Started for {1} Virtual Links]".format(vnr.id, len(vnr.net.edges(data=True))))

        # LINK EMBEDDING VIA GENETIC ALGORITHM
        original_copied_substrate = copy.deepcopy(copied_substrate)

        early_stopping = GAEarlyStopping(
            patience=config.STOP_PATIENCE_COUNT, verbose=True, delta=0.0001, copied_substrate=copied_substrate
        )
        ga_operator = GAOperator(vnr, all_s_paths, copied_substrate, config.POPULATION_SIZE)
        ga_operator.initialize()
        generation_idx = 0
        while True:
            generation_idx += 1
            ga_operator.selection()
            ga_operator.crossover()
            ga_operator.mutation()
            ga_operator.sort_population_and_set_elite()

            solved, best_elite = early_stopping.evaluate(
                elite=ga_operator.elite, evaluation_value=ga_operator.elite.fitness
            )

            if solved:
                break

        assert original_copied_substrate == copied_substrate

        # s_path is selected from elite chromosome
        # early_stopping.print_best_elite()

        for v_link, (s_links_in_path, v_bandwidth_demand) in best_elite.embedding_s_paths.items():
            for s_link in s_links_in_path:
                if copied_substrate.net.edges[s_link]['bandwidth'] < v_bandwidth_demand:
                    self.num_link_embedding_fails += 1

                    msg = "VNR {0} REJECTED ({1}): 'After GA (fitness: {2:.6f}) " \
                          "-> no suitable LINK for bandwidth demand: {3} {4}".format(
                        vnr.id, self.num_link_embedding_fails, best_elite.fitness, v_bandwidth_demand, vnr
                    )

                    self.logger.info("{0} {1}".format(utils.step_prefix(self.time_step), msg))
                    return None

                copied_substrate.net.edges[s_link]['bandwidth'] -= v_bandwidth_demand

        return best_elite.embedding_s_paths
