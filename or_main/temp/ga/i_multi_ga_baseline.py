import copy
import time

from algorithms.a_baseline import BaselineVNEAgent
from common import utils, config
from common.config import ALGORITHMS
from temp.ga.ga_utils import GAEarlyStopping, MultiGAOperator
from common.utils import peek_from_iterable


class MultiGAVNEAgent(BaselineVNEAgent):
    def __init__(self, logger):
        super(MultiGAVNEAgent, self).__init__(logger)
        self.type = ALGORITHMS.MULTI_GENETIC_ALGORITHM

    def embedding(self, VNRs_COLLECTED, COPIED_SUBSTRATE, action):
        sorted_vnrs = sorted(
            VNRs_COLLECTED.values(), key=lambda vnr: vnr.revenue, reverse=True
        )

        for vnr in sorted_vnrs:
            original_copied_substrate = copy.deepcopy(COPIED_SUBSTRATE)

            s_nodes_combinations = self.find_substrate_nodes_combinations(vnr, COPIED_SUBSTRATE)

            assert original_copied_substrate == COPIED_SUBSTRATE

            if s_nodes_combinations is None:
                action.vnrs_postponement[vnr.id] = vnr
                continue

            early_stopping = GAEarlyStopping(
                patience=config.STOP_PATIENCE_COUNT, verbose=True, delta=0.0001, copied_substrate=COPIED_SUBSTRATE
            )

            multi_ga_operator = MultiGAOperator(vnr, s_nodes_combinations, COPIED_SUBSTRATE)

            ### INITIALIZE ###
            is_ok = multi_ga_operator.initialize()

            if not is_ok:
                self.num_link_embedding_fails += 1

                msg = "VNR {0} REJECTED ({1}): 'no suitable LINK for bandwidth demand: {2}}".format(
                    vnr.id, self.num_link_embedding_fails, vnr
                )

                self.logger.info("{0} {1}".format(utils.step_prefix(self.time_step), msg))
                action.vnrs_postponement[vnr.id] = vnr
                continue

            multi_ga_operator.process()
            expected_generation = 1
            while True:
                if multi_ga_operator.is_all_one_generation_finished(expected_generation=expected_generation):
                    multi_ga_operator.evaluate_all_results_from_workers()

                    solved, _ = early_stopping.evaluate(
                        elite=multi_ga_operator.elite, evaluation_value=multi_ga_operator.elite.fitness
                    )

                    if solved:
                        break

                    multi_ga_operator.go_next_generation()
                    expected_generation += 1

                time.sleep(0.1)

            assert original_copied_substrate == COPIED_SUBSTRATE

    def find_substrate_nodes_combinations(self, vnr, COPIED_SUBSTRATE):
        sorted_v_nodes_with_node_ranking = utils.get_sorted_v_nodes_with_node_ranking(
            vnr=vnr, type_of_node_ranking=config.TYPE_OF_VIRTUAL_NODE_RANKING.TYPE_2
        )

        #print(sorted_v_nodes_with_node_ranking, "!!!!!")

        all_combinations = []

        self.make_top_n_combinations(
            sorted_v_nodes_with_node_ranking=sorted_v_nodes_with_node_ranking,
            idx=0,
            combination=[],
            all_combinations=all_combinations,
            copied_substrate=COPIED_SUBSTRATE,
            already_embedding_s_nodes=[]
        )

        print("TOTAL {0} combinations".format(len(all_combinations)))
        # for idx, combination in enumerate(all_combinations):
        #     print(idx, combination)

        s_nodes_combinations = []
        for combination_idx, combination in enumerate(all_combinations):
            if len(combination) != len(sorted_v_nodes_with_node_ranking):
                self.num_node_embedding_fails += 1
                msg = "VNR {0} REJECTED ({1}): 'no suitable SUBSTRATE NODE for nodal constraints' {2}".format(
                    vnr.id, self.num_node_embedding_fails, vnr
                )
                self.logger.info("{0} {1}".format(utils.step_prefix(self.time_step), msg))
                return None

            #print(vnr.id, combination_idx, combination)

            embedding_s_nodes = {}
            for idx, selected_s_node_id in enumerate(combination):
                v_node_id = sorted_v_nodes_with_node_ranking[idx][0]
                v_cpu_demand = sorted_v_nodes_with_node_ranking[idx][1]['CPU']
                embedding_s_nodes[v_node_id] = (selected_s_node_id, v_cpu_demand)

            s_nodes_combinations.append(embedding_s_nodes)

        return s_nodes_combinations

    def make_top_n_combinations(
            self, sorted_v_nodes_with_node_ranking, idx, combination, all_combinations, copied_substrate,
            already_embedding_s_nodes
    ):
        is_last = (idx == len(sorted_v_nodes_with_node_ranking) - 1)

        v_cpu_demand = sorted_v_nodes_with_node_ranking[idx][1]['CPU']
        v_node_location = sorted_v_nodes_with_node_ranking[idx][1]['LOCATION']

        subset_S = utils.find_subset_S_for_virtual_node(
            copied_substrate, v_cpu_demand, v_node_location, already_embedding_s_nodes
        )

        is_empty, subset_S = peek_from_iterable(subset_S)
        if is_empty:
            return

        selected_subset_S = sorted(
            subset_S,
            key=lambda s_node_id: utils.calculate_node_ranking_2(
                copied_substrate.net.nodes[s_node_id]['CPU'],
                copied_substrate.net[s_node_id],
            ),
            reverse=True
        )[:config.MAX_NUM_CANDIDATE_S_NODES_PER_V_NODE]

        #print(idx, len(selected_subset_S), selected_subset_S, "###############")

        for s_node_id in selected_subset_S:
            new_combination = combination + [s_node_id]

            assert copied_substrate.net.nodes[s_node_id]['CPU'] >= v_cpu_demand
            copied_substrate.net.nodes[s_node_id]['CPU'] -= v_cpu_demand
            if not config.ALLOW_EMBEDDING_TO_SAME_SUBSTRATE_NODE:
                already_embedding_s_nodes.append(s_node_id)

            if is_last:
                all_combinations.append(new_combination)
            else:
                self.make_top_n_combinations(
                    sorted_v_nodes_with_node_ranking=sorted_v_nodes_with_node_ranking,
                    idx=idx + 1,
                    combination=new_combination,
                    all_combinations=all_combinations,
                    copied_substrate=copied_substrate,
                    already_embedding_s_nodes=already_embedding_s_nodes
                )

            copied_substrate.net.nodes[s_node_id]['CPU'] += v_cpu_demand
            if not config.ALLOW_EMBEDDING_TO_SAME_SUBSTRATE_NODE:
                already_embedding_s_nodes.remove(s_node_id)