import datetime
import itertools
import networkx as nx
from itertools import islice
import os, sys
import matplotlib.pyplot as plt
from matplotlib import MatplotlibDeprecationWarning
import warnings
import glob
import pandas as pd
from slack import WebClient
from slack.errors import SlackApiError

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)
plt.figure(figsize=(20, 10))

from or_main.common import config

client = WebClient(token=config.SLACK_API_TOKEN)


def get_revenue_VNR(vnr):
    revenue_cpu = sum((v_cpu_demand['CPU'] for _, v_cpu_demand in vnr.net.nodes(data=True)))

    revenue_bandwidth = sum((v_bandwidth_demand['bandwidth'] for _, _, v_bandwidth_demand in vnr.net.edges(data=True)))

    # revenue_cpu = 0.0
    # for _, v_cpu_demand in vnr.net.nodes(data=True):
    #     revenue_cpu += v_cpu_demand['CPU']
    #
    # revenue_bandwidth = 0.0
    # for _, _, v_bandwidth_demand in vnr.net.edges(data=True):
    #     revenue_bandwidth += v_bandwidth_demand['bandwidth']

    revenue = revenue_cpu + config.ALPHA * revenue_bandwidth

    return revenue


def get_cost_VNR(vnr, embedding_s_paths):
    cost_cpu = sum((v_cpu_demand['CPU'] for _, v_cpu_demand in vnr.net.nodes(data=True)))

    cost_embedded_s_path = sum(
        (len(s_links_in_path) * v_bandwidth_demand
         for _, (s_links_in_path, v_bandwidth_demand) in embedding_s_paths.items())
    )

    # cost_cpu = 0.0
    # for _, v_cpu_demand in vnr.net.nodes(data=True):
    #     cost_cpu += v_cpu_demand['CPU']

    # cost_embedded_s_path = 0.0
    # for v_link_id, (s_links_in_path, v_bandwidth_demand) in embedding_s_paths.items():
    #     cost_embedded_s_path += len(s_links_in_path) * v_bandwidth_demand

    cost = cost_cpu + config.ALPHA * cost_embedded_s_path

    return cost


def get_distance_factor_VNR(embedding_s_paths, copied_substrate):
    return 0.0


def get_attraction_strength_VNR(embedding_s_paths, copied_substrate):
    return 0.0


def get_total_hop_count_VNR(embedding_s_paths):
    total_hop_count = 0
    for _, (s_links_in_path, _) in embedding_s_paths.items():
        total_hop_count += len(s_links_in_path)
    return total_hop_count


def k_shortest_paths(G, source, target, k, weight=None):
    return list(
        islice(nx.shortest_simple_paths(G, source, target, weight=weight), k)
    )


def step_prefix(time_step):
    return "[STEP: {0:5d}]".format(time_step)


def agent_step_prefix(agent_label, time_step):
    return "[STEP: {0:5d}/{1:>7s}]".format(time_step, agent_label)


def run_agent_step_prefix(run, agent_label, time_step):
    return "[STEP: {0:5d}/{1:>7s}/R{2}]".format(time_step, agent_label, run)


def send_file_to_slack(filepath):
    try:
        response = client.files_upload(
            channels='#intelligent_network',
            file=filepath
        )
        assert response["file"]  # the uploaded file
    except SlackApiError as e:
        # You will get a SlackApiError if "ok" is False
        assert e.response["ok"] is False
        assert e.response["error"]  # str like 'invalid_auth', 'channel_not_found'
        print(f"Got an error: {e.response['error']}")


def peek_from_iterable(iterable):
    try:
        first = next(iterable)
    except StopIteration:
        return True, None
    return False, itertools.chain([first], iterable)


def get_sorted_v_links_with_edge_ranking(vnr):
    sorted_v_links_with_edge_ranking = []

    for src_v_node, dst_v_node, edge_data in vnr.net.edges(data=True):
        sorted_v_links_with_edge_ranking.append((src_v_node, dst_v_node, edge_data))

    sorted_v_links_with_edge_ranking.sort(
        key=lambda v_links_element: v_links_element[2]['bandwidth'], reverse=True
    )

    return sorted_v_links_with_edge_ranking


def get_sorted_v_nodes_with_node_ranking(vnr, type_of_node_ranking=config.TYPE_OF_VIRTUAL_NODE_RANKING.TYPE_1, beta=None):
    sorted_v_nodes_with_node_ranking = []

    # calculate the vnr node ranking
    for v_node_id, v_node_data in vnr.net.nodes(data=True):
        if type_of_node_ranking == config.TYPE_OF_VIRTUAL_NODE_RANKING.TYPE_1:
            vnr_node_ranking = calculate_node_ranking_1(
                vnr.net.nodes[v_node_id]['CPU'],
                vnr.net[v_node_id],
                beta
            )
        elif type_of_node_ranking == config.TYPE_OF_VIRTUAL_NODE_RANKING.TYPE_2:
            vnr_node_ranking = calculate_node_ranking_2(
                vnr.net.nodes[v_node_id]['CPU'],
                vnr.net[v_node_id]
            )
        else:
            raise ValueError()
        sorted_v_nodes_with_node_ranking.append((v_node_id, v_node_data, vnr_node_ranking))

    # sorting the vnr nodes with node's ranking
    sorted_v_nodes_with_node_ranking.sort(
        key=lambda v_nodes_element: v_nodes_element[2], reverse=True
    )

    return sorted_v_nodes_with_node_ranking


# PAPER: Topology-aware - 2013
def calculate_node_ranking_1(node_cpu_capacity, adjacent_links, beta):
    total_node_bandwidth = sum((adjacent_links[link_id]['bandwidth'] for link_id in adjacent_links))

    # total_node_bandwidth = 0.0
    # for link_id in adjacent_links:
    #     total_node_bandwidth += adjacent_links[link_id]['bandwidth']

    return beta * node_cpu_capacity + (1.0 - beta) * len(adjacent_links) * total_node_bandwidth


# PAPER: Rethinking (Baseline) - 2008
def calculate_node_ranking_2(node_cpu_capacity, adjacent_links):
    total_node_bandwidth = sum((adjacent_links[link_id]['bandwidth'] for link_id in adjacent_links))

    return node_cpu_capacity * total_node_bandwidth


def find_subset_S_for_virtual_node(copied_substrate, v_cpu_demand, v_node_location, already_embedding_s_nodes):
    '''
    find the subset S of the substrate nodes that satisfy restrictions and available CPU capacity
    :param substrate: substrate network
    :param v_cpu_demand: cpu demand of the given virtual node
    :return:
    '''

    if config.LOCATION_CONSTRAINT:
        subset_S = (
            s_node_id for s_node_id, s_node_data in copied_substrate.net.nodes(data=True)
            if s_node_data['CPU'] >= v_cpu_demand and
               s_node_id not in already_embedding_s_nodes and
               s_node_data['LOCATION'] == v_node_location and
               s_node_id < config.SUBSTRATE_NODES
        )
    else:
        subset_S = (
            s_node_id for s_node_id, s_node_data in copied_substrate.net.nodes(data=True)
            if s_node_data['CPU'] >= v_cpu_demand and
               s_node_id not in already_embedding_s_nodes and
               s_node_id < config.SUBSTRATE_NODES
        )

    return subset_S


num_calls = 0
def make_all_paths(sorted_v_links_with_node_ranking, idx, all_s_paths, copied_substrate, embedding_s_nodes):
    global num_calls
    num_calls += 1

    is_last = (idx == len(sorted_v_links_with_node_ranking) - 1)

    print(idx, len(sorted_v_links_with_node_ranking) - 1, is_last, num_calls, "---------")

    src_v_node = sorted_v_links_with_node_ranking[idx][0]
    dst_v_node = sorted_v_links_with_node_ranking[idx][1]
    v_link = (src_v_node, dst_v_node)
    src_s_node = embedding_s_nodes[src_v_node][0]
    dst_s_node = embedding_s_nodes[dst_v_node][0]
    v_bandwidth_demand = sorted_v_links_with_node_ranking[idx][2]['bandwidth']

    all_s_paths[v_link] = {}

    if src_s_node == dst_s_node:
        all_s_paths[v_link][0] = ([], v_bandwidth_demand)
    else:
        subnet = nx.subgraph_view(
            copied_substrate.net,
            filter_edge=lambda node_1_id, node_2_id: \
                True if copied_substrate.net.edges[(node_1_id, node_2_id)]['bandwidth'] >= v_bandwidth_demand else False
        )

        if len(subnet.edges) == 0 or not nx.has_path(subnet, source=src_s_node, target=dst_s_node):
            all_s_paths[v_link] = None

        paths = nx.all_simple_paths(
            subnet, source=src_s_node, target=dst_s_node, cutoff=config.MAX_EMBEDDING_PATH_LENGTH
        )

        is_empty, paths = peek_from_iterable(paths)
        if is_empty:
            all_s_paths[v_link] = None
        else:
            s_path_idx = 0
            for s_path in paths:
                print("[{0}]".format(s_path_idx), end=" ")

                s_links_in_path = []

                for node_idx in range(len(s_path) - 1):
                    s_link = (s_path[node_idx], s_path[node_idx + 1])
                    s_links_in_path.append(s_link)
                    copied_substrate.net.edges[s_link]['bandwidth'] -= v_bandwidth_demand

                all_s_paths[v_link][s_path_idx] = (s_links_in_path, v_bandwidth_demand)
                s_path_idx += 1

                if not is_last:
                    make_all_paths(
                        sorted_v_links_with_node_ranking=sorted_v_links_with_node_ranking,
                        idx=idx + 1,
                        all_s_paths=all_s_paths,
                        copied_substrate=copied_substrate,
                        embedding_s_nodes=embedding_s_nodes
                    )

                for node_idx in range(len(s_path) - 1):
                    s_link = (s_path[node_idx], s_path[node_idx + 1])
                    copied_substrate.net.edges[s_link]['bandwidth'] += v_bandwidth_demand

        print()


def find_all_s_paths_1(copied_substrate, embedding_s_nodes, vnr):
    sorted_v_links_with_node_ranking = get_sorted_v_links_with_edge_ranking(vnr=vnr)

    all_s_paths = {}

    print(sorted_v_links_with_node_ranking)

    make_all_paths(
        sorted_v_links_with_node_ranking=sorted_v_links_with_node_ranking,
        idx=0,
        all_s_paths=all_s_paths,
        copied_substrate=copied_substrate,
        embedding_s_nodes=embedding_s_nodes
    )


def find_all_s_paths_2(copied_substrate, embedding_s_nodes, vnr):
    all_s_paths = {}

    # 각 v_link 당 가능한 모든 s_path (set of s_link) 구성하여 all_s_paths에 저장
    for src_v_node, dst_v_node, edge_data in vnr.net.edges(data=True):
        v_link = (src_v_node, dst_v_node)
        src_s_node = embedding_s_nodes[src_v_node][0]
        dst_s_node = embedding_s_nodes[dst_v_node][0]
        v_bandwidth_demand = edge_data['bandwidth']

        if src_s_node == dst_s_node:
            all_s_paths[v_link][0] = ([], v_bandwidth_demand)
        else:
            subnet = nx.subgraph_view(
                copied_substrate.net,
                filter_edge=lambda node_1_id, node_2_id: \
                    True if copied_substrate.net.edges[(node_1_id, node_2_id)]['bandwidth'] >= v_bandwidth_demand else False
            )

            if len(subnet.edges) == 0 or not nx.has_path(subnet, source=src_s_node, target=dst_s_node):
                return False, (v_link, v_bandwidth_demand)

            all_paths = nx.all_simple_paths(
                subnet, source=src_s_node, target=dst_s_node, cutoff=config.MAX_EMBEDDING_PATH_LENGTH
            )

            all_s_paths[v_link] = {}
            s_path_idx = 0
            for path in all_paths:
                s_links_in_path = []
                for node_idx in range(len(path) - 1):
                    s_links_in_path.append((path[node_idx], path[node_idx + 1]))

                all_s_paths[v_link][s_path_idx] = (s_links_in_path, v_bandwidth_demand)
                s_path_idx += 1

            if s_path_idx == 0:
                return False, (v_link, None)

    return True, all_s_paths


def print_env_and_agent_info(env, agent, logger):
    msg = "[ENVIRONMENT]\n"
    msg += "TOTAL NUMBER of SUBSTRATE nodes: {0}\n".format(len(env.SUBSTRATE.net.nodes()))
    msg += "TOTAL NUMBER of SUBSTRATE edges: {0}\n".format(len(env.SUBSTRATE.net.edges()))
    msg += "DIAMETER of SUBSTRATE: {0}\n".format(nx.diameter(env.SUBSTRATE.net))
    msg += "TOTAL NUMBER of VNRs: {0}\n".format(len(env.VNRs_INFO))

    if logger:
        logger.info(msg)
    print(msg)

    msg = "[AGENT]\n"
    msg += "AGENT: {0}".format(agent.type.value)

    if logger:
        logger.info(msg)
    print(msg)

    print()

    response = input("Are you OK for All environment and agent information ? [y/n]: ")
    if not (response == "Y" or response == "y"):
        sys.exit(-1)


def draw_performance(
        agents, agent_labels, run, time_step,
        performance_revenue, performance_acceptance_ratio,
        performance_rc_ratio, performance_link_fail_ratio,
        send_image_to_slack=False
):
    files = glob.glob(os.path.join(config.graph_save_path, "*"))
    for f in files:
        os.remove(f)

    plt.style.use('seaborn-dark-palette')

    x_range = range(config.FIGURE_START_TIME_STEP, time_step + 1, config.TIME_WINDOW_SIZE)

    plt.subplot(411)

    for agent_id in range(len(agents)):
        plt.plot(
            x_range,
            performance_revenue[agent_id, config.FIGURE_START_TIME_STEP: time_step + 1: config.TIME_WINDOW_SIZE],
            label=agent_labels[agent_id]
        )

    plt.ylabel("Revenue")
    plt.xlabel("Time unit")
    plt.title("Revenue")
    plt.legend(loc="best", fancybox=True, framealpha=0.3, fontsize=12)
    plt.grid(True)

    plt.subplot(412)
    for agent_id in range(len(agents)):
        plt.plot(
            x_range,
            performance_acceptance_ratio[agent_id, config.FIGURE_START_TIME_STEP: time_step + 1: config.TIME_WINDOW_SIZE],
            label=agent_labels[agent_id]
        )

    plt.ylabel("Acceptance Ratio")
    plt.xlabel("Time unit")
    plt.title("Acceptance Ratio")
    plt.legend(loc="best", fancybox=True, framealpha=0.3, fontsize=12)
    plt.grid(True)

    plt.subplot(413)
    for agent_id in range(len(agents)):
        plt.plot(
            x_range,
            performance_rc_ratio[agent_id, config.FIGURE_START_TIME_STEP: time_step + 1: config.TIME_WINDOW_SIZE],
            label=agent_labels[agent_id]
        )

    plt.ylabel("R/C Ratio")
    plt.xlabel("Time unit")
    plt.title("R/C Ratio")
    plt.legend(loc="best", fancybox=True, framealpha=0.3, fontsize=12)
    plt.grid(True)

    plt.subplot(414)
    for agent_id in range(len(agents)):
        plt.plot(
            x_range,
            performance_link_fail_ratio[agent_id, config.FIGURE_START_TIME_STEP: time_step + 1: config.TIME_WINDOW_SIZE],
            label=agent_labels[agent_id]
        )

    plt.ylabel("Link Fails Ratio")
    plt.xlabel("Time unit")
    plt.title("Link Embedding Fails / Total Fails Ratio")
    plt.legend(loc="best", fancybox=True, framealpha=0.3, fontsize=12)
    plt.grid(True)

    plt.tight_layout()

    plt.subplots_adjust(top=0.9)

    plt.suptitle('EXECUTING RUNS: {0}/{1} FROM HOST: {2}'.format(
        run + 1, config.NUM_RUNS, config.HOST
    ))

    now = datetime.datetime.now()

    new_file_path = os.path.join(config.graph_save_path, "results_{0}.png".format(now.strftime("%Y_%m_%d_%H_%M")))
    plt.savefig(new_file_path)

    new_csv_file_path_revenue = os.path.join(config.csv_save_path, "performance_revenue_results_{0}.csv".format(run))
    new_csv_file_path_acceptance_ratio = os.path.join(config.csv_save_path, "performance_acceptance_ratio_results_{0}.csv".format(run))
    new_csv_file_path_rc_ratio = os.path.join(config.csv_save_path, "performance_rc_ratio_results_{0}.csv".format(run))
    new_csv_file_path_link_fail_ratio = os.path.join(config.csv_save_path, "performance_link_ratio_results_{0}.csv".format(run))

    if send_image_to_slack:
        send_file_to_slack(new_file_path)
        print("SEND IMAGE FILE {0} TO SLACK !!!".format(new_file_path))

        df_revenue = pd.DataFrame(performance_revenue)
        df_revenue.to_csv(new_csv_file_path_revenue, header=None, index=None)
        df_acceptance_ratio = pd.DataFrame(performance_acceptance_ratio)
        df_acceptance_ratio.to_csv(new_csv_file_path_acceptance_ratio, header=None, index=None)
        df_rc_ratio = pd.DataFrame(performance_rc_ratio)
        df_rc_ratio.to_csv(new_csv_file_path_rc_ratio, header=None, index=None)
        df_link_fail_ratio = pd.DataFrame(performance_link_fail_ratio)
        df_link_fail_ratio.to_csv(new_csv_file_path_link_fail_ratio, header=None, index=None)

    # if config.HOST.startswith("COLAB"):
    #     plt.show()
    #
    # plt.clf()

    plt.show()
