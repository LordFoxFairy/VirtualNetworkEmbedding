from vine.utils.compare import get_run_result
import numpy as np
from vine.utils.graph_utils import load_data, create_network_graph


def graphViNE_run(physical_graph, requests, save=True, load=2000, max_time=500, verbose=True, n_clusters = 4,arrive_time=[]):
    physical_graph_vine = physical_graph.copy()
    request_graphs_graph_vine = [r.copy() for r in requests]
    results = get_run_result(
        physical_graph_vine, request_graphs_graph_vine,arrive_time, method="graphViNE", traffic_load=load, max_time=max_time, cost_revenue=True, utils=True, verbose=verbose,
        n_clusters=n_clusters)
    print(results)

def gateViNE_run(physical_graph, requests, save=True, load=2000, max_time=500, verbose=True, n_clusters = 4,arrive_time=[]):
    physical_graph_vine = physical_graph.copy()
    request_graphs_graph_vine = [r.copy() for r in requests]
    results = get_run_result(
        physical_graph_vine, request_graphs_graph_vine,arrive_time, method="gateViNE", traffic_load=load, max_time=max_time, cost_revenue=True, utils=True, verbose=verbose,
        n_clusters=n_clusters)
    print(results)

def compute():
    np.random.seed(64)  # to get a unique result every time
    # physical_graph = create_network_graph(nodes_num=100)
    # requests = [create_network_graph(np.random.randint(3, 11), min_feature_val=4, max_feature_val=10,
    #                                  min_link_val=4, max_link_val=10, connection_prob=0.7, life_time=(100, 900)) for i in range(12500)]
    #
    physical_graph,requests,arrive_time = load_data(numbers=1000)
    # print(physical_graph)

    load = 1000
    max_time = arrive_time[-1]+1

    select = 2

    if select == 1:
        print("graphViNE")
        graphViNE_run(physical_graph, requests, load=load,
                  max_time=max_time, verbose=False, n_clusters=4,arrive_time=arrive_time)
    else:
        print("gateViNE")
        gateViNE_run(physical_graph, requests, load=load,
                  max_time=max_time, verbose=False, n_clusters=4,arrive_time=arrive_time)

if __name__ == "__main__":
    compute()
