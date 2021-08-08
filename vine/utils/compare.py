from vine.graphViNE.model import cluster_using_argva as graphViNE_cluster_using_argva
from vine.gateViNE.gate import cluster_using_argva as gateViNE_cluster_using_argva
from vine.utils.graph_utils import from_networkx
import torch
from sklearn.cluster import KMeans, SpectralClustering, MiniBatchKMeans
from vine.utils.vine_best_fit import vine_embed, free_embedded_request
from vine.utils.compare_utils import compute_cost, compute_utils, compute_revenue
import networkx as nx
import numpy as np


def get_total_node_resources(nodes):
    node_resources = 0
    for node in nodes:
        node_resources += node[1]
    return node_resources


def get_total_link_resources(links):
    link_resources = 0
    for link in links:
        link_resources += link[2]
    return link_resources


def get_total_resources(g: nx.Graph, flag=False):
    s_nodes = g.nodes.data("CPU")
    s_links = g.edges.data("Bandwidth")
    if flag:
        print("释放后的物理网络node资源为:{}，link资源为:{}".format(get_total_node_resources(s_nodes),
                                                     get_total_link_resources(s_links)))
    else:
        print("释放前的物理网络node资源为:{}，link资源为:{}".format(get_total_node_resources(s_nodes),
                                                     get_total_link_resources(s_links)))


def get_run_result(physical, request_graphs, arrive_time, method="gateViNE", max_time=3000,
                   traffic_load=150, avg_life_time=500, verbose=True, cost_revenue=False, utils=False, n_clusters=4):
    r"""
    traffic load is in erlang
    """
    blockeds = 0
    num = 1
    probs = []
    embeddeds = []
    load_acc = 0
    revenues = []
    costs = []
    request_index = 0
    link_utils = []
    cpu_utils = []
    gpu_utils = []
    memory_utils = []
    pred = []
    model = None
    current_time = arrive_time[request_index]
    train_iter_time = 5
    for t in range(1, max_time):  # Loop over all times
        # compute number of request recieves on this time slot
        load_acc += (traffic_load / avg_life_time)
        req_n_in_this_time = int(load_acc)
        load_acc -= req_n_in_this_time
        if current_time == t:
            get_total_resources(physical)
            print("当前第{}时刻，第{}号虚拟网络请求进行嵌入".format(t, request_index))
            ##################################### GraphVine Method ################################################
            if method == 'graphViNE':
                if request_index % train_iter_time == 0:  # retrain or not
                    # Change data to torch_geometric data
                    print("当前第{}次训练".format(request_index // train_iter_time))
                    data = from_networkx(physical, normalize=True)
                    data.edge_attr = data.edge_attr / data.edge_attr.sum()
                    device = torch.device(
                        'cuda' if torch.cuda.is_available() else 'cpu')
                    data = data.to(device)
                    # train model
                    model = graphViNE_cluster_using_argva(data, verbose=False,
                                                          max_epoch=100) if model is None else graphViNE_cluster_using_argva(
                        data, verbose=False, max_epoch=5, pre_trained_model=model)
                    with torch.no_grad():
                        z = model.encode(
                            data.x, data.edge_index, data.edge_attr)
                    # print(z.cpu().data)
                    pred = KMeans(n_clusters=n_clusters).fit_predict(z.cpu().data)
                request_embedded, physical, request_graphs[request_index] = vine_embed(
                    physical, pred, request_graphs[request_index], verbose=False, N_CLUSTERS=n_clusters)

            if method == 'gateViNE':
                if request_index % train_iter_time == 0:  # retrain or not
                    # Change data to torch_geometric data
                    print("当前第{}次训练".format(request_index // train_iter_time))
                    data = from_networkx(physical, normalize=True)
                    data.edge_attr = data.edge_attr / data.edge_attr.sum()
                    device = torch.device(
                        'cuda' if torch.cuda.is_available() else 'cpu')
                    data = data.to(device)
                    # train model
                    model = gateViNE_cluster_using_argva(data, verbose=False,
                                                         max_epoch=100) if model is None else gateViNE_cluster_using_argva(
                        data, verbose=False, max_epoch=5, pre_trained_model=model)
                    with torch.no_grad():
                        z = model.encode(
                            data.x, data.edge_index, data.edge_attr)
                    pred = MiniBatchKMeans(n_clusters=n_clusters).fit_predict(z.cpu().data)
                request_embedded, physical, request_graphs[request_index] = vine_embed(
                    physical, pred, request_graphs[request_index], verbose=False, N_CLUSTERS=n_clusters)

            num += 1
            if not request_embedded:
                blockeds += 1
            if request_embedded:
                embeddeds.append(
                    [t, request_graphs[request_index], request_index])
                if cost_revenue:
                    revenues.append(compute_revenue(
                        request_graphs[request_index]))
                    costs.append(compute_cost(request_graphs[request_index]))
                if utils:
                    if 'GPU' in physical.nodes[0]:
                        cpu_util, link_util, gpu_util, mem_util = compute_utils(
                            physical)
                        gpu_utils.append(gpu_util)
                        memory_utils.append(mem_util)
                    else:
                        cpu_util, link_util = compute_utils(physical)
                    cpu_utils.append(cpu_util)
                    link_utils.append(link_util)

            if verbose:
                if request_embedded:
                    print(
                        f'\033[92m request {request_index} embedded successfully')
                else:
                    print(
                        f'\033[93m request {request_index} could not embedded')
            request_index += 1
            print("prob:{}".format(1 - blockeds / num))
        if request_index < len(arrive_time):
            current_time = arrive_time[request_index]



        probs.append(blockeds / num)

        new_embeddeds = []
        for e in embeddeds:
            embedded_time = e[0]
            request = e[1]
            if t - embedded_time == request.graph['LifeTime']:
                free_embedded_request(physical, request)
                if verbose:
                    print(f'\033[94m free embedded graph {e[2]} successfully')
            else:
                new_embeddeds.append(e)
        embeddeds = new_embeddeds

        if t == arrive_time[request_index - 1]:
            get_total_resources(physical, flag=True)

        if request_index == len(arrive_time):
            break
    return_values = [probs, blockeds, num]

    # if method == "gateViNE":
    #     torch.save(model, 'gateViNE')

    if cost_revenue:
        return_values.append(costs)
        return_values.append(revenues)
    if utils:
        return_values.append(cpu_utils)
        return_values.append(link_utils)
        if 'GPU' in physical.nodes[0]:
            return_values.append(gpu_utils)
            return_values.append(memory_utils)
    return return_values
