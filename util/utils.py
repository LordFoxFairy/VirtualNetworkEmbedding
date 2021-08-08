# -*- coding: utf-8 -*-
import os, torch
from itertools import islice

import numpy as np
from util import config
from layer.gcn import GCN
from layer.gat import GAT

os.chdir(os.getcwd())

import networkx as nx


"""
工具类
"""

####################################################################################

def get_graph(link, node):
    """
    根据链路以及节点构造网络图
    :param link:
    :param node:
    :return:
    """
    graph = np.diag(node)
    for n1, n2, bandwith in link:
        graph[int(n1), int(n2)] = graph[int(n2), int(n1)] = bandwith
    return graph

# 给定nodes和links，生成网络的输入数据input
def get_input(nodes, links):
    node_num = len(nodes)
    node_resource = torch.Tensor(nodes).view(size=(node_num,))
    node_neighbour_link_resource_sum = torch.zeros(size=(node_num,))
    node_neighbour_link_resource_min = torch.zeros(size=(node_num,))
    node_neighbour_link_resource_max = torch.ones(size=(node_num,)) * config.INF
    for link in links:
        u_node = link[0]
        v_node = link[1]
        bandwidth = link[2]
        node_neighbour_link_resource_sum[u_node] += bandwidth
        node_neighbour_link_resource_sum[v_node] += bandwidth
        node_neighbour_link_resource_min[u_node] = min(node_neighbour_link_resource_min[u_node], bandwidth)
        node_neighbour_link_resource_min[v_node] = min(node_neighbour_link_resource_min[v_node], bandwidth)
        node_neighbour_link_resource_max[u_node] = max(node_neighbour_link_resource_max[u_node], bandwidth)
        node_neighbour_link_resource_max[v_node] = max(node_neighbour_link_resource_max[v_node], bandwidth)

    graph = get_graph(links,nodes)
    graph = torch.FloatTensor(graph)
    adj = torch.where(graph>0,torch.ones_like(graph),graph)
    nfeat = adj.size(0)
    nout = config.NCLASS
    input = None

    graph_features = None

    if config.IS_GCN:
        x = get_features(graph)
        gcn = GCN(x.shape[1],nout)
        graph_features = gcn(x,adj)

    if config.IS_GAT:
        x = get_features(graph)
        gat = GAT(n_feat=x.shape[1],
                        n_hid=16,
                        n_class=3,
                        dropout=.0,
                        n_heads=8,
                        alpha=.1)
        graph_features = gat(x,adj)
        # graph_features = torch.abs(graph_features)
        # graph_features = normlization(graph_features)

        # graph_features = torch.stack(
        #     [
        #         standardization(node_resource),
        #         torch.squeeze(graph_features[:,0]),
        #         torch.squeeze(graph_features[:,1]),
        #         # torch.squeeze(graph_features[:,2]),
        #         node_resource,
        #     ],
        #     dim=1
        # )


    if nout == 1:
        input = torch.stack(
            [
                node_resource,
                node_neighbour_link_resource_sum,
                # node_neighbour_link_resource_min,
                # node_neighbour_link_resource_max,
                torch.squeeze(graph_features)
            ],
            dim=1
        )
    else:
        input = graph_features
    return input

# 给定一个网络输入数据input，输出多个乱序的inputs
def get_shuffled_indexes_and_inputs(input, batch_size=10):
    node_num = input.size()[0]
    node_indexes = []

    for i in range(batch_size):
        shuffled_index = torch.randperm(node_num)
        node_indexes.append(shuffled_index)

    node_indexes = torch.stack(node_indexes, dim=0).long()
    inputs = input[node_indexes]
    node_indexes = node_indexes.unsqueeze(dim=2)
    return node_indexes, inputs

####################################################################################

def get_cost_matrix(n,v_nodes,mapping_nodes,mapping_links):
    cost_node_matrix = np.zeros(n)
    cost_link_matrix = np.zeros((n,n))
    try:
        for i in range(len(v_nodes)):
            cost_node_matrix[mapping_nodes[i]] = v_nodes[i]

        for (s,v,bandwidth),path in mapping_links.items():
            for i in range(1,len(path)):
                v = int(path[i])
                u = int(path[i - 1])
                cost_link_matrix[v, u] = cost_link_matrix[u, v] = bandwidth
    except:
        cost_node_matrix = get_cost_node_matrix(n, mapping_nodes)
        cost_link_matrix = get_cost_link_matrix(n, mapping_links)
    return  {
        "cost_link_matrix": cost_link_matrix,
        "cost_node_matrix": cost_node_matrix
    }

def get_cost_link_matrix(n,links):
    cost_link_matrix = np.zeros((n,n))
    for link,bandwidth in links:
        for i in range(1,len(link)):
            v = int(link[i])
            u = int(link[i-1])
            cost_link_matrix[v,u] = cost_link_matrix[u,v] = bandwidth
    return cost_link_matrix


def get_cost_node_matrix(n,nodes):
    cost_node_matrix = np.zeros(n)
    for vnode,[snode,bandwidth] in nodes.items():
        cost_node_matrix[snode] = bandwidth
    return cost_node_matrix

####################################################################################

def link_embedding(s_links, slink_path, v_bandwidth):
    """
    一次链路嵌入
    :param s_links:
    :param slink_path:
    :param v_bandwidth:
    :return:
    """
    for i in range(1, len(slink_path)):
        u = slink_path[i - 1]
        v = slink_path[i]
        for j in range(len(s_links)):
            u2 = s_links[j][0]
            v2 = s_links[j][1]
            if (u == u2 and v == v2) or (u == v2 and v == u2):
                s_links[j][2] -= v_bandwidth

def link_release(s_links, slink_path, v_bandwidth):
    """
    一次链路释放
    :param s_links:
    :param slink_path:
    :param v_bandwidth:
    :return:
    """
    for i in range(1, len(slink_path)):
        u = slink_path[i - 1]
        v = slink_path[i]
        for j in range(len(s_links)):
            u2 = s_links[j][0]
            v2 = s_links[j][1]
            if (u == u2 and v == v2) or (u == v2 and v == u2):
                s_links[j][2] += v_bandwidth

# 更新物理网络资源
def update_network(s_nodes,s_links,snode_update_matrix,slink_update_matrix):
    '''
        s_nodes : 当前物理网络节点资源
        s_links : 当前物理网络链路资源
        snode_update_matrix: 物理节点资源更新矩阵，映射网络时是负的节点资源cost矩阵，释放网络时是正的节点资源cost矩阵
        slink_update_matrix: 物理链路资源更新矩阵，映射网络时是负的链路资源cost矩阵，释放网络时是正的链路资源cost矩阵
        return: s_nodes,更新后的s_nodes; s_links,更新后的s_links
    '''
    for i in range(len(s_nodes)):
        s_nodes[i] += snode_update_matrix[i]

    for i in range(len(s_links)):
        u = s_links[i][0]
        v = s_links[i][1]
        s_links[i][2] += slink_update_matrix[u][v]

    return s_nodes, s_links

####################################################################################

def get_revenue_cost_ratio(revenue, cost):
    try:
        revenue_cost_ratio = revenue / cost
    except ZeroDivisionError:
        revenue_cost_ratio = 0
    return revenue_cost_ratio

def get_total_resources(nodes,links):
    """
    计算总资源
    :param nodes:
    :param links:
    :return:
    """
    return get_total_link_resources(links) + get_total_node_resources(nodes)

def get_total_node_resources(nodes):
    """
    计算节点资源
    :param nodes:
    :return:
    """
    node_resources = 0
    for node in nodes:
        node_resources += node
    return node_resources

def get_total_link_resources(links):
    """
    计算链路资源
    :param links:
    :return:
    """
    link_resources = 0
    for link in links:
        link_resources += link[2]
    return link_resources

def get_node_utilization(current_sn_nodes, original_sn_nodes):
    current_node_resources = get_total_node_resources(current_sn_nodes)
    total_node_resources = get_total_node_resources(original_sn_nodes)
    used_node_resources = total_node_resources - current_node_resources

    node_utilization = 0
    try:
        node_utilization = used_node_resources / total_node_resources
    except ZeroDivisionError:
        print('除0错误')

    return node_utilization

def get_link_utilization(current_sn_links, original_sn_links):
    current_link_resources = get_total_link_resources(current_sn_links)
    total_link_resources = get_total_link_resources(original_sn_links)
    used_link_resources = total_link_resources - current_link_resources

    link_utilization = 0
    try:
        link_utilization = used_link_resources / total_link_resources
    except ZeroDivisionError:
        print('除0错误')

    return link_utilization

def get_total_utilization(current_sn_nodes,original_sn_nodes,current_sn_links,original_sn_links):
    """
    计算总资源占用率
    :param current_sn_nodes:
    :param original_sn_nodes:
    :param current_sn_links:
    :param original_sn_links:
    :return:
    """
    current_resources = get_total_resources(current_sn_nodes, current_sn_links)
    total_resources = get_total_resources(original_sn_nodes, original_sn_links)
    used_resources = total_resources - current_resources

    utilization = 0
    try:
        utilization = used_resources / total_resources
    except ZeroDivisionError:
        print('除0错误')

    return utilization

####################################################################################

def short_path_graph(nodes,links, s, e, v_bandwidth):
    """
    寻找物理节点si与sj的最短路径，其中约束条件为v_bandwidth
    :param graph:
    :param s:
    :param e:
    :param v_bandwidth:
    :return:
    """
    g = get_graph(node=nodes,link=links)
    G = nx.Graph(g)
    # all_paths = [p for p in nx.all_simple_paths(G,source=s,target=e)]
    # real_paths = []
    # minlen = np.Inf
    # for path in all_paths:
    #     flag = True
    #     for i in range(1,len(path)):
    #         if graph[path[i],path[i-1]] < v_bandwidth:
    #             flag = False
    #             break
    #     if flag:
    #         real_paths.append(path)
    #         if minlen > len(path):
    #             minlen = len(path)
    #
    # real_shortest_path = []
    # for path in real_paths:
    #     if minlen == len(path):
    #         real_shortest_path.append(path)
    #
    # solutions = len(real_shortest_path)
    # # print(real_shortest_path)
    # # print("一共有{}中条路径".format(solutions))
    #
    # return real_shortest_path

    return nx.shortest_path(G, s, e, weight=v_bandwidth)

def short_path_link(slinks, start, end, v_bandwidth):  # 最短路径
    # 节点数目
    node_num = np.max(np.array(slinks,dtype=int)[:,:-1])+1
    inf = np.inf
    node_map = np.zeros(shape=(node_num, node_num), dtype=float)
    for link in slinks:
        node_map[link[0]][link[1]] = node_map[link[1]][link[0]] = link[2]

    hop = [inf for i in range(node_num)]
    visited = [False for i in range(node_num)]
    pre = [-1 for i in range(node_num)]
    hop[start] = 0

    for i in range(node_num):
        u = -1
        for j in range(node_num):
            if not visited[j]:
                if u == -1 or hop[j] < hop[u]:
                    u = j
        visited[u] = True
        if u == end:
            break

        for j in range(node_num):
            if hop[j] > hop[u] + 1 and node_map[u][j] >= v_bandwidth:
                hop[j] = hop[u] + 1
                pre[j] = u

    path = []
    if pre[end] !=-1:
        v = end
        while v!=-1:
            path.append(v)
            v = pre[v]
        path.reverse()
    # print("当前节点{}与节点{}的最短路径为:{}".format(start,end,path))
    return path

def get_hops_and_link_consumptions(s_nodes, s_links, origin_v_nodes, origin_v_links, origin_node_mapping):
    '''
    :param s_nodes: 物理节点资源 list (s_node_num,)
    :param s_links: 物理链路资源 list (s_link_num,), struct s_link = (u, v, bandwidth)
    :param v_nodes: 虚拟节点资源 list (v_node_num,)
    :param v_links: 虚拟链路资源 list (v_link_num,), struct v_link = (u, v, bandwidth)
    :param node_mapping: 节点映射方案
    :return: embedding success 是否映射成功; link_mapping_solutions 链路映射方案, link_consumptions 链路映射消耗, hops 链路映射消耗跳数
    '''

    batch_size = origin_node_mapping.shape[0]
    v_node_num = origin_node_mapping.shape[1]
    hops = torch.zeros(size=(batch_size, 1))
    link_consumptions = torch.zeros(size=(batch_size, 1))
    flag = {}
    link_mapping_solutions = [dict() for i in range(batch_size)]
    embedding_success = [False for i in range(batch_size)]

    for i in range(batch_size):
        node_mapping_success = True
        link_mapping_success = True

        # 是否进行多对一
        if config.MANY_TO_ONE:
            v_nodes,v_links,node_mapping,flag = get_action(origin_v_nodes,origin_v_links,origin_node_mapping[i])
            if not flag:
                v_nodes,v_links,node_mapping = origin_v_nodes,origin_v_links,origin_node_mapping[i]
        else:
            v_nodes,v_links,node_mapping = origin_v_nodes,origin_v_links,origin_node_mapping[i]

        v_node_num = len(v_nodes)

        for j in range(v_node_num):
            if s_nodes[node_mapping[j]] < v_nodes[j]:
                node_mapping_success = False
                link_mapping_success = False
                # print('节点映射失败')
                break

        v_link_consumption_sum = 0
        for v_link in v_links:
            v_link_consumption_sum += v_link[2]

        if node_mapping_success:
            embedded_paths = []
            for v_link in v_links:
                v_from_node = v_link[0]
                v_to_node = v_link[1]
                v_bandwidth = v_link[2]

                s_from_node = node_mapping[v_from_node]
                s_to_node = node_mapping[v_to_node]

                s_path = short_path_link(s_links, s_from_node, s_to_node, v_bandwidth)

                if s_path == []:
                    link_mapping_success = False
                    # print('链路映射失败')
                    break
                else:
                    link_embedding(s_links, s_path, v_bandwidth)
                    embedded_paths.append([s_path, v_bandwidth])
                    hops[i][0] += len(s_path) - 1
                    link_consumptions[i][0] += (len(s_path)-1) * v_bandwidth
                    link_mapping_solutions[i].update({v_link: s_path})

            for path, v_bandwidth in embedded_paths:
                link_release(s_links, path, v_bandwidth)

            if not link_mapping_success:
                hops[i] = 7 * len(v_links)
                index = [i-1 for i in np.bincount(node_mapping) if i]
                ans = sum(index)
                if ans:
                    link_consumptions[i] = 100000
                else:
                    link_consumptions[i] = v_link_consumption_sum * 14
                link_mapping_solutions[i] = dict()
        else:
            hops[i] = 7 * len(v_links)
            index = [i - 1 for i in np.bincount(node_mapping) if i]
            ans = sum(index)
            if ans:
                link_consumptions[i] = 100000
            else:
                link_consumptions[i] = v_link_consumption_sum * 14
            link_mapping_solutions[i] = dict()

        if node_mapping_success and link_mapping_success:
            embedding_success[i] = True

    return embedding_success, link_mapping_solutions, link_consumptions, hops

####################################################################################

def get_nodes_links(graph):
    n = len(graph)
    nodes = []
    links = []
    for i in range(n):
        nodes.append(graph[i,i])
        for j in range(i,n):
            if graph[i,j] and i!=j:
                links.append((i,j,graph[i,j]))
    return nodes,links

def unique(data):
    new_data = []
    for d in data:
        if d not in new_data:
            new_data.append(d)
    return new_data

# import numba
#
# @numba.jit()
def get_action(nodes, links, action):
    new_action = {}
    idx_new_action = {}

    action_unique = {a: i for i, a in enumerate(unique(action))}
    idx_action_unique = {i: a for i, a in enumerate(unique(action))}

    action_ = [action_unique.get(a) for a in action]

    for i, a in enumerate(action_):
        if a in new_action.keys():
            value = new_action.get(a)
            value.append(i)
        else:
            new_action.update({a: [i]})
        idx_new_action.update({i: a})

    new_n = len(new_action)
    new_graph = np.zeros((new_n, new_n))
    n_len = int(np.ceil(np.sqrt(len(nodes))))

    new_action = list(new_action.items())

    graph = get_graph(link=links, node=nodes)

    flag = True

    if new_n < n_len:
        return None, None, None, False

    for i in range(new_n):
        newi = i
        new_graph[newi, newi] = sum([nodes[j] for j in new_action[i][1]])
        current_action = new_action[i][1]
        Q = [current_action[0]]
        for _ in range(1, len(current_action)):
            for i in range(1, len(current_action)):
                a = current_action[i]
                for b in Q:
                    if graph[a, b] != 0 and a != b:
                        Q.append(a)
                if len(Q) == len(current_action):
                    break
        if len(set(Q)) == len(current_action):
            continue
        elif new_n != 1:
            flag = False
            return None, None, None, flag

    Q = []
    for i in range(new_n):
        a_set = new_action[i]
        for link in links:
            u, v, bandwith = link
            a = idx_new_action.get(u)
            b = idx_new_action.get(v)
            if u in a_set[1] and v in a_set[1]:
                continue
            elif u in a_set[1] or v in a_set[1]:
                if link not in Q:
                    new_graph[a, b] = new_graph[b, a] = new_graph[b, a] + bandwith
                    Q.append((u, v, bandwith))

    # print(Q)
    new_action = [idx_action_unique.get(new_action[i][0]) for i in range(new_n)]
    nodes, links = get_nodes_links(new_graph)
    return nodes, links, new_action, flag

####################################################################################

def load_model(model, path):
    if os.path.exists(path):
        model = torch.load(path)
        print('Load model in {} successfully\n'.format(path))
    else:
        print('Cannot find {}'.format(path))
    return model

####################################################################################
# 将数据规范化
def normlization(x):
    return (x - x.min()) / (x.max() - x.min()+1e-6)

def standardization(x):
    mu = torch.mean(x,dim=0)
    sigma = torch.std(x,dim=0)
    return (x - mu) / (sigma+1e-6)
####################################################################################
def get_graph_features(graph):
    """
    获取图相关特征
    :param graph:
    :return:
    """
    features = dict()
    G = nx.Graph(graph)

    # 度中心性
    degree_centrality = list(nx.degree_centrality(G).values())
    # # 图的特征向量中心性
    # eigenvector_centrality = list(nx.eigenvector_centrality(G,max_iter=100).values())
    # # 图节点的 Katz 中心性
    # katz_centrality = list(nx.katz_centrality(G,max_iter=100).values())
    # 边的介数中心性
    closeness_centrality = list(nx.closeness_centrality(G).values())
    # 每个节点的子图中心性
    subgraph_centrality = list(nx.subgraph_centrality(G).values())
    # 节点pagerank
    pagerank = list(nx.pagerank(G,max_iter=100).values())
    # 节点聚类系数
    clustering = list(nx.clustering(G).values())

    nodes,links = get_nodes_links(graph)
    node_num = len(nodes)
    node_resource = torch.Tensor(nodes).view(size=(node_num,))
    node_neighbour_link_resource_sum = torch.zeros(size=(node_num,))
    node_neighbour_link_resource_min = torch.zeros(size=(node_num,))
    node_neighbour_link_resource_max = torch.ones(size=(node_num,)) * config.INF
    for link in links:
        u_node = link[0]
        v_node = link[1]
        bandwidth = link[2]
        node_neighbour_link_resource_sum[u_node] += bandwidth
        node_neighbour_link_resource_sum[v_node] += bandwidth
        node_neighbour_link_resource_min[u_node] = min(node_neighbour_link_resource_min[u_node], bandwidth)
        node_neighbour_link_resource_min[v_node] = min(node_neighbour_link_resource_min[v_node], bandwidth)
        node_neighbour_link_resource_max[u_node] = max(node_neighbour_link_resource_max[u_node], bandwidth)
        node_neighbour_link_resource_max[v_node] = max(node_neighbour_link_resource_max[v_node], bandwidth)

    graph = torch.FloatTensor(graph)
    adj = torch.where(graph > 0, torch.ones_like(graph), graph)
    degree = adj.sum(dim=1) # 节点度

    # features.update({"degree_centrality":degree_centrality})
    # features.update({"clustering":clustering})
    # features.update({"eigenvector_centrality":eigenvector_centrality})
    # features.update({"katz_centrality":katz_centrality})
    # features.update({"closeness_centrality":closeness_centrality})
    # features.update({"subgraph_centrality":subgraph_centrality})
    features.update({"pagerank":pagerank})
    features.update({"node_neighbour_link_resource_sum":list(standardization(node_neighbour_link_resource_sum).numpy())})
    # features.update({"node_neighbour_link_resource_max":list(standardization(node_neighbour_link_resource_max).numpy())})
    features.update({"node_resource":list(standardization(node_resource).numpy())})

    # features.update({"node_neighbour_link_resource_sum":list((node_neighbour_link_resource_sum).numpy())})
    # features.update({"node_neighbour_link_resource_max":list((node_neighbour_link_resource_max).numpy())})
    # features.update({"node_resource":list((node_resource).numpy())})

    # features.update({"node_neighbour_link_resource_min":list(standardization(node_neighbour_link_resource_min).numpy())})
    # features.update({"degree":list(degree.numpy())})
    return features


def get_features(graph):
    # torch.set_printoptions(precision=2,sci_mode=False)
    features_dict = get_graph_features(graph.numpy())
    features = torch.from_numpy(np.array(list(features_dict.values())).T).float()
    # features = standardization(features)
    features = features
    return features
####################################################################################

# slinks = [[136, 144, 160000.0], [136, 145, 160000.0], [137, 146, 160000.0], [137, 147, 160000.0], [138, 144, 160000.0], [138, 145, 160000.0], [139, 146, 160000.0], [139, 147, 160000.0], [140, 144, 160000.0], [140, 145, 160000.0], [141, 146, 160000.0], [141, 147, 160000.0], [142, 144, 160000.0], [142, 145, 160000.0], [143, 146, 160000.0], [143, 147, 160000.0], [128, 136, 40000.0], [128, 137, 40000.0], [129, 136, 40000.0], [129, 137, 40000.0], [130, 138, 40000.0], [130, 139, 40000.0], [131, 138, 40000.0], [131, 139, 40000.0], [132, 140, 40000.0], [132, 141, 40000.0], [133, 140, 40000.0], [133, 141, 40000.0], [134, 142, 40000.0], [134, 143, 40000.0], [135, 142, 40000.0], [135, 143, 40000.0], [0, 128, 10000.0], [1, 128, 10000.0], [2, 128, 10000.0], [3, 128, 10000.0], [4, 128, 10000.0], [5, 128, 10000.0], [6, 128, 10000.0], [7, 128, 10000.0], [8, 128, 10000.0], [9, 128, 10000.0], [10, 128, 10000.0], [11, 128, 10000.0], [12, 128, 10000.0], [13, 128, 10000.0], [14, 128, 10000.0], [15, 128, 10000.0], [16, 129, 10000.0], [17, 129, 10000.0], [18, 129, 10000.0], [19, 129, 10000.0], [20, 129, 10000.0], [21, 129, 10000.0], [22, 129, 10000.0], [23, 129, 10000.0], [24, 129, 10000.0], [25, 129, 10000.0], [26, 129, 10000.0], [27, 129, 10000.0], [28, 129, 10000.0], [29, 129, 10000.0], [30, 129, 10000.0], [31, 129, 10000.0], [32, 130, 10000.0], [33, 130, 10000.0], [34, 130, 10000.0], [35, 130, 10000.0], [36, 130, 10000.0], [37, 130, 10000.0], [38, 130, 10000.0], [39, 130, 10000.0], [40, 130, 10000.0], [41, 130, 10000.0], [42, 130, 10000.0], [43, 130, 10000.0], [44, 130, 10000.0], [45, 130, 10000.0], [46, 130, 10000.0], [47, 130, 10000.0], [48, 131, 10000.0], [49, 131, 10000.0], [50, 131, 10000.0], [51, 131, 10000.0], [52, 131, 10000.0], [53, 131, 10000.0], [54, 131, 10000.0], [55, 131, 10000.0], [56, 131, 10000.0], [57, 131, 10000.0], [58, 131, 10000.0], [59, 131, 10000.0], [60, 131, 10000.0], [61, 131, 10000.0], [62, 131, 10000.0], [63, 131, 10000.0], [64, 132, 10000.0], [65, 132, 10000.0], [66, 132, 10000.0], [67, 132, 10000.0], [68, 132, 10000.0], [69, 132, 10000.0], [70, 132, 10000.0], [71, 132, 10000.0], [72, 132, 10000.0], [73, 132, 10000.0], [74, 132, 10000.0], [75, 132, 10000.0], [76, 132, 10000.0], [77, 132, 10000.0], [78, 132, 10000.0], [79, 132, 10000.0], [80, 133, 10000.0], [81, 133, 10000.0], [82, 133, 10000.0], [83, 133, 10000.0], [84, 133, 10000.0], [85, 133, 10000.0], [86, 133, 10000.0], [87, 133, 10000.0], [88, 133, 10000.0], [89, 133, 10000.0], [90, 133, 10000.0], [91, 133, 10000.0], [92, 133, 10000.0], [93, 133, 10000.0], [94, 133, 10000.0], [95, 133, 10000.0], [96, 134, 10000.0], [97, 134, 10000.0], [98, 134, 10000.0], [99, 134, 10000.0], [100, 134, 10000.0], [101, 134, 10000.0], [102, 134, 10000.0], [103, 134, 10000.0], [104, 134, 10000.0], [105, 134, 10000.0], [106, 134, 10000.0], [107, 134, 10000.0], [108, 134, 10000.0], [109, 134, 10000.0], [110, 134, 10000.0], [111, 134, 10000.0], [112, 135, 10000.0], [113, 135, 10000.0], [114, 135, 10000.0], [115, 135, 10000.0], [116, 135, 10000.0], [117, 135, 10000.0], [118, 135, 10000.0], [119, 135, 10000.0], [120, 135, 10000.0], [121, 135, 10000.0], [122, 135, 10000.0], [123, 135, 10000.0], [124, 135, 10000.0], [125, 135, 10000.0], [126, 135, 10000.0], [127, 135, 10000.0]]
#
# s,e,v_bandwidth = 83,59,1500.0
# print(short_path_link(slinks,s,e,v_bandwidth))
#
# # 测试数据
# # 测试数据
# graph = np.array([
#     [1,2,1,0.5],
#     [2,1,0,0.5],
#     [1,0,1,1],
#     [0.5,0.5,1,1]
# ])
# nodes,links = get_nodes_links(graph)
# print(short_path_graph(nodes,links,s=0,e=2,v_bandwidth=.5)[0])
