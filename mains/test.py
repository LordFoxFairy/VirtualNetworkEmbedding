# -*- coding: utf-8 -*-

"""
测试文件
"""

from data_loader import predata
from util import config


# 测试函数-用于测试函数功能
def test_func(id):
    # SN_path = predata.get_SN_Path(config.SnFile)
    # print(SN_path)

    # VN_path = predata.get_VN_Path(config.VnFile)
    # print(VN_path)

    # SN_Link = predata.get_SN_Link(config.SnFile,predata.get_SN_Node(config.SnFile))
    # print(SN_Link)

    # VN_Link = predata.get_VN_Link(config.SnFile,predata.get_VN_Node(config.VnFile))
    # print(VN_Link)

    (solution, SN_Link, SN_Node, VN_Link, VN_Node, VN_Life) = predata.read_SN_VN(config.SnFile, config.VnFile)

    # print("solution：{}, SN_Link：{}, SN_Node：{}, VN_Link：{}, VN_Node：{}, VN_Life：{}".format(
    #     solution[0], SN_Link[0], SN_Node[0], VN_Link[0], VN_Node[0], VN_Life[0][0]
    # ))

    return VN_Node[id], VN_Link[id]
    # return SN_Node, SN_Link


import numpy as np


def get_nodes_links(graph):
    n = len(graph)
    nodes = []
    links = []
    for i in range(n):
        nodes.append(graph[i, i])
        for j in range(i, n):
            if graph[i, j] and i != j:
                links.append([i, j, graph[i, j]])
    return nodes, links


from util import utils


def unique(data):
    new_data = []
    for d in data:
        if d not in new_data:
            new_data.append(d)
    return new_data


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

    graph = utils.get_graph(link=links, node=nodes)

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


sn = np.array([
    [10, 4, 5, 0, 0],
    [4, 5, 2, 0, 3],
    [5, 2, 3, 6, 1],
    [0, 0, 6, 4, 3],
    [0, 3, 1, 3, 6],
])

vn = np.array([
    [1, 2, 3, 1],
    [2, 2, 4, 0],
    [3, 4, 3, 1],
    [1, 0, 1, 4],
])

actions = np.array([
    [18, 19, 118, 43, 43, 90, 89],
    [0, 0, 0, 0, 0, 0, 0]
])


def get_mapping_links(mapping_links, actions):
    pass


# nodes, links = test_func(4)
# print(nodes, links)
# print(get_action(nodes, links, actions[1]))

# def func(a):
#     a = a + a
#
# def func2(a):
#     a += a
#
# a = [1]
# print(a)
# func(a)
# print(a)
# func2(a)
# print(a)

import torch

# encoder = torch.rand(10,1,256)
# rnn = torch.nn.LSTM(input_size=128*2, hidden_size=128, bidirectional=True, batch_first=True,
#                                  bias=False)
# decoder,(hn,cn) = rnn(encoder)
# print(decoder.shape)

# ffn_weight = torch.rand(10,148,128)
# h0 = torch.rand(10,148,2)
# hn = torch.einsum('ijl,ijk->kil', ffn_weight, h0)
# print(hn.shape)
# device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
# print(device)

from network import gat
from util import utils
import networkx

nodes, links = test_func(4)
graph = utils.get_graph(node=nodes,link=links)
graph = torch.FloatTensor(graph)

# adj = torch.where(graph>0,torch.ones_like(graph),graph)
#
# # 模型和优化器
# # model = gat.GraphAttentionLayer(
# #     in_features=graph.shape[0],
# #     out_features=3,
# #     dropout=.5,
# #     alpha=.3, concat=True)
#
# model = gat.GAT(n_feat=graph.shape[0],
#           n_hid=8,
#           n_class=3,
#           dropout=.1,
#           n_heads=3,
#           alpha=.05)

# print(model(graph,adj))

def get_features(graph):

    torch.set_printoptions(precision=2,sci_mode=False)

    features_dict = utils.get_graph_features(graph.numpy())
    features = torch.from_numpy(np.array(list(features_dict.values())).T).float()
    features = utils.standardization(features)
    return features

x = get_features(graph)
model = gat.GAT(n_feat=x.shape[1],
          n_hid=6,
          n_class=5,
          dropout=.0,
          n_heads=8,
          alpha=.1)

adj = torch.where(graph>0,torch.ones_like(graph),graph)
y = model(x,adj)
print(y)




















