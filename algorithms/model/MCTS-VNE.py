import numpy as np
from collections import namedtuple
from util import config
from data_loader import predata
from algorithms.model.base import BaseModel
from mcst_vne.mcts import MCTS
import networkx as nx


class MCTSModel(BaseModel):
    def  __init__(self,):
        """
        :param solution: 求解
        :param SN_Link: 底层链路/三元组(si,sj,bandwidth)
        :param SN_Node: 底层结点/cpu可利用资源
        :param VN_Link: 虚拟链路
        :param VN_Node: 虚拟节点
        :param VN_Life: 虚拟请求生存时间
        """
        super().__init__()

        self.mcts = MCTS(computation_budget=5, exploration_constant=.5)

    def create_graph(self,nodes, links):

        g = nx.Graph()
        g.add_weighted_edges_from(links)
        for i in range(len(nodes)):
            g.nodes[i]["weight"] = nodes[i]
        # print(g.edges.data('weight'))
        # print(g.nodes.data('weight'))
        return g

    def get_nodes_links(self,graph:nx.Graph):
        edges = graph.edges.data("weight")
        nodes = [node [1] for node in graph.nodes.data("weight")]
        return nodes,edges

    def get_solution(self,net, s_nodes, v_nodes, s_links, v_links, *args, **kwargs):


        sn = self.create_graph(s_nodes,s_links)
        vnr = self.create_graph(v_nodes,v_links)

        # 节点映射
        mapping_nodes = self.mcts.run(sn, vnr)
        print(mapping_nodes)
        mapping_links = []

        # 用于记录是否映射成功
        code = False
        # 用于记录描述信息
        info = ""

        InfoMapping = namedtuple("InfoMapping", "solution code info")

        if len(mapping_nodes):
            # 链路映射
            mapping_links = self.virtual_link_mapping(mapping_nodes, v_links, s_links)
            if len(mapping_links):
                code = True
                info = "映射成功"
            else:
                info = "链路映射失败"
        else:
            info = "节点映射成功"

        return {
            'net': None,
            'best_mapping_solution': {
            "mapping_links": mapping_links, "mapping_nodes": mapping_nodes,"code":code,"info":info
            }
        }

def run(current_iter_time,numbers):

    model = MCTSModel()
    data = predata.load_data(numbers)
    model.experience2(None,data,full_request=True,load_model_path='MCTS',current_iter_time=current_iter_time)

numbers = 1000
run(current_iter_time=numbers,numbers=numbers)