import numpy as np
from collections import namedtuple
from util import config
from data_loader import predata
from algorithms.model.base import BaseModel


class VNRRWModel(BaseModel):
    def __init__(self,):
        """
        :param solution: 求解
        :param SN_Link: 底层链路/三元组(si,sj,bandwidth)
        :param SN_Node: 底层结点/cpu可利用资源
        :param VN_Link: 虚拟链路
        :param VN_Node: 虚拟节点
        :param VN_Life: 虚拟请求生存时间
        """
        super().__init__()

    def node_importance(self, graph):
        """
        进行求解图节点的重要性
        :param graph: 一个图
        :return:
        """
        eta = 1e-1
        graph = np.array(graph)
        n = len(graph)
        cpu = np.array([graph[i, i] for i in range(n)])  # ni的cpu可利用资源
        bw = [sum([graph[i, j] for j in range(n) if i != j]) for i in range(n)]  # 节点的邻接链路带宽
        H = np.array([cpu[i] * (bw[i] + eta) for i in range(n)])  # eta是避免数值为0
        NR = np.array([H[i] / (H.sum()) for i in range(n)])

        P1 = np.array([[H[j] / H.sum() for j in range(n)] for i in range(n)])
        P2 = np.zeros((n, n))

        p1 = np.diag(np.full(n, fill_value=0.15))
        p2 = np.diag(np.full(n, fill_value=0.85))

        for i in range(n):
            Hnbr = 0
            for j in range(n):
                if graph[i, j]:
                    Hnbr += H[j]
            for j in range(n):
                if graph[i, j]:
                    if Hnbr != 0:
                        P2[i, j] = H[j] / Hnbr
                    else:
                        P2[i, j] = 1
        T = P1.dot(p1) + P2.dot(p2)
        eta = 1e-1  # 阈值
        delta = np.abs(np.max(NR)) + 1
        while delta > eta:
            oldNR = NR
            newNR = T.dot(NR)
            delta = np.linalg.norm(newNR - oldNR)
            # print("*"*100)
            # print(delta)
            NR = newNR

        # 根据重要度进行排序，从大到小进行排序
        index = np.argsort(NR)[::-1]
        return index

    def get_solution(self,net, s_nodes, v_nodes, s_links, v_links, *args, **kwargs):

        # 节点映射
        mapping_nodes = self.virtual_node_mapping(v_nodes, v_links, s_links, s_nodes)
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

        # info_mapping = InfoMapping({"mapping_links": mapping_links, "mapping_nodes": mapping_nodes}, code, info)

        return {
            'net': None,
            'best_mapping_solution': {
            "mapping_links": mapping_links, "mapping_nodes": mapping_nodes,"code":code,"info":info
            }
        }

def run(current_iter_time,numbers):

    model = VNRRWModel()
    data = predata.load_data(numbers)
    # print(data["VN_Arrive_Time"])
    model.experience2(None,data,full_request=True,load_model_path='NodeRank',current_iter_time=current_iter_time)

numbers = 1000
run(current_iter_time=numbers,numbers=numbers)
# from data_loader import predata
# predata.plot_data.plot_result_data("NodeRank",0)