import torch

from util import utils, config
from data_loader import predata
from algorithms.model.base import BaseModel
from algorithms.network.PNNetModel import LNet

class NetModel(BaseModel):
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

    def get_solution(self,net, s_nodes, v_nodes, s_links, v_links,
                     original_s_nodes,original_s_links
                     ,batch_size=10,dropout=.5,alpha=.5,iter_time=config.iter_time,b=-1,longterm_rc=.5):
        """
        :param s_nodes:
        :param v_nodes:
        :param s_links:
        :param v_links:
        :param args:
        :param kwargs:
        :return:
        """

        s_input = utils.get_input(nodes=s_nodes, links=s_links)
        v_input = utils.get_input(nodes=v_nodes, links=v_links)

        best_mapping_solution = {
            'code': False,
            'mapping_nodes': [],
            'mapping_links': dict(),
            'link_consumption': config.INF
        }
        baseline = -1

        device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
        if config.IS_GPU:
            net.to(device=device)

        ceiterion = torch.optim.SGD(net.parameters(), lr=1e-5,momentum=.9)
        iter_time = 2
        for i in range(iter_time):
            s_node_indexes, s_inputs = utils.get_shuffled_indexes_and_inputs(input=s_input, batch_size=batch_size)
            if config.IS_GPU:
                s_node_indexes = s_node_indexes.cuda()
                s_inputs = s_inputs.cuda()
                v_input = v_input.cuda()
            node_mapping_solutions, shuffled_node_mapping_solutions,output_weights = net.get_node_mapping(
                s_node_indexes=s_node_indexes,
                s_inputs=s_inputs,
                v_input=v_input
            )

            # 检测node mapping solutions是否符合，若符合则进行链路映射
            embedding_successes, link_mapping_solutions, link_consumptions, hops = utils.get_hops_and_link_consumptions(
                s_nodes=s_nodes,
                s_links=s_links,
                origin_v_nodes=v_nodes,
                origin_v_links=v_links,
                origin_node_mapping=node_mapping_solutions
            )

            # 记录下最优
            j = torch.argmin(link_consumptions)
            if link_consumptions[j] < best_mapping_solution['link_consumption']:
                best_mapping_solution['mapping_nodes'] = node_mapping_solutions[j]
                best_mapping_solution['mapping_links'] = link_mapping_solutions[j]
                best_mapping_solution['link_consumption'] = link_consumptions[j]
                best_mapping_solution['code'] = embedding_successes[j]

            if baseline == -1:
                baseline = link_consumptions.mean()

            # 计算loss
            adv = (baseline - link_consumptions).squeeze()
            if config.IS_GPU:
                adv = adv.cuda()
            cross_entropy_loss = net.get_CrossEntropyLoss(output_weights, shuffled_node_mapping_solutions)
            loss = torch.dot(cross_entropy_loss, adv)

            # Adam优化参数
            net.zero_grad()
            loss.backward(retain_graph=True)
            ceiterion.step()

            baseline = baseline * alpha + (1 - baseline) * link_consumptions.mean()

        return {
            'net': net,
            'best_mapping_solution': best_mapping_solution
        }

def run(current_iter_time,numbers):

    model = NetModel()
    data = predata.load_data(numbers)
    # 网络模型
    net = LNet(nhidden=128, batch_size=10, nembedding=128, dropout=.5)
    print(net)
    model.experience2(net,data,load_model_path="PN-VNE",full_request = config.full_request,current_iter_time=current_iter_time,numbers=numbers)

numbers = 1000
run(current_iter_time=numbers,numbers=numbers)