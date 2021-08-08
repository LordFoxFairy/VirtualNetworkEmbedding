import torch
from util import utils, config
from data_loader import predata
from algorithms.model.base import BaseModel
from algorithms.network.GRNetModel import GRNet
from algorithms.Optimizer.AdaxModel import AdaX,AdaXW
import adabound
from tqdm import tqdm
import torch.multiprocessing as mp


# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.enabled=False


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

    def get_solution(self, net, s_nodes, v_nodes, s_links, v_links,
                     original_s_nodes, original_s_links, batch_size=10,
                     dropout=.5, alpha=.2, iter_time=config.ITER_TIMES, b=-1, gamma=0.9, eps=.5,longterm_rc=0.1):
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

        device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
        if config.IS_GPU:
            net = net.to(device=device)

        best_mapping_solution = {
            'code': False,
            'mapping_nodes': [],
            'mapping_links': dict(),
            'link_consumption': config.INF
        }
        # b = -1
        lamda = 1e-2
        ceiterion = torch.optim.Adadelta(net.parameters(), lr=1e-4, eps=1e-2,weight_decay=1e-4)
        # ceiterion = torch.optim.Adam(net.parameters(), lr=1e-6, weight_decay=1e-6)
        # ceiterion = torch.optim.RMSprop(net.parameters(), lr=1e-4, alpha=0.99, eps=1e-08, weight_decay=1e-4, momentum=0., centered=False)
        #
        # ceiterion = AdaX(net.parameters(), lr=1e-8,weight_decay=1e-8)
        # ceiterion = torch.optim.SGD(net.parameters(), lr=1e-6, momentum=.812,weight_decay=1e-4)
        current_iter_max_time = 20
        flag = 0
        # total_loss = 0
        current_iter_time = 0
        for i in tqdm(range(iter_time)):

            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()

            s_node_indexes, s_inputs = utils.get_shuffled_indexes_and_inputs(input=s_input, batch_size=batch_size)
            if config.IS_GPU:
                s_node_indexes = s_node_indexes.cuda()
                s_inputs = s_inputs.cuda()
                v_input = v_input.cuda()
            node_mapping_solutions, shuffled_node_mapping_solutions, output_weights = net.get_node_mapping(
                s_node_indexes=s_node_indexes,
                s_inputs=s_inputs,
                v_input=v_input,
            )
            # 检测node mapping solutions是否符合，若符合则进行链路映射
            embedding_successes, link_mapping_solutions, link_consumptions, hops = utils.get_hops_and_link_consumptions(
                s_nodes=s_nodes,
                s_links=s_links,
                origin_v_nodes=v_nodes,
                origin_v_links=v_links,
                origin_node_mapping=node_mapping_solutions
            )
            flag += sum(embedding_successes)
            if flag:
                current_iter_max_time = current_iter_max_time - 1
            current_iter_time = current_iter_time+1
            if current_iter_max_time <= 0:
                break

            # link_consumptions = link_consumptions - link_consumptions.mean()
            if config.IS_GPU:
                link_consumptions = link_consumptions.to(device=device)

            if b == -1:
                b = link_consumptions.mean()

            # 记录下最优
            j = torch.argmin(link_consumptions)
            if link_consumptions[j] < best_mapping_solution['link_consumption']:
                best_mapping_solution['mapping_nodes'] = node_mapping_solutions[j]
                best_mapping_solution['mapping_links'] = link_mapping_solutions[j]
                best_mapping_solution['link_consumption'] = link_consumptions[j]
                best_mapping_solution['code'] = embedding_successes[j]

            # print(embedding_successes)

            link_consumptions = (1-longterm_rc + eps) * link_consumptions
            reward = ((b - link_consumptions)).squeeze()
            if config.IS_GPU:
                reward = reward.cuda()
            # 计算loss
            cross_entropy_loss = net.get_CrossEntropyLoss(output_weights, shuffled_node_mapping_solutions)
            loss = torch.dot(cross_entropy_loss,reward)

            regularization_loss = 0
            for param in net.parameters():
                regularization_loss += torch.sum(abs(param))

            loss = loss + lamda * regularization_loss

            # loss = torch.log(loss)

            if config.IS_FLOODING:
                b_loss = config.FLOODING_LOSS
                loss = (loss - b_loss).abs() + b_loss

            # total_loss += loss

            b = b * alpha + (1 - alpha) * link_consumptions.mean()

            ceiterion.zero_grad()
            loss.backward(retain_graph=True)
            ceiterion.step()

            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()

        # total_loss /= current_iter_time
        print("当前解决方法有：{}".format(flag))
        # print("当前损失loss为：{}".format(total_loss))

        return {
            'net': net,
            'best_mapping_solution': best_mapping_solution,
            'b': b
        }

def run(current_iter_time,numbers):
    model = NetModel()
    data = predata.load_data(numbers)
    net = GRNet(hidden_dim=128, batch_size=config.BATCH_SIZE, embedding_dim=128, dropout=.05)
    print(net)
    model.experience2(net, data, load_model_path="GR-VNE", full_request=config.FULL_REQUEST,current_iter_time=current_iter_time,numbers=numbers)

# from or_main.main import vne_main
# vne_main.main()
numbers = 1000
run(current_iter_time=numbers,numbers=numbers)
