import copy, sys, time,os
from queue import Queue
import numpy as np
import torch
from util import utils, config
import pandas as pd

class BaseModel(object):
    def __init__(self, solution, SN_Link, SN_Node, VN_Link, VN_Node, VN_Life):
        """
        :param solution: 求解
        :param SN_Link: 底层链路/三元组(si,sj,bandwidth)
        :param SN_Node: 底层结点/cpu可利用资源
        :param VN_Link: 虚拟链路
        :param VN_Node: 虚拟节点
        :param VN_Life: 虚拟请求生存时间
        """
        self.solution = solution
        self.SN_Link = SN_Link
        self.SN_Node = SN_Node
        self.VN_Link = VN_Link
        self.VN_Node = VN_Node
        self.VN_Life = VN_Life

    def node_importance(self, graph):
        """
        进行求解图节点的重要性
        :param graph: 一个图
        :return:
        """
        pass

    def get_graph(self, link, node):
        """
        根据链路以及节点构造网络图
        :param link:
        :param node:
        :return:
        """
        return torch.FloatTensor(utils.get_graph(link,node))

    def virtual_node_mapping(self, vn_nodes, vn_links, sn_links, sn_nodes):
        """
        虚拟节点映射过程
        :param vn_nodes:
        :param vn_links:
        :param sn_links:
        :param sn_nodes:
        :return:
        """

        # 通过链路以及节点构造网络图
        vgraph = self.get_graph(vn_links, vn_nodes)
        sgraph = self.get_graph(sn_links, sn_nodes)

        # 节点重要性
        vindex = self.node_importance(vgraph)  # 虚拟节点重要性
        sindex = self.node_importance(sgraph)  # 物理节点重要性

        mapping_nodes = {}

        # 标记当前物理节点是否被选取
        labels = np.ones(len(sindex))

        # 进行节点映射
        for i in range(len(vindex)):
            for j in range(len(sindex)):
                if sn_nodes[sindex[j]] >= vn_nodes[vindex[i]] and labels[j]:
                    # {虚拟节点序号：[物理节点序号，带宽]}
                    mapping_nodes.update({vindex[i]: [sindex[j], vn_nodes[vindex[i]]]})
                    labels[j] = 0
                    break
                if len(mapping_nodes) == len(vindex):
                    break

        # 判断当前节点映射是否成功
        if len(mapping_nodes) != len(vindex):
            mapping_nodes = {}
        return mapping_nodes

    def virtual_link_mapping(self, mapping_nodes, vn_links, sn_links):
        """
        虚拟链路映射过程
        :param mapping_node:
        :param vn_nodes:
        :param vn_links:
        :param sn_links:
        :return:
        """
        # 用于存储映射的链路路径
        mapping_links = []

        for link in vn_links:
            vi, vj, v_bandwidth = link
            si, sj = mapping_nodes.get(vi)[0], mapping_nodes.get(vj)[0]
            # 寻找物理节点si与sj的最短路径，其中约束条件为v_bandwidth
            path = utils.short_path_link(sn_links, si, sj, v_bandwidth)
            if len(path) == 0:
                break
            # [最短路径,约束带宽条件]
            mapping_links.append([path, v_bandwidth])

        # 判断当前链路映射是否成功
        if len(mapping_links) != len(vn_links):
            mapping_links = []
        return mapping_links

    def update_network(self, s_links, s_nodes, cost_link_matrix, cost_node_matrix):
        return utils.update_network(
            s_nodes=s_nodes,
            s_links=s_links,
            snode_update_matrix=cost_node_matrix,
            slink_update_matrix=cost_link_matrix
        )

    def link_embedding(self, s_links, s_paths):
        """
        链路嵌入
        :param s_links:
        :param s_paths:
        :return:
        """
        for path, bandwidth in s_paths:
            utils.link_embedding(s_links, path, bandwidth)

    def node_embedding(self, s_nodes, m_nodes):
        """
        节点嵌入
        :param s_nodes:
        :param m_nodes:
        :return:
        """
        for key,(index,bandwidth) in m_nodes.items():
            s_nodes[index] -= bandwidth


    def link_release(self, s_links, s_paths):
        """
        链路资源释放
        :param s_links:
        :param s_paths:
        :return:
        """
        for path, bandwidth in s_paths:
            utils.link_release(s_links, path, bandwidth)

    def node_release(self, s_nodes, m_nodes):
        """
        节点资源释放
        :param s_nodes:
        :param m_nodes:
        :return:
        """
        for key,(index,bandwidth) in m_nodes.items():
            s_nodes[index] += bandwidth


    def get_solution(self):
        pass

    # 程序入口
    def experience(self,net, data, times=config.TIMES, step=config.STEP,new_try_numbers = config.NEW_TRY_NUMBERS,
                   max_try_numbers=config.MAX_TRY_NUMBERS, first_request_num=100, full_request=True, dropout=.2,
                   load_model_path='LVne'):

        # 数据
        SN_Link = data.get("SN_Link")
        SN_Node = data.get("SN_Node")
        VN_Node = data.get("VN_Node")
        VN_Link = data.get("VN_Link")
        VN_Life = data.get("VN_Life")
        # solution = data.get("solution")

        n = len(SN_Link)

        if load_model_path != '':
            path = copy.deepcopy(load_model_path)
            try:
                path = os.path.join(os.getcwd(),path)
            except:
                pass
            net = utils.load_model(net, path)
        try:
            from network import LNetModel
            from network import GRNetModel
            try:
                net.apply(LNetModel.weights_init)
            except:
                net.apply(GRNetModel.weights_init)
        except:
            pass

        valid_queue = Queue()  # 即将填充的虚拟网络
        failed_queue = Queue()  # 填充失败的虚拟网络

        original_s_links = copy.deepcopy(SN_Link)  # 保存当前链路以及节点信息
        original_s_nodes = copy.deepcopy(SN_Node)
        s_nodes = copy.deepcopy(original_s_nodes)
        s_links = copy.deepcopy(original_s_links)

        node_utilization = []  # 结点利用率
        link_utilization = []  # 链路，同上
        accept_ratio = []  # 链路，同上
        rc = []  # 收益成本比
        period_cost_matrix = dict()  # 各个时期，资源消耗情况，为了便于释放资源

        success_num = 0
        finish_time = 0
        total_num = 0

        longterm_revenue = 0
        longterm_cost = 0
        longterm_rc = .5

        b = -1
        flag_released = False

        all_time = []

        # 设置测试数量
        max_request_num = len(VN_Link) if full_request else first_request_num

        # 将虚拟网路请求放入队列，进行等待部署
        for i in range(max_request_num):
            valid_queue.put(VN_Life[0][i])
            # id = VN_Life[0][i][0]

        # 时间段
        for t in range(0, times, step):

            # 队列禁止，每个时刻尝试20次
            # # 取出失败队列的虚拟网络，重新进行嵌入
            # while not failed_queue.empty():
            #     valid_queue.put(failed_queue.get())


            # 打印每个时刻剩余的虚拟网络请求数量以及当前剩余的物理网络资源
            if not valid_queue.empty():
                print("-" * 60)
                print("这是在{}时刻下的虚拟网络映射请求，当前还剩下{}个虚拟网络请求等待映射".format(t, valid_queue.qsize()))
                print("释放前的物理网络node资源为:{}，link资源为:{}".format(utils.get_total_node_resources(s_nodes),
                                                             utils.get_total_link_resources(s_links)))

            # 释放资源
            if t in period_cost_matrix.keys():
                if not valid_queue.empty():
                    print('--当前时刻要释放{}个虚拟网络--'.format(period_cost_matrix[t]["released_numbers"]))

                s_nodes, s_links = self.update_network(
                    s_nodes=s_nodes,
                    s_links=s_links,
                    cost_node_matrix=period_cost_matrix[t]["cost_node_matrix"],
                    cost_link_matrix=period_cost_matrix[t]["cost_link_matrix"],
                )
                period_cost_matrix[t]["released_numbers"] -= 1
                flag_released = True

                # # 当前的资源状态
                # original_s_links = copy.deepcopy(s_links)
                # original_s_nodes = copy.deepcopy(s_nodes)

            if flag_released:
                print("释放后的物理网络node资源为:{}，link资源为:{}".format(utils.get_total_node_resources(s_nodes),
                                                             utils.get_total_link_resources(s_links)))
                flag_released = False

            # 如果队列为空，就表示当前无虚拟网络请求
            if valid_queue.empty():
                continue

            # 设置尝试请求次数
            if t == 0:
                try_numbers = min(valid_queue.qsize(), max_try_numbers)
            else:
                try_numbers = new_try_numbers

            # count = 0
            # count_success = 0

            # 每个时刻尝试映射valid_queue中的try_numbers个虚拟网络请求
            for i in range(try_numbers):
                # 用于记录尝试次数
                sys.stdout.write('[{}/{}]'.format(i + 1, try_numbers))
                sys.stdout.flush()

                start_time = time.time()

                # 如果没有虚拟网络发起请求，就退出
                if valid_queue.empty():
                    break
                vn = valid_queue.get()  # [id,生存周期,start,end]
                id = vn[0]
                life_time = vn[1] // config.time_unit
                v_links = VN_Link[id]
                v_nodes = VN_Node[id]

                print('[当前{}时刻,已映射{}个虚拟网络，当前尝试映射的第{}号虚拟网络，还剩下{}个虚拟网络等待嵌入]'.format(t, success_num, id,valid_queue.qsize()))
                # print('v_node_num = ', len(v_nodes), 'v_link_num = ', len(v_links), 'v_life_time = ',life_time)

                # 得到映射结果
                result = self.get_solution(
                    net=net,
                    s_nodes=s_nodes,
                    v_nodes=v_nodes,
                    s_links=s_links,
                    v_links=v_links,
                    original_s_nodes=original_s_nodes,
                    original_s_links=original_s_links,
                    b=b,
                    longterm_rc=longterm_rc
                )

                b = result.get("b")
                best_mapping_solution = result.get("best_mapping_solution")
                embedding_status = best_mapping_solution.get("code")
                net = result.get("net")

                total_num = total_num + 1
                print(best_mapping_solution)

                # embedding_status为false时表示映射失败
                ## 将嵌入失败的虚拟节点放入失败队列中，进行等待
                if not embedding_status:
                    failed_queue.put(vn)
                    print("映射失败")
                else:
                    print("当前第{}号虚拟网络嵌入成功".format(id))

                    # count_success = count_success + 1

                    # print(result.solution)

                    # 记录嵌入成功的个数
                    success_num += 1

                    ## 更新网络
                    current_solution = best_mapping_solution
                    mapping_nodes = current_solution.get("mapping_nodes")
                    mapping_links = current_solution.get("mapping_links")
                    cost_matrix = utils.get_cost_matrix(n, v_nodes, mapping_nodes, mapping_links)

                    ## 性能
                    added_revenue = utils.get_total_resources(nodes=v_nodes, links=v_links)
                    added_node_cost = cost_matrix['cost_node_matrix'].sum()
                    added_link_cost = cost_matrix['cost_link_matrix'].sum() / 2.0
                    added_total_cost = added_node_cost + added_link_cost
                    longterm_revenue += added_revenue
                    longterm_cost += added_total_cost

                    s_nodes, s_links = self.update_network(
                        s_nodes=s_nodes,
                        s_links=s_links,
                        cost_link_matrix=-1 * cost_matrix["cost_link_matrix"],
                        cost_node_matrix=-1 * cost_matrix["cost_node_matrix"],
                    )

                    ## 记录当前消耗的资源，在k时刻后，进行释放
                    if (t + life_time) not in period_cost_matrix:
                        period_cost_matrix.update({
                            t + life_time: {
                                "cost_link_matrix": np.zeros((n, n)),
                                "cost_node_matrix": np.zeros(n),
                                "released_numbers": 0
                            }
                        })
                    period_cost_matrix[t + life_time]["cost_link_matrix"] += cost_matrix["cost_link_matrix"]
                    period_cost_matrix[t + life_time]["cost_node_matrix"] += cost_matrix["cost_node_matrix"]
                    period_cost_matrix[t + life_time]["released_numbers"] += 1

                    print("当前映射情况，映射成功虚拟请求数量：{}，总虚拟请求数量：{}".format(success_num,total_num))

                # # 计算性能
                longterm_rc = utils.get_revenue_cost_ratio(longterm_revenue, longterm_cost)
                # node_ut = utils.get_node_utilization(current_sn_nodes=s_nodes, original_sn_nodes=original_s_nodes)
                # link_ut = utils.get_link_utilization(current_sn_links=s_links, original_sn_links=original_s_links)
                # rc.append('{:.3f}'.format(longterm_rc))
                # node_utilization.append('{:.3f}'.format(node_ut))
                # link_utilization.append('{:.3f}'.format(link_ut))
                # all_time.append(t)
                # accept_ratio.append(count_success / count)

            # 计算性能
            longterm_rc = utils.get_revenue_cost_ratio(longterm_revenue, longterm_cost)
            # node_ut = utils.get_node_utilization(current_sn_nodes=s_nodes, original_sn_nodes=original_s_nodes)
            # link_ut = utils.get_link_utilization(current_sn_links=s_links, original_sn_links=original_s_links)
            rc.append('{:.3f}'.format(longterm_rc))
            # node_utilization.append('{:.3f}'.format(node_ut))
            # link_utilization.append('{:.3f}'.format(link_ut))
            all_time.append(t)
            accept_ratio.append(success_num / total_num)

        print("当前映射成功的虚拟网络数量为：{},虚拟网络总数量为{}".format(success_num,total_num))
        if net:
            # 保存模型
            torch.save(net, load_model_path)

        df = pd.DataFrame()
        df["time"] = all_time
        df["r/c"] = rc
        df["accept_ratio"] = accept_ratio

        try:
            df.to_csv(config.ResultFile + "/{}_{}_{}.csv".format(load_model_path,config.batch_size,config.iter_time))
        except:
            df.to_csv("./data/result/{}_{}_{}.csv".format(load_model_path,config.batch_size,config.iter_time))