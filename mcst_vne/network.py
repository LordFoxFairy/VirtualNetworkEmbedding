import copy
import networkx as nx
from itertools import islice


class Network:

    def __init__(self, path):
        self.files_dir = path

    def get_networks(self, sub_filename, req_num, child_num=0, granularity=1):
        """读取 req_num 个虚拟网络及 req_num*child_num 个子虚拟网络请求，构成底层虚拟网络请求事件队列和子虚拟网络请求事件队列"""
        # 底层物理网络
        sub = self.read_network_file(sub_filename, granularity)
        # 第1层虚拟网络请求
        queue1 = self.get_reqs(req_num, granularity)
        # 第2层虚拟网络请求
        queue2 = self.get_child_reqs(req_num, child_num, granularity)
        return sub, queue1, queue2

    def get_reqs(self, req_num, granularity=1):
        """读取req_num个虚拟网络请求文件，构建虚拟网络请求事件队列"""
        queue = []
        if req_num == 1000:
            offset = 2000-req_num
        else:
            offset = 0
        for i in range(req_num):
            index = i + offset
            filename = 'req%d.txt' % index
            req = self.read_network_file(filename, granularity)
            req.graph['parent'] = -1
            req.graph['id'] = index
            queue.append(req)
        return queue

    def get_reqs_for_train(self, req_num, granularity=1):
        """读取req_num个虚拟网络请求文件，构建虚拟网络请求事件队列"""
        queue = []
        for i in range(req_num):
            filename = 'req%d.txt' % i
            req = self.read_network_file(filename, granularity)
            req.graph['parent'] = -1
            req.graph['id'] = i
            queue.append(req)
        return queue

    def get_child_reqs(self, req_num, child_req_num, granularity):
        """读取子虚拟网络请求文件，构建子虚拟网络请求事件队列"""
        queue = []
        if child_req_num != 0:
            for i in range(req_num):
                for j in range(child_req_num):
                    child_req_filename = 'req%d-%d.txt' % (i, j)
                    child_req = self.read_network_file(child_req_filename, granularity)
                    child_req.graph['parent'] = i
                    child_req.graph['id'] = j
                    queue.append(child_req)
        return queue

    def read_network_file(self, filename, granularity=1):
        """读取网络文件并生成networkx.Graph实例"""

        mapped_info = {}
        node_id, link_id = 0, 0

        with open(self.files_dir + filename) as f:
            lines = f.readlines()

        # Step 1: 获取网络节点数量和链路数量，并根据网络类型进行初始化
        if len(lines[0].split()) == 2:
            """物理网络"""
            node_num, link_num = [int(x) for x in lines[0].split()]
            graph = nx.Graph(mapped_info=mapped_info)
        else:
            """虚拟网络"""
            node_num, link_num, time, duration, max_dis = [int(x) for x in lines[0].split()]
            graph = nx.Graph(type=0, time=time, duration=duration, mapped_info=mapped_info)

        # Step 2: 依次读取节点信息
        for line in lines[1: node_num + 1]:

            x, y, c, f, q = [float(x) for x in line.split()]
            if granularity == 1:
                graph.add_node(node_id,
                               x_coordinate=x, y_coordinate=y,
                               cpu=c, cpu_remain=c)
            elif granularity == 2:
                graph.add_node(node_id,
                               x_coordinate=x, y_coordinate=y,
                               cpu=c, cpu_remain=c,
                               flow=f, flow_remain=f)
            else:
                graph.add_node(node_id,
                               x_coordinate=x, y_coordinate=y,
                               cpu=c, cpu_remain=c,
                               flow=f, flow_remain=f,
                               queue=q, queue_remain=q)
            node_id = node_id + 1

        # Step 3: 依次读取链路信息
        for line in lines[-link_num:]:
            """依次读取链路信息"""
            src, dst, bw, dis = [float(x) for x in line.split()]
            graph.add_edge(int(src), int(dst), link_id=link_id, bw=bw, bw_remain=bw, distance=dis)
            link_id = link_id + 1

        # Step 4: 返回网络实例
        return graph

    @staticmethod
    def get_path_capacity(sub, path):
        """找到一条路径中带宽资源最小的链路并返回其带宽资源值"""

        bandwidth = 1000
        head = path[0]
        for tail in path[1:]:
            if sub[head][tail]['weight'] <= bandwidth:
                bandwidth = sub[head][tail]['weight']
            head = tail
        return bandwidth

    @staticmethod
    def calculate_adjacent_bw(graph, u, kind='bw'):
        """计算一个节点的相邻链路带宽和，默认为总带宽和，若计算剩余带宽资源和，需指定kind属性为bw-remain"""

        bw_sum = 0
        for v in graph.neighbors(u):
            bw_sum += graph[u][v][kind]
        return bw_sum

    @staticmethod
    def k_shortest_path(graph, source, target, k=5):
        """K最短路径算法"""
        return list(islice(nx.shortest_simple_paths(graph, source, target), k))

    @staticmethod
    def cut_then_find_path(sub, req, node_map):
        """求解链路映射问题"""

        link_map = {}
        sub_copy = copy.deepcopy(sub)

        for vLink in req.edges:
            vn_from, vn_to = vLink[0], vLink[1]
            resource = req[vn_from][vn_to]['bw']
            # 剪枝操作，先暂时将那些不满足当前待映射虚拟链路资源需求的底层链路删除
            sub_tmp = copy.deepcopy(sub_copy)
            sub_edges = []
            for sLink in sub_tmp.edges:
                sub_edges.append(sLink)
            for edge in sub_edges:
                sn_from, sn_to = edge[0], edge[1]
                if sub_tmp[sn_from][sn_to]['bw_remain'] <= resource:
                    sub_tmp.remove_edge(sn_from, sn_to)

            # 在剪枝后的底层网络上寻找一条可映射的最短路径
            sn_from, sn_to = node_map[vn_from], node_map[vn_to]
            if nx.has_path(sub_tmp, source=sn_from, target=sn_to):
                path = Network.k_shortest_path(sub_tmp, sn_from, sn_to, 1)[0]
                link_map.update({vLink: path})

                # 这里的资源分配是暂时的
                start = path[0]
                for end in path[1:]:
                    bw_tmp = sub_copy[start][end]['bw_remain'] - resource
                    sub_copy[start][end]['bw_remain'] = round(bw_tmp, 6)
                    start = end
            else:
                break

        # 返回链路映射集合
        return link_map

    @staticmethod
    def find_path(sub, req, node_map, k=1):
        """求解链路映射问题"""

        link_map = {}
        for vLink in req.edges:
            vn_from = vLink[0]
            vn_to = vLink[1]
            sn_from = node_map[vn_from]
            sn_to = node_map[vn_to]
            if nx.has_path(sub, source=sn_from, target=sn_to):
                for path in Network.k_shortest_path(sub, sn_from, sn_to, k):
                    if Network.get_path_capacity(sub, path) >= req[vn_from][vn_to]['weight']:
                        link_map.update({vLink: path})
                        break
                    else:
                        continue

        # 返回链路映射集合
        return link_map

    @staticmethod
    def allocate(sub, req, node_map, link_map, granularity=1):
        """分配节点和链路资源"""

        # 分配节点资源
        for v_id, s_id in node_map.items():
            cpu_tmp = sub.nodes[s_id]['cpu_remain'] - req.nodes[v_id]['cpu']
            sub.nodes[s_id]['cpu_remain'] = round(cpu_tmp, 6)
            if granularity > 1:
                flow_tmp = sub.nodes[s_id]['flow_remain'] - req.nodes[v_id]['flow']
                sub.nodes[s_id]['flow_remain'] = round(flow_tmp, 6)
                if granularity > 2:
                    queue_tmp = sub.nodes[s_id]['queue_remain'] - req.nodes[v_id]['queue']
                    sub.nodes[s_id]['queue_remain'] = round(queue_tmp, 6)

        # 分配链路资源
        for vl, path in link_map.items():
            link_resource = req[vl[0]][vl[1]]['bw']
            start = path[0]
            for end in path[1:]:
                bw_tmp = sub[start][end]['bw_remain'] - link_resource
                sub[start][end]['bw_remain'] = round(bw_tmp, 6)
                start = end

        # 更新映射信息
        mapped_info = sub.graph['mapped_info']
        mapped_info.update({req.graph['id']: (node_map, link_map)})
        sub.graph['mapped_info'] = mapped_info

    @staticmethod
    def recover(sub, req, granularity=1):
        """收回节点和链路资源"""

        req_id = req.graph['id']
        mapped_info = sub.graph['mapped_info']
        if req_id in mapped_info.keys():
            print("\nRelease the resources which are occupied by request%s" % req_id)

            # 读取该虚拟网络请求的映射信息
            node_map = mapped_info[req_id][0]
            link_map = mapped_info[req_id][1]

            # 释放节点资源
            for v_id, s_id in node_map.items():
                cpu_tmp = sub.nodes[s_id]['cpu_remain'] + req.nodes[v_id]['cpu']
                sub.nodes[s_id]['cpu_remain'] = round(cpu_tmp, 6)
                if granularity > 1:
                    flow_tmp = sub.nodes[s_id]['flow_remain'] + req.nodes[v_id]['flow']
                    sub.nodes[s_id]['flow_remain'] = round(flow_tmp, 6)
                    if granularity > 2:
                        queue_tmp = sub.nodes[s_id]['queue_remain'] + req.nodes[v_id]['queue']
                        sub.nodes[s_id]['queue_remain'] = round(queue_tmp, 6)

            # 释放链路资源
            for vl, path in link_map.items():
                link_resource = req[vl[0]][vl[1]]['bw']
                start = path[0]
                for end in path[1:]:
                    bw_tmp = sub[start][end]['bw_remain'] + link_resource
                    sub[start][end]['bw_remain'] = round(bw_tmp, 6)
                    start = end

            # 移除相应的映射信息
            mapped_info.pop(req_id)
            sub.graph['mapped_info'] = mapped_info
