# -*- coding: utf-8 -*-

"""
数据预处理
"""

import numpy as np
import pandas as pd
import re

def get_SN_Node(file):  # 从文件中读取物理网络的结点，返回以时序为key，以矩阵为value的字典
    SN_dict = {}
    fo = open(file, "r")
    pattern = re.compile(r'This is PS network for virtual network----')
    line = fo.readline()
    while line:
        if re.match(pattern, line):  # 匹配pattern语句
            # line=fo.readline() #读取下一行
            num = [int(r) for r in re.findall(r"\d+\.?\d*", line)]  # 提取数字存为list
            key = num[0]
            value = np.zeros([148, 148])  # 定义一个空矩阵
            for i in range(148):  # 读取结点存入矩阵
                line = fo.readline()
                num = [int(r) for r in re.findall(r"\d+\.?\d*", line)]
                value[num[0]][num[0]] = num[1]
            SN_dict.update({key: value})  # 存入字典
        line = fo.readline()
    fo.close()
    return SN_dict


def get_VN_Node(file):  # 从文件中读取虚拟网络的结点，返回以时序为key，以矩阵为value的字典
    VN_dict = {}
    vector = []
    value = []
    fo = open(file, "r")
    pattern = re.compile(r'This is  virtual network')  # pattern=re.compile(r'This is virtual network----')#开始标志
    end_pattern = re.compile(r'node-link-information:')  # 结束标志
    line = fo.readline()
    while line:
        if re.match(pattern, line):  # 匹配pattern语句
            num = [int(r) for r in re.findall(r"\d+\.?\d*", line)]  # 提取数字存为list
            key = num[0]
            vector = []
            value = []
            #            value=np.zeros([10,10]) #定义一个空矩阵
            line = fo.readline()
            while not re.match(end_pattern, line) and line.strip() != '':  # 读取结点存入矩阵
                num = [int(r) for r in re.findall(r"\d+\.?\d*", line)]
                vector.append(num[0])
                value.append(num[1])
                #                value[num[0]%10][num[0]%10]=num[1] #虚拟网络最大节点数为10
                line = fo.readline()
            size = len(vector)
            values = np.zeros([size, size])
            min_v = min(vector)
            for i in range(len(vector)):
                vector[i] = vector[i] - min_v
                values[vector[i]][vector[i]] = value[i]
            VN_dict.update({key: values})
        line = fo.readline()
    fo.close()
    return VN_dict

def get_SN_Link(file, SN_dict):  # 读取物理网络的边
    fo = open(file, "r")
    SN_key_pattern = re.compile(r'This is PS network for virtual network----')  # 判断key
    link_pattern = re.compile(r'node-link-information:')  # 判断link information
    SN_end_pattern = re.compile(r'This is virtual network')  # 判断结束 结尾处要么是空行 要么是vn
    line = fo.readline()
    while line:
        if re.match(SN_key_pattern, line):  # 找到SN
            # line=fo.readline() #读取下一行
            num = [int(r) for r in re.findall(r"\d+\.?\d*", line)]  # 将一行数字处理成list
            key = num[0]  # SN的序号
            value = SN_dict.get(key)
            line = fo.readline()  # 读取下一行
            while not re.match(link_pattern, line):
                line = fo.readline()
            line = fo.readline()  # 读取下一行
            while (not re.match(SN_end_pattern, line)) and line.strip() != '':
                num = [float(r) for r in re.findall(r"\d+\.?\d*", line)]
                value[int(num[0])][int(num[1])] = num[2]
                value[int(num[1])][int(num[0])] = num[2]
                line = fo.readline()  # 读取下一行
            SN_dict.update({key: value})
        line = fo.readline()
    fo.close()
    return SN_dict

def get_VN_Link(file, VN_dict):  # 读取虚拟网路的边
    fo = open(file, "r")
    VN_key_pattern = re.compile(r'This is  virtual network')  # 判断key
    link_pattern = re.compile(r'node-link-information:')  # 判断link information
    VN_end_pattern = re.compile(r'The life time is:')  # 判断结束
    line = fo.readline()
    while line:
        if re.match(VN_key_pattern, line):  # 找到VN
            num = [int(r) for r in re.findall(r"\d+\.?\d*", line)]  # 将一行数字处理成list
            key = num[0]  # VN的序号
            vector1 = []
            vector2 = []
            value = []

            values = VN_dict.get(key)
            line = fo.readline()  # 读取下一行
            while not re.match(link_pattern, line):
                line = fo.readline()
            line = fo.readline()  # 读取下一行
            while not re.match(VN_end_pattern, line) and line.strip() != '':
                # print(line)
                num = [float(r) for r in re.findall(r"\d+\.?\d*", line)]
                vector1.append(num[0])
                vector2.append(num[1])
                value.append(num[2])
                line = fo.readline()  # 读取下一行
            if len(vector1) > 0:
                min_v1 = min(vector1)
                min_v2 = min(vector2)
                min_v = min(min_v1, min_v2)
            for i in range(len(vector1)):
                vector1[i] = int(vector1[i] - min_v)
                vector2[i] = int(vector2[i] - min_v)
                values[vector1[i]][vector2[i]] = value[i]
                values[vector2[i]][vector1[i]] = value[i]

            VN_dict.update({key: values})
        line = fo.readline()
    fo.close()
    return VN_dict


def get_SN_Path(file):  # 读取物理网络的边，按三元组方式存，即(v1,v2,vbandwith)
    SN_path = {}
    fo = open(file, "r")
    SN_key_pattern = re.compile(r'This is PS network for virtual network----')  # 判断时序
    link_pattern = re.compile(r'node-link-information:')  # 判断link information
    SN_end_pattern = re.compile(r'This is virtual network')  # 判断结束 结尾处要么是空行 要么是vn
    line = fo.readline()
    while line:
        if re.match(SN_key_pattern, line):  # 找到SN
            # line=fo.readline() #读取下一行
            num = [int(r) for r in re.findall(r"\d+\.?\d*", line)]  # 将一行数字处理成list
            key = num[0]  # SN的序号
            value = []
            line = fo.readline()  # 读取下一行
            while not re.match(link_pattern, line):
                line = fo.readline()
            line = fo.readline()  # 读取下一行
            while (not re.match(SN_end_pattern, line)) and (line != '\n'):
                num = [float(r) for r in re.findall(r"\d+\.?\d*", line)]
                node_list = [int(num[0]), int(num[1]), num[2]]
                value.append(node_list)
                line = fo.readline()  # 读取下一行
            SN_path.update({key: value})
        line = fo.readline()
    fo.close()
    return SN_path

def get_VN_Path(file):  # 读取虚拟网络的边，按三元组方式存，即(v1,v2,vbandwith)
    VN_path = {}
    fo = open(file, "r")
    VN_key_pattern = re.compile(r'This is  virtual network')  # 判断时序
    link_pattern = re.compile(r'node-link-information:')  # 判断link information
    VN_end_pattern = re.compile(r'The life time is:')  # 判断结束
    line = fo.readline()
    while line:
        if re.match(VN_key_pattern, line):  # 找到VN
            # line=fo.readline() #读取下一行
            num = [int(r) for r in re.findall(r"\d+\.?\d*", line)]  # 将一行数字处理成list
            key = num[0]  # VN的序号
            vector1 = []
            vector2 = []
            value = []
            values = []

            line = fo.readline()  # 读取下一行
            while not re.match(link_pattern, line):
                line = fo.readline()
            line = fo.readline()  # 读取下一行
            while (not re.match(VN_end_pattern, line)) and (line != '\n'):
                num = [float(r) for r in re.findall(r"\d+\.?\d*", line)]
                vector1.append(num[0])
                vector2.append(num[1])
                value.append(num[2])

                #                node_list=(int(num[0]%6),int(num[1]%6),num[2])     #固定6个节点
                #                values.append(node_list)
                line = fo.readline()  # 读取下一行
            if len(vector1) > 0:
                min_v1 = min(vector1)
                min_v2 = min(vector2)
                min_v = min(min_v1, min_v2)
            for i in range(len(vector1)):
                vector1[i] = int(vector1[i] - min_v)
                vector2[i] = int(vector2[i] - min_v)
                node_list = (vector1[i], vector2[i], value[i])
                values.append(node_list)
            VN_path.update({key: values})
        line = fo.readline()
    fo.close()
    return VN_path

def get_Solution(file):  # 从文件中读取的映射结果
    MP_dict = {}
    fo = open(file, "r")
    pattern = re.compile(r'This is MP Solution for virtual network')  # 开始标志
    # bad_pattern=re.compile(r'.*\[.*\].*')#不读入的行
    end_pattern = re.compile(r'link MP Solution:')  # 结束标志
    line = fo.readline()
    while line:
        if re.match(pattern, line):
            num = [int(r) for r in re.findall(r"\d+\.?\d*", line)]  # 将一行数字处理成list
            key = num[0]  # MP的序号
            value = np.zeros([148, 10])
            line = fo.readline()  # 读取下一行
            while not re.match(end_pattern, line):  # and not re.match(bad_pattern,line):
                num = [int(r) for r in re.findall(r"\d+\.?\d*", line)]  # 将一行数字处理成list
                value[num[1]][num[0] % 10] = 1
                line = fo.readline()  # 读取下一行
            MP_dict.update({key: value})
        line = fo.readline()  # 读取下一行
    fo.close()
    return MP_dict

def get_period(file):  # 从文件中读取虚拟网络的生命周期，返回VN_period={时序key:[[序号num,生命周期period,结束时间end_time],]}
    VN_path = {}
    fo = open(file, "r")
    VN_key_pattern = re.compile(r'This is  virtual network')  # 判断序号
    time_pattern = re.compile(r'The life time is:')  # 判断life time
    line = fo.readline()
    value = []
    while line:
        if re.match(VN_key_pattern, line):  # 找到VN
            # line=fo.readline() #读取下一行
            num = [int(r) for r in re.findall(r"\d+\.?\d*", line)]  # 将一行数字处理成list
            key = num[0]  # VN的序号
            while not re.match(time_pattern, line):
                line = fo.readline()
            life_time = [int(r) for r in re.findall(r"\d+\.?\d*", line)]
            item = [key, life_time[0], 0, 0]
            value.append(item)
        line = fo.readline()  # 读取下一行
    VN_period = {0: value}
    fo.close()
    return (VN_period)

def read_SN_VN(file1, file2):
    print("文件地址为:{};{}".format(file1,file2))
    print("正在读取数据中....")
    solution = get_Solution(file1)
    sn_link = get_SN_Link(file1, get_SN_Node(file1))
    vn_link = get_VN_Link(file2, get_VN_Node(file2))
    sn_path = get_SN_Path(file1)

    SN_Link = sn_path[0]
    VN_Link = get_VN_Path(file2)
    VN_Life = get_period(file2)
    SN_Node = []
    for i in range(len(sn_link[0])):
        SN_Node.append(sn_link[0][i][i])
    VN_Node = {}
    for i in range(len(vn_link)):
        v_node = []
        for j in range(len(vn_link[i])):
            v_node.append(vn_link[i][j][j])
        VN_Node.update({i: v_node})
    print("数据读取完毕！！！")
    print("*"*100)
    return (solution, SN_Link, SN_Node, VN_Link, VN_Node, VN_Life)

def get_CostRatio_UtilizationRate(file):  # 从文件中读取资源利用率和成本比，返回CostRatio和UtilizationRate两个字典
    fo = open(file, "r", encoding="gbk")
    line = fo.readline()
    # print(type(line))
    pattern1 = re.compile(r'.*当前接受的虚拟网络数为.*')  # 匹配有用行
    pattern2 = re.compile(r'(?<=-)\d+(?=\.\d-当前接受的虚拟网络数为)')  # 匹配第一个数字（时序）
    # pattern5=re.compile(r'(?#-\d+\.\d)(?<=-当前接受的虚拟网络数为-)\d+(?=-)')#匹配网络的序号
    pattern3 = re.compile(
        r'(?#-\d+\.\d-当前接受的虚拟网络数为-\d+-total fitness is:-\d+\.\d+-total cost is-\d+\.\d+-benifit-)(?<=cost ratio is:-)\d+\.\d+(?=-)')  # 匹配cost_ratio
    pattern4 = re.compile(
        r'(?#-\d+\.\d-当前接受的虚拟网络数为-\d+-total fitness is:-\d+\.\d+-total cost is-\d+\.\d+-benifit-cost ratio is:-\d+\.\d+)(?<=-Utilization rate is:-)\d+\.\d+(?=-)')  # 匹配utilization_rate
    CostRatio = {}
    UtilizationRate = {}
    while line:
        # print(line)
        if re.match(pattern1, line):
            # print("ok")
            key = int(re.findall(pattern2, line)[0])  # 读取时序作为键值
            print(key)
            if key not in CostRatio:
                Cvalue = []
                CostRatio.update({key: Cvalue})
            if key not in UtilizationRate:
                Uvalue = []
                UtilizationRate.update({key: Uvalue})
            # num=int(re.findall(pattern4,line)[0])#序号
            cost_ratio = float(re.findall(pattern3, line)[0])
            utilization_rate = float(re.findall(pattern4, line)[0])
            # Cvalue_item=[num,cost_ratio]
            # Uvalue_item=[num,utilization_rate]
            # CostRatio[key].append(Cvalue_item)
            # UtilizationRate[key].append(Uvalue_item)
            CostRatio[key].append(cost_ratio)
            UtilizationRate[key].append(utilization_rate)
        line = fo.readline()
    fo.close()
    return CostRatio, UtilizationRate

def loda_data(file1,file2):
    pass

#######################################################################################################################
import networkx as nx
import random
import numpy as np
from util import config
from draw import plot_data
# 通过networkx生成数据

def create_data(G,min_link_resource,max_link_resource,
                min_node_resource,max_node_resource):
    for (start, end) in G.edges:
        G.edges[start, end]['weight'] = float(random.randint(min_link_resource,max_link_resource+1))
    for i in G.nodes:
        G.nodes[i]["weight"] = float(random.randint(min_node_resource,max_node_resource+1))
    return G

def create_physical_network_data(n,p,min_link_resource=50,max_link_resource=100,
                min_node_resource=50,max_node_resource=100):
    G = nx.erdos_renyi_graph(n,p)
    # graph = nx.watts_strogatz_graph(n,p)
    # graph = nx.barabasi_albert_graph(n,p)
    # graph = nx.random_lobster(n,p)
    return create_data(G,min_link_resource,max_link_resource,
                       min_node_resource,max_node_resource)

def create_virutal_network_data(p=.5,min_nodes_numbers=2,max_nodes_numbers=12,min_link_resource=1,max_link_resource=50,
                min_node_resource=1,max_node_resource=50):
    n = random.randint(min_nodes_numbers,max_nodes_numbers+1)
    G = nx.erdos_renyi_graph(n, p)
    G = create_data(G, min_link_resource, max_link_resource,
                       min_node_resource, max_node_resource)
    return G

def create_virtual_network_lift_time(size,min_life_time=100,max_life_time=900,lambda_=1000):
    # return np.random.randint(min_life_time,max_life_time+1, size=size)
    return np.array([np.ceil(random.expovariate(1/lambda_) ) for _ in range(size)]).astype(np.int)

def get_sn_data():
    Sn = create_physical_network_data(100,.2)
    SN_Link = Sn.edges.data("weight")
    SN_Node = Sn.nodes.data("weight")
    new_Sn_Link = []
    new_SN_Node = []
    for link in SN_Link:
        new_Sn_Link.append(list(link))

    for node in SN_Node:
        new_SN_Node.append(node[1])

    return new_Sn_Link,new_SN_Node

def get_vn_life_time_data(n):
    life_time = create_virtual_network_lift_time(n)
    new_lift_time = []
    for i in range(len(life_time)):
        new_lift_time.append([i,life_time[i],0,0])
    return new_lift_time

def get_vn_data(n):
    VN_Link = {}
    Vn_Node = {}
    VN_Life = {}
    for i in range(n):
        vn = create_virutal_network_data()
        links = vn.edges.data("weight")
        nodes = vn.nodes.data("weight")
        nodes = [node[1] for node in nodes]
        Vn_Node.update({i:nodes})
        VN_Link.update({i:list(links)})
    life_time = get_vn_life_time_data(n)
    VN_Life.update({0:life_time})
    return Vn_Node,VN_Link,VN_Life

import os,json
def save_data(data,n):

    SN_Link = np.array(data["SN_Link"]).tolist()
    SN_Node = np.array(data["SN_Node"]).tolist()
    VN_Link = {key:np.array(value).tolist() for key,value in data["VN_Link"].items()}
    VN_Node = {key:np.array(value).tolist() for key,value in data["VN_Node"].items()}
    VN_Life = {key:np.array(value).tolist() for key,value in data["VN_Life"].items()}
    VN_Arrive_Time = np.array(data["VN_Arrive_Time"]).tolist()
    a = {
        "SN_Link": SN_Link,
        "SN_Node": SN_Node,
        "VN_Node": VN_Node,
        "VN_Link": VN_Link,
        "VN_Life": {0: VN_Life},
        "VN_Arrive_Time": VN_Arrive_Time,
    }
    b = json.dumps(a)
    try:
        f = open('../../data/data/{}.json'.format(n), 'w')
        f.write(b)
        f.close()
        np.save('../../data/data/{}.npy'.format(n),data)
    except:
        f = open('../data/data/{}.json'.format(n), 'w')
        f.write(b)
        f.close()
        np.save('../data/data/{}.npy'.format(n), data)
    print("数据保存完毕")


def get_data(n=1000):
    SN_Link, SN_Node = get_sn_data()
    VN_Node,VN_Link,VN_Life = get_vn_data(n)

    data = {
        "SN_Link": SN_Link,
        "SN_Node": SN_Node,
        "VN_Node": VN_Node,
        "VN_Link": VN_Link,
        "VN_Life": VN_Life,
    }

    save_data(data,n)

# 服从泊松分布
def arrive_time(n):
    arrive = np.random.poisson(20,n)
    arrive = np.cumsum(arrive)
    # print(arrive)
    return arrive

def load_data(n):
    try:
        data = np.load('../../data/data/{}.npy'.format(n), allow_pickle=True).item()
    except:
        data = np.load('../data/data/{}.npy'.format(n), allow_pickle=True).item()
    return data

TEST = False
numbers = 1000
if TEST:
    get_data(numbers)  # 测试
    # data = load_data(numbers)
    # arrive_time(1000)


# data = load_data(numbers)
# save_data(data,numbers)