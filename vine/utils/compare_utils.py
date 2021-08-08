import networkx as nx
import numpy as np


def compute_revenue(req):
    revenue = 0
    for i in req.nodes:
        revenue += req.nodes[i]['CPU']
        if 'GPU' in req.nodes[i]:
            revenue += (req.nodes[i]['GPU'])
        if 'Memory' in req.nodes[i]:
            revenue += (req.nodes[i]['Memory'])

    for e in req.edges:
        revenue += req.edges[e[0], e[1]]['Bandwidth']
    return revenue


def compute_cost(req):
    cost = 0
    for i in req.nodes:
        cost += req.nodes[i]['CPU']
        if 'GPU' in req.nodes[i]:
            cost += req.nodes[i]['GPU']
        if 'Memory' in req.nodes[i]:
            cost += req.nodes[i]['Memory']

    for e in req.edges:
        num_of_occupied_edges = 0
        if 'Embedded' in req.edges[e[0], e[1]]:
            num_of_occupied_edges = len(req.edges[e[0], e[1]]['Embedded'])
        cost += req.edges[e[0], e[1]]['Bandwidth'] * num_of_occupied_edges
    return cost


def compute_utils(physical):
    cpu_util = 0
    link_util = 0

    for i in physical.nodes:
        cpu_util += 1 - (physical.nodes[i]['CPU']/physical.nodes[i]['MaxCPU'])
    cpu_util /= physical.number_of_nodes()

    for e in physical.edges:
        link_util += 1 - (physical.edges[e[0], e[1]]['Bandwidth'] /
                          physical.edges[e[0], e[1]]['MaxBandwidth'])
    link_util /= physical.number_of_edges()

    if 'GPU' in physical.nodes[0]:
        gpu_util, memory_util = 0, 0
        for i in physical.nodes:
            gpu_util += 1 - \
                (physical.nodes[i]['GPU']/physical.nodes[i]['MaxGPU'])
            memory_util += 1 - \
                (physical.nodes[i]['Memory']/physical.nodes[i]['MaxMemory'])
        gpu_util /= physical.number_of_nodes()
        memory_util /= physical.number_of_nodes()
        return cpu_util, link_util, gpu_util, memory_util
    return cpu_util, link_util
