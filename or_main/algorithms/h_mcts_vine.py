from or_main.algorithms.a_baseline import BaselineVNEAgent
from or_main.common import utils
from mcst_vne.mcts import MCTS
import networkx as nx

class MCST_VNEAgent(BaselineVNEAgent):
    def __init__(
            self, beta, logger, time_window_size, agent_type, type_of_virtual_node_ranking,
            allow_embedding_to_same_substrate_node, max_embedding_path_length
    ):
        super(MCST_VNEAgent, self).__init__(
            logger, time_window_size, agent_type, type_of_virtual_node_ranking,
            allow_embedding_to_same_substrate_node, max_embedding_path_length
        )
        self.beta = beta
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

    def find_substrate_nodes(self, copied_substrate, vnr):
        '''
        Execute Step 1
        :param copied_substrate: copied substrate network
        :param vnr: virtual network request
        :return: embedded substrate nodesƒ
        '''
        embedding_s_nodes = {}

        s_nodes = copied_substrate.net.nodes.data("CPU")
        s_links = copied_substrate.net.edges.data("bandwidth")

        v_nodes = vnr.net.nodes.data("CPU")
        v_links = vnr.net.edges.data("bandwidth")

        sn = self.create_graph(s_nodes,s_links)
        vnr = self.create_graph(v_nodes,v_links)

        mapping_nodes = self.mcts.run(sn, vnr)
        for k,v in mapping_nodes.items():
            embedding_s_nodes.update({k:tuple(v)})
        return embedding_s_nodes
