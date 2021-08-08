from vine.utils.graph_utils import create_network_graph, from_networkx
from vine.graphViNE.model import cluster_using_argva

def main():
    physical_graph =  create_network_graph(nodes_num=100)

    data = from_networkx(physical_graph, normalize=True)
    data.edge_attr = data.edge_attr / data.edge_attr.max()

    model = cluster_using_argva(data, verbose=True, gpu=False)

if __name__ == "__main__":
    main()