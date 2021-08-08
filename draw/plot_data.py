import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from util import config
import pandas as pd
import os

def create_graph(matrix):
    fig, ax = plt.subplots()
    G = nx.from_numpy_matrix(np.matrix(matrix), create_using=nx.Graph)
    for v in G.nodes():
        G.nodes[v]['c'] = "c:{}".format(matrix[v][v])

    for n in G.edges():
        if n[0] != n[1]:
            G.edges[n[0], n[1]]['b'] = 'b:{}'.format(matrix[n[0]][n[1]])

    layout = nx.spring_layout(G)
    edge_labels = nx.get_edge_attributes(G, 'b')
    node_labels = nx.get_node_attributes(G, 'c')
    nx.draw(G, layout, node_size=500)
    nx.draw_networkx_edge_labels(G, pos=layout, edge_labels=edge_labels, font_size=6)
    nx.draw_networkx_labels(G, layout, labels=node_labels, font_size=6)
    plt.tight_layout()
    plt.show()

def plot_data(x,y,x_label,y_label,title):
    x_ticks = [i for i in x if i % 2000 == 0]
    x_ticks.append(x[-1])
    plt.plot(x,y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xlim(2000,x[-1])
    plt.ylim(.3,1.0)
    plt.xticks(x_ticks,x_ticks)
    plt.grid(b=True)
    plt.title(title)
    plt.show()

def plot_result_data():
    path = "../data/result"
    files = os.listdir(path)
    new_files = []
    MAX_LEN = 0
    for file in files:
        if MAX_LEN == 0:
            MAX_LEN = len(pd.read_csv(os.path.join(path, file)))
        file = file.split('_')[0]
        new_files.append(os.path.join(path, file))

    columns = ["r/c","accept_ratio","revenus","time"]

    time_step = 18000

    performance_acceptance_ratio = np.zeros((len(files),time_step),dtype=float)
    performance_revenue = np.zeros((len(files),time_step),dtype=float)
    performance_rc_ratio = np.zeros((len(files),time_step),dtype=float)

    FIGURE_START_TIME_STEP = 2000
    TIME_WINDOW_SIZE = 1
    x_range = range(FIGURE_START_TIME_STEP,time_step,TIME_WINDOW_SIZE)

    plt.style.use('seaborn-dark-palette')

    for i in range(len(files)):
        data = pd.read_csv(os.path.join(path, files[i]))
        data = data.iloc[:,1:]
        performance_acceptance_ratio[i] = np.array(data["accept_ratio"][:time_step])
        try:
            performance_revenue[i] = np.array(data["revenus"][:time_step])
        except:
            pass
        performance_rc_ratio[i] = np.array(data["r/c"][:time_step])

    plt.rcParams['figure.constrained_layout.use'] = True

    for id in range(len(files)):
        plt.plot(
            x_range,
            performance_acceptance_ratio[id,FIGURE_START_TIME_STEP:time_step:TIME_WINDOW_SIZE],
            label=files[id].split('_')[0]
        )
    plt.xlim(x_range[0], x_range[-1])
    plt.ylim(.3, 1.0)
    plt.ylabel("accept ratio")
    plt.xlabel("Time unit")
    plt.legend(loc="best", fancybox=True, framealpha=0.3, fontsize=12)
    plt.grid(True)
    plt.show()

# plot_result_data()