import torch
from torch.nn import Linear
import torch.nn.functional as F
from sklearn.cluster import KMeans,SpectralClustering,MiniBatchKMeans
from sklearn.metrics import accuracy_score, silhouette_score, davies_bouldin_score
from torch_geometric.nn.conv import MessagePassing
from vine.auto_encoder import InnerProductDecoder, ARGVA
from vine.gateViNE.layer import GATConv
from algorithms.Optimizer.AdaxModel import AdaX,AdaXW
from torch_geometric.nn import GATv2Conv,GCN2Conv
import torch_geometric

"""
图注意力自动编码器
"""

class Discriminator(torch.nn.Module):
    def __init__(self, in_channel, hidden, out_channel):
        super(Discriminator, self).__init__()
        self.dense1 = torch.nn.Linear(in_channel, hidden)
        self.dense2 = torch.nn.Linear(hidden, out_channel)
        self.dense3 = torch.nn.Linear(out_channel, out_channel)

    def forward(self, x):
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = self.dense3(x)
        return x

class Encoder(torch.nn.Module):
    def __init__(self, in_channels, hidden, out_channels):
        super(Encoder, self).__init__()
        self.conv1 = GATConv(in_channels, hidden)
        self.conv2 = GATConv(hidden, hidden*2)
        self.conv3 = GATConv(hidden*2, hidden*3)
        self.conv4 = GATConv(hidden*3, hidden*4)
        self.conv_mu = GATConv(hidden*4, out_channels)
        self.conv_logvar = GATConv(hidden*4, out_channels)

    def forward(self, x, edge_index, edge_attr):
        # 为了防止过平滑
        # edge_index, edge_attr = torch_geometric.utils.dropout_adj(edge_index=edge_index,edge_attr=edge_attr,p=.1)
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        # edge_index, edge_attr = torch_geometric.utils.dropout_adj(edge_index=edge_index, edge_attr=edge_attr, p=.1)
        x = F.elu(self.conv2(x, edge_index, edge_attr))
        # edge_index, edge_attr = torch_geometric.utils.dropout_adj(edge_index=edge_index, edge_attr=edge_attr, p=.1)
        x = F.elu(self.conv3(x, edge_index, edge_attr))
        # edge_index, edge_attr = torch_geometric.utils.dropout_adj(edge_index=edge_index, edge_attr=edge_attr, p=.1)
        x = F.elu(self.conv4(x, edge_index, edge_attr))
        return self.conv_mu(x, edge_index, edge_attr), self.conv_logvar(x, edge_index, edge_attr)

class GATDecoder(torch.nn.Module):
    def __init__(self, input_numbers, hidden_depth, link_creator_numbers, output_numbers):
        super(GATDecoder, self).__init__()
        # self.dense1 = torch.nn.Linear(input_numbers, hidden_depth)
        # self.dense2 = torch.nn.Linear(hidden_depth, output_numbers)
        # self.dense3 = torch.nn.Linear(hidden_depth, link_creator_numbers)

        self.dense1 = GATConv(input_numbers, hidden_depth)
        self.dense2 = GATConv(hidden_depth, hidden_depth*2)
        self.dense3 = GATConv(hidden_depth*2, hidden_depth*3)
        self.dense4 = GATConv(hidden_depth*3, hidden_depth*4)
        self.dense_a = torch.nn.Linear(hidden_depth*4, link_creator_numbers)
        self.dense_z = torch.nn.Linear(hidden_depth*4, output_numbers)
        self.inner_product = InnerProductDecoder()

    def forward(self, z, edge_index, sigmoid=True):
        z = F.leaky_relu(self.dense1(z,edge_index))
        z = F.leaky_relu(self.dense2(z,edge_index))
        z = F.leaky_relu(self.dense3(z,edge_index))
        z = F.leaky_relu(self.dense4(z,edge_index))
        zprim = self.dense_z(z)
        aprim = self.dense_a(z)
        return zprim, self.inner_product(aprim, edge_index)


def cluster_using_argva(data, verbose=True, max_epoch=150, pre_trained_model=None, gpu=True):
    data = data.clone()

    def train(data):
        model.train()
        model_optimizer.zero_grad()
        z = model.encode(data.x, data.edge_index, data.edge_attr)

        for i in range(5):
            discriminator.train()
            discriminator_optimizer.zero_grad()
            discriminator_loss = model.discriminator_loss(z)
            discriminator_loss.backward()
            discriminator_optimizer.step()

        regularization_loss = 0
        for param in model.parameters():
            regularization_loss += torch.sum(abs(param))

        loss = model.gat_loss(z, data.edge_index, data.x)
        loss = loss + (1 / data.num_nodes) * model.kl_loss() + 1e-4 * regularization_loss

        # b_loss = .1314159
        # loss = (loss - b_loss).abs() + b_loss # flooding

        loss.backward()
        model_optimizer.step()
        return loss

    def test(data):
        model.eval()
        kmeans_input = []
        with torch.no_grad():
            z = model.encode(data.x, data.edge_index, data.edge_attr)
            kmeans_input = z.cpu().data.numpy()
        pred_label = MiniBatchKMeans(n_clusters=4).fit_predict(kmeans_input)
        X = data.x.cpu().data

        s = silhouette_score(X, pred_label)
        davies = davies_bouldin_score(X, pred_label)

        return s, davies

    encoder = Encoder(data.num_features, 16, 16)
    discriminator = Discriminator(16, 32, 16)
    decoder = GATDecoder(16, 16, 8, data.num_features)
    model = None
    if pre_trained_model is not None:
        model = pre_trained_model
    else:
        model = ARGVA(encoder, discriminator, decoder=decoder)

    # discriminator_optimizer = torch.optim.Adam(
    #     model.discriminator.parameters(), lr=0.001,weight_decay=1e-4)
    # model_optimizer = torch.optim.Adam(model.parameters(), lr=5 * 0.001,weight_decay=1e-4)

    discriminator_optimizer = torch.optim.Adam(model.discriminator.parameters(), lr=1e-4,weight_decay=1e-4)
    model_optimizer = torch.optim.Adam(model.parameters(), lr=1e-4,weight_decay=1e-4)

    device = torch.device(
        'cuda' if torch.cuda.is_available() and gpu else 'cpu')
    discriminator.to(device)
    model.to(device)
    data = data.to(device)

    for epoch in range(max_epoch):
        loss = train(data)
        if verbose or epoch == max_epoch - 1:
            s, davies = test(data)
            print('Epoch: {:05d}, '
                  'Train Loss: {:.3f}, Silhoeute: {:.3f}, Davies: {:.3f}'.format(epoch, loss, s, davies))
    return model
