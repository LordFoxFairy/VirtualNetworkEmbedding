import torch, copy
from util import config
import numpy as np
import torch.nn.functional as F

"""
端到端 虚拟网络嵌入，优化了Encoder层
"""


def clones(module, n):
    return torch.nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class AttentionLayer(torch.nn.Module):
    """
    attention layer
    """

    def __init__(self, input_dim=128, nheads=8, dropout=.05):
        super(AttentionLayer, self).__init__()

        self.nheads = nheads
        self.input_dim = input_dim

        # self.attention = torch.nn.MultiheadAttention(input_dim,nheads)

        self.w_q = torch.nn.Linear(input_dim, input_dim)
        self.w_k = torch.nn.Linear(input_dim, input_dim)
        self.w_v = torch.nn.Linear(input_dim, input_dim)

        self.fc = torch.nn.Linear(input_dim, input_dim)

        self.scale = torch.sqrt(torch.FloatTensor([input_dim // nheads]))

        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):

        if config.IS_GPU:
            query = query.cuda()
            key = key.cuda()
            value = value.cuda()
            self.scale = self.scale.cuda()


        # Q,K,V计算与变形
        bsz = query.shape[0]
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)

        Q = Q.view(bsz, -1, self.nheads, self.input_dim // self.nheads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.nheads, self.input_dim // self.nheads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.nheads, self.input_dim // self.nheads).permute(0, 2, 1, 3)

        # Q, K相乘除以scale，这是计算scaled dot product attention的第一步
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        # 然后对Q,K相乘的结果计算softmax加上dropout，这是计算scaled dot product attention的第二步：
        attention = self.dropout(torch.softmax(energy, dim=-1))

        # 第三步，attention结果与V相乘
        x = torch.matmul(attention, V)

        # 最后将多头排列好，就是multi-head attention的结果了

        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(bsz, -1, self.nheads * (self.input_dim // self.nheads))
        x = self.fc(x)

        return x


class NormLayer(torch.nn.Module):
    def __init__(self, features, eps=1e-6):
        super(NormLayer, self).__init__()
        self.gamma = torch.nn.Parameter(torch.ones(features), requires_grad=True)
        self.beta = torch.nn.Parameter(torch.zeros(features), requires_grad=True)
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        y = self.gamma * (x - mean) / (std + self.eps) + self.beta
        return y


class PositionWiseFeedForward(torch.nn.Module):
    def __init__(self, input_dim, output_dim, dropout=.05):
        super(PositionWiseFeedForward, self).__init__()
        self.w1 = torch.nn.Linear(input_dim, output_dim)
        self.w2 = torch.nn.Linear(output_dim, input_dim)
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, x):
        return self.w2(self.dropout(F.relu(self.w1(x))))


class ConnectionLayer(torch.nn.Module):
    def __init__(self, input_dim=128, dropout=.05):
        super(ConnectionLayer, self).__init__()
        self.norm = NormLayer(input_dim)
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, x, layer):
        return x + self.dropout(layer(self.norm(x)))

class EncoderLayer(torch.nn.Module):
    def __init__(self, layer: torch.nn.Module, n: int, input_dim=128):
        super(EncoderLayer, self).__init__()
        self.layers = clones(layer, n)
        self.norm = NormLayer(input_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

class Encoder(torch.nn.Module):
    def __init__(self, atten_layer, ffn_layer, input_dim=3, embedding_dim=128, hidden_dim=128, dropout=.05):
        super(Encoder, self).__init__()

        self.dropout = torch.nn.Dropout(p=dropout)

        # multi head attention layer
        self.attn = atten_layer
        # feed forward neural network
        self.ffn = ffn_layer
        # concat
        self.sublayer = clones(ConnectionLayer(input_dim=embedding_dim, dropout=dropout), 3)

        self.Embedding = torch.nn.Linear(input_dim, embedding_dim, bias=False)

        # attn
        self.w = torch.nn.Linear(hidden_dim, 2 * hidden_dim)
        self.v = torch.nn.Linear(2 * hidden_dim, 2 * hidden_dim)
        self.a = torch.nn.Linear(hidden_dim*2, 2 * hidden_dim)

        # encoder
        self.fc = torch.nn.Linear(hidden_dim, 2 * hidden_dim)

        # hn
        self.h = torch.nn.Linear(hidden_dim,2)

        # cn
        self.c = torch.nn.Linear(hidden_dim,2)


    def norm(self,x):
        """
        避免数据数值过大，显示为nan
        :param x:
        :return:
        """
        return x

    def forward(self, x):
        """
        :param x:不能将数据归一化，不然无法训练
        :return:
        """
        x = self.Embedding(x)
        atten_weight = self.sublayer[0](x, lambda func: self.attn(x, x, x))
        ffn_weight = self.sublayer[1](atten_weight, self.ffn)
        encoder = self.fc(ffn_weight)
        attn = self.v(self.w(ffn_weight) + self.a(encoder))
        hn = torch.einsum('ijl,ijk->kil', atten_weight, self.h(atten_weight))
        cn = torch.einsum('ijl,ijk->kil', ffn_weight, self.c(ffn_weight))
        return encoder,attn,(hn,cn)

class Decoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=.05):
        super(Decoder, self).__init__()

        self.dropout = torch.nn.Dropout(p=dropout)
        self.rnn = torch.nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, bidirectional=True, batch_first=True,
                                 bias=False)

    def forward(self, input, status):
        """
        :param input: (batch_size,s_node_numbers,embedding_size*2)
        :param status: (hn,cn)
        :return:
        """
        if config.IS_GPU:
            input = input.cuda().contiguous()
            status = list(status)
            status[0] = status[0].cuda().contiguous()
            status[1] = status[1].cuda().contiguous()
            status = tuple(status)

        decoder_output, decoder_status = self.rnn(input, status)

        return decoder_output, decoder_status


class GRNet(torch.nn.Module):

    def __init__(self, hidden_dim=128, batch_size=10, embedding_dim=128, input_dim=3, dropout=0.05, nheads=8, eta=1e-4):
        super(GRNet, self).__init__()
        # 参数设置
        self.eta = eta

        # 避免过拟合
        self.dropout = torch.nn.Dropout(p=dropout)

        self.embedding_dim = embedding_dim

        # 定义layer
        self.attn_layer = AttentionLayer(input_dim=embedding_dim, nheads=nheads)
        self.ffn_layer = PositionWiseFeedForward(input_dim=embedding_dim, output_dim=embedding_dim)

        # embedding
        self.Embedding = torch.nn.Linear(input_dim, embedding_dim, bias=False)

        # encoder-decoder
        self.Encoder = Encoder(atten_layer=self.attn_layer, ffn_layer=self.ffn_layer, input_dim=input_dim,
                               embedding_dim=embedding_dim)
        self.Decoder = Decoder(input_dim=embedding_dim * 2, hidden_dim=hidden_dim, dropout=.05)

        # output_weight
        self.w1 = torch.nn.Linear(2 * hidden_dim, 2 * hidden_dim)
        self.v1 = torch.nn.Linear(2 * hidden_dim, 1)
        self.a1 = torch.nn.Linear(2 * hidden_dim, 2 * hidden_dim)

        # attention_weight
        self.w2 = torch.nn.Linear(2 * hidden_dim, 2 * hidden_dim)
        self.v2 = torch.nn.Linear(2 * hidden_dim, 1)
        self.a2 = torch.nn.Linear(2 * hidden_dim, 2 * hidden_dim)

        # 损失函数
        self.cross = torch.nn.CrossEntropyLoss(reduction='none')

    def get_CrossEntropyLoss(self, output_weights, node_mappings):
        node_mappings = node_mappings.astype(float)
        node_mappings = torch.from_numpy(node_mappings).long()
        if config.IS_GPU:
            node_mappings = node_mappings.cuda()
        v_node_num = node_mappings.size()[1]
        loss = 0
        for i in range(v_node_num):
            output_weight = output_weights[i]
            if config.IS_GPU:
                output_weight = output_weight.cuda()
            loss += self.cross(
                output_weight,
                node_mappings[:, i]
            )
        return loss

    def calc_output_weight(self,n,x, y):
        """
        采取RNN的计算方式
        :param n:
        :param x:
        :param y:
        :return:
        """
        a = torch.tanh(
            self.w1(x) + self.a1(y)
        )
        y = torch.squeeze(self.v1(a))

        return y

    def calc_attention_weight(self,n, x, y):
        """
        :param n:
        :param x:
        :param y:
        :return:
        """
        output = F.softmax(
            torch.squeeze(self.v2(torch.tanh(
                self.w2(x) + self.a2(y)
            ))), dim=1
        )
        return output

    def norm(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        y = (x - mean) / (std + 1e-6)
        return y

    def get_node_mapping(self, s_node_indexes, s_inputs, v_input):
        """
        :param s_node_indexes: 节点索引,(batch_size,s_node_numbers,1)
        :param s_inputs: 物理节点特征输入，(batch_size,s_node_numbers,nfeatures)
        :param v_input:  虚拟节点特征输入，(v_node_numbers,nfeatures)
        :return:
        """
        # batch_size与节点数目
        batch_size = s_node_indexes.size(0)
        s_node_numbers = s_node_indexes.size(1)
        v_node_numbers = v_input.size(0)
        embedding_size = self.embedding_dim

        # encoder
        encoder_output, decoder_input,decoder_status = self.Encoder(s_inputs)

        batch_size = encoder_output.size(0)
        actions = torch.zeros(batch_size, s_node_numbers)

        if config.IS_GPU:
            actions = actions.cuda()

        # 保存结果
        output_weights = []
        decoder_outputs = []

        for i in range(v_node_numbers):
            encoder_output = self.dropout(encoder_output)
            decoder_output, decoder_status = self.Decoder(decoder_input, decoder_status)

            nodes_without_enough = torch.lt(s_inputs[:, :,0], v_input[i,0]).int()
            nodes_without_enough += torch.lt(s_inputs[:, :,1], v_input[i,1]).int()
            nodes_without_enough += torch.lt(s_inputs[:, :,2], v_input[i,2]).int()

            if config.IS_GPU:
                nodes_without_enough = nodes_without_enough.cuda()

            not_satisfying_nodes = nodes_without_enough
            not_satisfying_nodes = not_satisfying_nodes + actions

            output_weight = self.calc_output_weight(s_node_numbers,encoder_output, decoder_output)
            output_weight = output_weight - not_satisfying_nodes*self.eta

            output_weights.append(output_weight)

            # 注意力机制，用于计算decoder_input，作为下一次的输入
            attention_weight = self.calc_attention_weight(s_node_numbers,encoder_output, decoder_output)
            decoder_input = torch.einsum('ij,ijk->ik', attention_weight, encoder_output)
            decoder_input = torch.unsqueeze(
                decoder_input,dim=1
            )

            decoder_outputs.append(torch.argmax(output_weight,dim=1))

            # 选择合适节点
            selected_actions = torch.zeros_like(actions)
            selected_actions = selected_actions.scatter_(
                1,torch.unsqueeze(
                    decoder_outputs[-1],
                    dim=1),1
            )
            actions += selected_actions

        # 随机抽取的解
        shuffled_node_mapping = np.array([list(output.cpu()) for output in decoder_outputs]).T
        # 原始排序的解
        original_node_mapping = np.zeros(shape=(batch_size, v_node_numbers), dtype=int)

        # 成功映射的解
        for i in range(batch_size):
            for j in range(v_node_numbers):
                original_node_mapping[i][j] = s_node_indexes[i][shuffled_node_mapping[i][j]]
        return original_node_mapping, shuffled_node_mapping,output_weights

def weights_init(m):
    if isinstance(m, torch.nn.LSTM):
        torch.nn.init.uniform_(m.weight_ih_l0.data, a=-0.08, b=0.08)
        torch.nn.init.uniform_(m.weight_hh_l0.data, a=-0.08, b=0.08)
        torch.nn.init.uniform_(m.weight_ih_l0_reverse.data, a=-0.08, b=0.08)
        torch.nn.init.uniform_(m.weight_hh_l0_reverse.data, a=-0.08, b=0.08)
    else:
        torch.nn.init.uniform_(m.weight.data, a=-0.08, b=0.08)
        torch.nn.init.uniform_(m.bias.data, a=-0.08, b=0.08)
