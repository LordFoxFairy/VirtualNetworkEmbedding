import numpy as np
import torch
from util import config
import torch.nn.functional as F

class LNet(torch.nn.Module):
    def __init__(self,nembedding=128,nhidden=128,nheads=2,ninput=3,batch_size=10,dropout=0,eta=1e-6):
        super(LNet,self).__init__()

        self.embedding = torch.nn.Linear(ninput,nembedding,bias=False)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.eta = eta
        self.cross = torch.nn.CrossEntropyLoss(reduction='none')

        # Encoder
        self.Encoder = torch.nn.LSTM(input_size=nembedding, hidden_size=nhidden,bidirectional=True,batch_first=True,bias=False)

        # attention
        self.w1 = torch.nn.Linear(nheads*nhidden, nheads*nhidden)
        self.a1 = torch.nn.Linear(nheads*nhidden, nheads*nhidden)
        self.v1 = torch.nn.Linear(nheads*nhidden, 1)
        # self.attention1 = torch.nn.MultiheadAttention(nembedding,nheads,bias=False)
        # self.attention2 = torch.nn.MultiheadAttention(nembedding,nheads,bias=False)

        # Decoder
        self.Decoder = torch.nn.LSTM(input_size=nembedding*2, hidden_size=nhidden,bidirectional=True,batch_first=True)
        self.w2 = torch.nn.Linear(nheads*nhidden, nheads*nhidden)
        self.a2 = torch.nn.Linear(nheads*nhidden, nheads*nhidden)
        self.v2 = torch.nn.Linear(nheads*nhidden, 1)

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

    def norm(self,x):
        """
        避免数据数值过大，显示为nan
        :param x:
        :return:
        """
        return (x - x.min()) / (x.max() - x.min())


    def get_node_mapping(self,s_node_indexes,s_inputs,v_input):

        if config.IS_GPU:
            s_inputs = s_inputs.cuda()
            v_input = v_input.cuda()

        # batch_size、节点数目
        batch_size = s_node_indexes.size(0)
        s_node_numbers = s_node_indexes.size(1)
        v_node_numbers = v_input.size(0)

        # s_node_embedding:(batch,s_node_numbers,embedding)
        s_node_embedding = self.embedding(s_inputs)

        #encoder:(batch,s_node_numbers,nheads*nhidden)
        #hn:(batch,num_layers * nheads,  hidden_size)
        #cn:(batch,num_layers * nheads,  hidden_size)
        encoder,(hn,cn) = self.Encoder(self.norm(s_node_embedding),None)

        decoder_input = torch.zeros(encoder.size(0),1,encoder.size(2))
        decoder_state = (hn, cn)
        if config.IS_GPU:
            decoder_state = (hn.cuda(),cn.cuda())
        actions = torch.zeros(encoder.size(0),s_node_numbers)
        if config.IS_GPU:
            actions = actions.cuda()
            decoder_input = decoder_input.cuda()

        decoder_outputs = []
        output_weights = []

        for i in range(v_node_numbers):
            # Decoder
            decoder_output , decoder_state = self.Decoder(decoder_input,decoder_state)
            decoder_output = self.norm(decoder_output)
            decoder_output = self.dropout(decoder_output)

            if config.IS_GPU:
                decoder_output = decoder_output.cuda()
                decoder_state = list(decoder_state)
                decoder_state[1] = decoder_state[1].cuda()
                decoder_state[0] = decoder_state[0].cuda()
                decoder_state = tuple(decoder_state)


            encoder = self.dropout(encoder)

            # 筛选出满足的节点
            statisfying_nodes = torch.lt(s_inputs[:,:,0],v_input[i,0])
            if config.IS_GPU:
                statisfying_nodes = statisfying_nodes.cuda()

            # 不满足的
            cannot_satisfying_nodes = statisfying_nodes
            cannot_node = cannot_satisfying_nodes + actions

            output_weight = torch.squeeze(
                self.v1(torch.tanh(
                    self.w1(encoder) + self.a1(decoder_output.repeat(1,s_node_numbers,1))
                ))
            ) - cannot_node*self.eta
            output_weights.append(output_weight)

            # 输入 dropout后的e0 和 decoder的输入， 计算出attetion权重，并输出权重
            attention_weight = F.softmax(
                torch.squeeze(
                    self.v2(torch.tanh(
                        self.w2(encoder) + self.a2(decoder_output)
                    ))
                ),dim=1
            )

            # decoder_input:(batch,1,nembedding*nheads)
            decoder_input = torch.unsqueeze(
                torch.einsum('ij,ijk->ik', attention_weight, encoder),
                dim=1
            )

            decoder_outputs.append(torch.argmax(output_weight,dim=1))

            selected_actions = torch.zeros(encoder.size(0),s_node_numbers)

            if config.IS_GPU:
                selected_actions = selected_actions.cuda()

            selected_actions = selected_actions.scatter_(
                1,torch.unsqueeze(
                    decoder_outputs[-1],
                    dim=1),1
            )

            actions += selected_actions

        # 随机抽取的解
        shuffled_node_mapping = np.array([list(output) for output in decoder_outputs]).T
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
        # torch.nn.init.uniform_(m.bias_ih_l0.data, a=-0.08, b=0.08)
        # torch.nn.init.uniform_(m.bias_hh_l0.data, a=-0.08, b=0.08)
        torch.nn.init.uniform_(m.weight_ih_l0_reverse.data, a=-0.08, b=0.08)
        torch.nn.init.uniform_(m.weight_hh_l0_reverse.data, a=-0.08, b=0.08)
        # torch.nn.init.uniform_(m.bias_ih_l0_reverse.data, a=-0.08, b=0.08)
        # torch.nn.init.uniform_(m.bias_hh_l0_reverse.data, a=-0.08, b=0.08)
    else:
        try:
            torch.nn.init.uniform_(m.weight.data, a=-0.08, b=0.08)
            torch.nn.init.uniform_(m.bias.data, a=-0.08, b=0.08)
        except Exception:
            1 + 1


