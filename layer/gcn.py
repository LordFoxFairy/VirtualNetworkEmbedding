# 图卷积
import torch, math
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F

class GraphConvolution(torch.nn.Module):
    """
    GCN Layer
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_featues = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

        # 初始化参数
        self.reset_parameters()

    def reset_parameters(self, ):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(0, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(0, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)  # torch.mm：矩阵相乘
        output = torch.spmm(adj, support)  # 稀疏矩阵相乘
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr(self, ):
        return "{}({}->{})".format(self.__class__.__name__, str(self.in_features), str(self.out_features))

class GCN(torch.nn.Module):

    def __init__(self, nfeat, nout):
        super(GCN, self).__init__()
        self.gcn = GraphConvolution(nfeat,nout)

    def forward(self, x, adj):
        return self.gcn(x,adj)