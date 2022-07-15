import torch
from torch import nn
from torch.autograd import Function
import torch.nn.functional as F
from torch_geometric.nn import SGConv, global_add_pool


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class rgnn(nn.Module):
    def __init__(self, num_in,  K, num_class, weight,num_hidden=30,dropout=0.7, domain_adaptation=0):
        super(rgnn,self).__init__()
        self.conv = SGConv(num_in,num_hidden,K)
        self.fc = torch.nn.Linear(num_hidden,num_class)
        self.domain_adaptation = domain_adaptation
        self.dropout = dropout
        self.weight = weight
        self.weight = nn.Parameter(weight, requires_grad=True)
        if self.domain_adaptation == 1:
            self.domain_classifier = nn.Linear(num_hidden, 2)

    def forward(self,len_y,x,index,batch,alpha=0):
        batch_size = len_y
        x, edge_index, edge_weight = x,index,self.weight
        iter = len_y - 1
        for _ in range(iter):
            edge_weight = torch.cat((edge_weight,self.weight),dim=0)
        x = self.conv(x,edge_index,edge_weight)
        x = nn.functional.softmax(x)
        domain_output = None
        if self.domain_adaptation == 1:
            reverse_x = ReverseLayerF.apply(x, alpha)
            domain_output = self.domain_classifier(reverse_x)

        x = global_add_pool(x, batch, size=batch_size)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc(x)
        x = nn.functional.softmax(x)

        return x, domain_output