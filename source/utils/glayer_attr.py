import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax
import torch.nn.functional as F
from torch.nn import Parameter, Linear
from torch import Tensor
from torch_scatter import scatter_add
from torch_geometric.nn.inits import glorot, zeros
import torch_geometric.transforms as T
import json
from torch_sparse import SparseTensor, set_diag
import os
print(os.getcwd())
edge_type = json.load(open("tokens/edge_type.json"))

num_edge_type = len(edge_type)


class GINConv(MessagePassing):
    """
    Extension of GIN aggregation to incorporate edge information by concatenation.
    Args:
        emb_dim (int): dimensionality of embeddings for nodes and edges.
        embed_input (bool): whether to embed input or not. 
        
    See https://arxiv.org/abs/1810.00826
    """
    def __init__(self, input_channel, output_channel, aggr = "add"):
        super(GINConv, self).__init__()
        #multi-layer perceptron
        self.mlp = torch.nn.Sequential(torch.nn.Linear(input_channel, 2*input_channel), torch.nn.ReLU(), torch.nn.Linear(2*input_channel, output_channel))
        self.edge_embedding1 = torch.nn.Embedding(num_edge_type, output_channel)
        self.edge_embedding2 = torch.nn.Embedding(2, output_channel)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr=None):
        #add self loops in the edge space
        if edge_attr!= None:
            edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
            edge_index,_ = add_self_loops(edge_index, num_nodes = x.size(0))

            #add features corresponding to self-loop edges.
            self_loop_attr = torch.zeros(x.size(0), 2)
            self_loop_attr[:,0] = edge_type["Self-Loop"]
            self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
            edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)
        
            edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])
    
            return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)
        else:
            edge_index, _ = remove_self_loops(edge_index, None)
            edge_index,_ = add_self_loops(edge_index, num_nodes = x.size(0))
            return self.propagate(edge_index, x=x, edge_attr=None)

    def message(self, x_j, edge_attr):
        if edge_attr!=None:
            return x_j + edge_attr
        else:
            return x_j

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class GCNConv(MessagePassing):

    def __init__(self, input_channel, output_channel, aggr = "add"):
        super(GCNConv, self).__init__(aggr)

        self.emb_dim = output_channel
        self.linear = torch.nn.Linear(input_channel, output_channel)
        self.edge_embedding1 = torch.nn.Embedding(num_edge_type, output_channel)
        self.edge_embedding2 = torch.nn.Embedding(2, output_channel)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.aggr = aggr

    def norm(self, edge_index, num_nodes, dtype):
         ### assuming that self-loops have been already added in edge_index
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


    def forward(self, x, edge_index, edge_attr=None):
        if edge_attr!=None:
            #add self loops in the edge space
            edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
            edge_index,_ = add_self_loops(edge_index, num_nodes = x.size(0))

            #add features corresponding to self-loop edges.
            self_loop_attr = torch.zeros(x.size(0), 2)
            self_loop_attr[:,0] = edge_type["Self-Loop"] 
            self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
            edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)

            edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])
            norm = self.norm(edge_index, x.size(0), x.dtype)
            x = self.linear(x)
            return self.propagate(edge_index, x=x, edge_attr=edge_embeddings, norm = norm)
        else:
            edge_index, _ = remove_self_loops(edge_index)
            edge_index,_ = add_self_loops(edge_index, num_nodes = x.size(0))
            norm = self.norm(edge_index, x.size(0), x.dtype)
            x = self.linear(x)
            return self.propagate(edge_index, x=x, edge_attr=None, norm = norm)

    def message(self, x_j, edge_attr, norm):
        if edge_attr != None:
            return norm.view(-1, 1) * (x_j + edge_attr)
        else:
            return norm.view(-1, 1) * x_j 





class SAGEConv(MessagePassing):
    def __init__(self, input_channel, output_channel, aggr = "mean"):
        super(SAGEConv, self).__init__(aggr)

        self.emb_dim = input_channel
        self.linear = torch.nn.Linear(input_channel, output_channel)
        self.edge_embedding1 = torch.nn.Embedding(num_edge_type, output_channel)
        self.edge_embedding2 = torch.nn.Embedding(2, output_channel)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr=None):
        if edge_attr is not None:
            edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])
        else:
            edge_embeddings = None

        x = self.linear(x)

        return self.propagate( edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr if edge_attr is not None else x_j

    def update(self, aggr_out):
        return F.normalize(aggr_out, p = 2, dim = -1)


from typing import Union, Tuple, Optional
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType,
                                    OptTensor)
class GATConv(MessagePassing):

    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, heads: int = 1, concat: bool = True,
                 negative_slope: float = 0.2, dropout: float = 0.0,
                 add_self_loops: bool = True, bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(GATConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.edge_embedding1 = torch.nn.Embedding(num_edge_type, heads * out_channels)
        self.edge_embedding2 = torch.nn.Embedding(2, heads * out_channels)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        if isinstance(in_channels, int):
            self.lin_l = Linear(in_channels, heads * out_channels, bias=False)
            self.lin_r = self.lin_l
        else:
            self.lin_l = Linear(in_channels[0], heads * out_channels, False)
            self.lin_r = Linear(in_channels[1], heads * out_channels, False)

        self.att_l = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_r = Parameter(torch.Tensor(1, heads, out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self._alpha = None

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin_l.weight)
        glorot(self.lin_r.weight)
        glorot(self.att_l)
        glorot(self.att_r)
        zeros(self.bias)

    def reset_attention(self):
        glorot(self.att_l)
        glorot(self.att_r)
        

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,edge_attr:Tensor=None,
                size: Size = None, return_attention_weights=None):
        r"""
        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        H, C = self.heads, self.out_channels

        x_l: OptTensor = None
        x_r: OptTensor = None
        alpha_l: OptTensor = None
        alpha_r: OptTensor = None
      

        if isinstance(x, Tensor):
            assert x.dim() == 2, 'Static graphs not supported in `GATConv`.'
            x_l = x_r = self.lin_l(x).view(-1, H, C)
            alpha_l = (x_l * self.att_l).sum(dim=-1)
            alpha_r = (x_r * self.att_r).sum(dim=-1)
        else:
            x_l, x_r = x[0], x[1]
            assert x[0].dim() == 2, 'Static graphs not supported in `GATConv`.'
            x_l = self.lin_l(x_l).view(-1, H, C)
            alpha_l = (x_l * self.att_l).sum(dim=-1)
            if x_r is not None:
                x_r = self.lin_r(x_r).view(-1, H, C)
                alpha_r = (x_r * self.att_r).sum(dim=-1)

        assert x_l is not None
        assert alpha_l is not None

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_l.size(0)
                if x_r is not None:
                    num_nodes = min(num_nodes, x_r.size(0))
                if size is not None:
                    num_nodes = min(size[0], size[1])
                # print(edge_index.shape)
                # print(edge_attr.shape)
                edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
                edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)
            #add features corresponding to self-loop edges.
            if edge_attr!=None:
                self_loop_attr = torch.zeros(x.size(0), 2)
                self_loop_attr[:,0] = edge_type["Self-Loop"] 
                self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
                edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)
        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1]) if edge_attr != None else None
        
        # print(edge_embeddings.shape)
        # print(x_l.shape)
        # print(x_r.shape)
        # print(edge_index.shape)
        # propagate_type: (x: OptPairTensor, alpha: OptPairTensor)
        out = self.propagate(edge_index, x=(x_l, x_r),
                             alpha=(alpha_l, alpha_r), size=size, edge_attr=edge_embeddings)

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out


    def message(self, x_j: Tensor, alpha_j: Tensor, alpha_i: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int], edge_attr: Tensor) -> Tensor:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        #print(x_j.shape)
        if edge_attr!=None:
            H, C = self.heads, self.out_channels
            edge_attr = edge_attr.view(-1, H, C)
            return (x_j+edge_attr) * alpha.unsqueeze(-1)
        else:
            return x_j * alpha.unsqueeze(-1)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)