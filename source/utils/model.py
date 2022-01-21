import torch
from torch._C import device
from torch.nn.modules.dropout import Dropout
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set#, GINConv, GINEConv,GCNConv,GATConv,SAGEConv
from .glayer_attr import GINConv, GCNConv,GATConv,SAGEConv
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.nn.inits import glorot, zeros

import torch.nn.functional as F
import torch_geometric.transforms as T
import torch.nn as nn
from .NodeFatureAggregator import WordAttention, WordBag, WordLSTM, WordLSTMBag
from ._TransformerAggregator import TransformerModel

import json

edge_type = json.load(open("tokens/edge_type.json"))

num_atom_type = 120 #including the extra mask tokens

num_edge_type = len(edge_type)



class GNN(torch.nn.Module):
    """
    

    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat

    Output:
        node representations

    """
    def __init__(self, num_layer, vocab_size, subtoken_emb_dim, em_lstm_hidden_size,  JK = "last", 
                                    drop_ratio = 0, gnn_type = "gin", embedding_way="bag", padding_index=0, bidirectional=False, node_rep_dim=256):
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.subtoken_emb_dim = subtoken_emb_dim
        self.node_rep_dim = node_rep_dim
        if embedding_way == "bag":
            print("Using Embedding Bag")
            self.embedding = WordBag(vocab_size, subtoken_emb_dim)
            self.node_attr_dim = subtoken_emb_dim
        elif embedding_way =="lstmbag":
            self.embedding = WordLSTMBag(vocab_size, subtoken_emb_dim, em_lstm_hidden_size,padding_index = padding_index, 
                rnn_model='LSTM', bidirectional=bidirectional)
            self.node_attr_dim = subtoken_emb_dim + em_lstm_hidden_size
        elif embedding_way == "lstm":
            print("Using Embedding LSTM")
            self.embedding = WordLSTM(vocab_size, subtoken_emb_dim, em_lstm_hidden_size,padding_index = padding_index, 
                                    rnn_model='LSTM',dropout_ratio=drop_ratio, bidirectional=bidirectional)
            self.node_attr_dim = em_lstm_hidden_size*2
        elif embedding_way == "gru":
            print("Using Embedding GRU")
            self.embedding = WordLSTM(vocab_size, subtoken_emb_dim, em_lstm_hidden_size,padding_index = padding_index, 
                        rnn_model='GRU', bidirectional=bidirectional)
            self.node_attr_dim = em_lstm_hidden_size
        elif embedding_way == "attention":
            self.embedding = WordAttention(vocab_size, subtoken_emb_dim, em_lstm_hidden_size, 1, em_lstm_hidden_size, 0.3, padding_index)
            self.node_attr_dim = em_lstm_hidden_size
        elif embedding_way == "selfattention":
            self.embedding = TransformerModel(
                ntoken=vocab_size, # vocabulary size
                ninp=subtoken_emb_dim, # embedding layer hidden dim
                nhid=em_lstm_hidden_size, # hidden dim
                nhead=4,  # number of self attention heads, NOTE: hidden_size % nhead == 0
                nlayers=4 # self attention layers
            )
            self.node_attr_dim = subtoken_emb_dim
        else:
            raise "Embedding Way Name Wrong should be Ba or Lstm"

        #torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        #torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        ###List of MLPs
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            if gnn_type == "gin":
                self.gnns.append(GINConv(self.node_attr_dim, self.node_attr_dim))
                self.node_rep_dim = self.node_attr_dim
            elif gnn_type == "gcn":
                self.gnns.append(GCNConv(self.node_attr_dim, self.node_attr_dim))
                self.node_rep_dim = self.node_attr_dim
            elif gnn_type == "gat":
                self.gnns.append(GATConv(self.node_attr_dim, self.node_attr_dim))
                self.node_rep_dim = self.node_attr_dim
            elif gnn_type == "graphsage":
                self.gnns.append(SAGEConv(self.node_attr_dim, self.node_attr_dim))
                self.node_rep_dim = self.node_attr_dim
        self.edge_embedding1 = torch.nn.Embedding(num_edge_type, self.node_attr_dim)       
        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        ###List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(self.node_rep_dim))

    #def forward(self, x, edge_index, edge_attr):
    def forward(self, *argv):
        if len(argv) == 4:
            x, edge_index, edge_attr, ins_length = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, ins_length = data.x, data.edge_index, data.edge_attr, data.ins_length
        else:
            raise ValueError("unmatched number of arguments.")

        x = self.embedding(x, ins_length) 

        h_list = [x]
        for layer in range(self.num_layer):
           # print(f"{h_list[layer].size()}, {edge_attr.size()}")
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "concat":
            if self.node_attr_dim != self.node_rep_dim:
                h_list=h_list[1:]
            node_representation = torch.cat(h_list, dim = 1) # [ number of nodes, node representation dimension * (numer layer + 1) ]
        elif self.JK == "last":
            node_representation = h_list[-1] #[ number of nodes, node representation dimension ]
        elif self.JK == "max":
            h_list = [h.unsqueeze(0) for h in h_list]
            if self.node_attr_dim != self.node_rep_dim:
                h_list=h_list[1:]
            node_representation = torch.max(torch.cat(h_list, dim = 0), dim = 0) # [ number of nodes, node representation dimension ]
        elif self.JK == "sum":
            h_list = [h.unsqueeze(0) for h in h_list]
            if self.node_attr_dim != self.node_rep_dim:
                h_list=h_list[1:]
            node_representation = torch.sum(torch.cat(h_list, dim = 0), dim = 0) # [ number of nodes, node representation dimension ]

        return node_representation, x





class GNN_encoder(torch.nn.Module):
    """
    Extension of GIN to incorporate edge information by concatenation.

    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        num_tasks (int): number of tasks in multi-task learning scenario
        drop_ratio (float): dropout rate
        JK (str): last, concat, max or sum.
        graph_pooling (str): sum, mean, max, attention, set2set
        gnn_type: gin, gcn, graphsage, gat
        
    See https://arxiv.org/abs/1810.00826
    JK-net: https://arxiv.org/abs/1806.03536
    """
    def __init__(self, num_layer,vocab_size, sub_token_emb_dim, lstm_emb_dim, num_class=1, JK = "last", drop_ratio = 0, 
            graph_pooling = "mean", gnn_type = "gin", subword_emb="bag", bidrection=False, task="javasmall",repWay="append"):
        super(GNN_encoder, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.task = task
        self.repWay=repWay
        #self.emb_dim = emb_dim
        self.num_class = num_class
        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        
        self.gnn = GNN(num_layer, vocab_size, sub_token_emb_dim, lstm_emb_dim, JK, drop_ratio, gnn_type = gnn_type,
                     embedding_way=subword_emb, bidirectional=bidrection)
        
        #Different kind of graph pooling
        print("Configure === ")
        print(f"JK = {self.JK}")
        print(f"Target Class = {self.num_class}")
        print(f"Drop Ratio = {self.drop_ratio}")
        print(f"Graph_pooling = {graph_pooling}")
        self.dropout = Dropout()
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            if self.JK == "concat":
                if self.gnn.node_attr_dim != self.gnn.node_rep_dim:
                    self.pool = GlobalAttention(gate_nn = torch.nn.Linear( self.num_layer  * self.gnn.node_rep_dim, 1))
                else:
                    self.pool = GlobalAttention(gate_nn = torch.nn.Linear((self.num_layer + 1) * self.gnn.node_rep_dim, 1))
            else:
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear(self.gnn.node_rep_dim, 1))
        elif graph_pooling[-1] == "set2set":
            set2set_iter = int(graph_pooling[0])
            if self.JK == "concat":
                if self.gnn.node_attr_dim != self.gnn.node_rep_dim:
                    self.pool = Set2Set(self.num_layer  * self.gnn.node_rep_dim, set2set_iter)
                else:
                    self.pool = Set2Set( (self.num_layer+1)  * self.gnn.node_rep_dim, set2set_iter)
            else:
                self.pool = Set2Set(self.gnn.node_rep_dim, set2set_iter)
        else:
            raise ValueError("Invalid graph pooling type.")

        #For graph-level binary classification
        if graph_pooling[-1] == "set2set":
            self.mult = 2
        else:
            self.mult = 1
        
        classifier_input_dim = None
        if self.JK == "concat":
            if self.gnn.node_attr_dim != self.gnn.node_rep_dim:
                classifier_input_dim = self.mult * self.num_layer  * self.gnn.node_rep_dim * 2
                #self.prediction_linear = torch.nn.Linear(classifier_input_dim, self.num_class)
            else:
                classifier_input_dim = self.mult * ( self.num_layer +1)  * self.gnn.node_rep_dim * 2
                #self.prediction_linear = torch.nn.Linear(classifier_input_dim, self.num_class)
        else:
            classifier_input_dim = self.gnn.node_rep_dim
            #self.prediction_linear = torch.nn.Linear(classifier_input_dim, self.num_class)
        print(f"Node Rep dim {self.gnn.node_rep_dim}")
        print(f"Node Attr dim {self.gnn.node_attr_dim}")
        

        if self.repWay == "seq":
               # print("Only use seq")
                classifier_input_dim =  self.gnn.node_rep_dim
        elif self.repWay == "append":  
                classifier_input_dim = self.gnn.node_rep_dim * 2 
        elif self.repWay == "graph":
                classifier_input_dim =  self.gnn.node_rep_dim
        elif self.repWay == "alpha":
                #self.alpha = nn.Parameter(torch.tensor(0.), requires_grad=True)
                self.weight_inp1 = nn.Parameter(torch.ones(1, classifier_input_dim), requires_grad=True)
                self.weight_inp2 = nn.Parameter(torch.ones(1, classifier_input_dim), requires_grad=True)
                torch.nn.init.xavier_uniform_(self.weight_inp1.data)
                torch.nn.init.xavier_uniform_(self.weight_inp2.data)
               # print(self.weight_inp1.data)
                classifier_input_dim =  self.gnn.node_rep_dim
        else:
                assert False, f"Invalid repway {self.repWay}"

        if self.task == "codeclone":
            self.prediction_linear = torch.nn.Linear(classifier_input_dim*2, self.num_class)
        else:
            self.prediction_linear = torch.nn.Linear(classifier_input_dim, self.num_class)

        print(f"Rep way {self.repWay}")
       # assert self.repWay == "alpha"
        
        if self.task in ["GAE", "VGAE"]:
            if gnn_type == "gin":
                self.conv_mu = GINConv(self.gnn.node_attr_dim, self.gnn.node_attr_dim)
                self.conv_logstd = GINConv(self.gnn.node_attr_dim, self.gnn.node_attr_dim) 
                self.node_rep_dim = self.gnn.node_attr_dim
            elif gnn_type == "gcn":
                self.conv_mu = GCNConv(self.gnn.node_attr_dim, self.gnn.node_attr_dim)
                self.conv_logstd = GCNConv(self.gnn.node_attr_dim, self.gnn.node_attr_dim) 
                self.node_rep_dim = self.gnn.node_attr_dim
            elif gnn_type == "gat":
                self.conv_mu = GATConv(self.gnn.node_attr_dim, self.gnn.node_attr_dim)
                self.conv_logstd = GATConv(self.gnn.node_attr_dim, self.gnn.node_attr_dim)
                self.node_rep_dim = self.gnn.node_attr_dim
            elif gnn_type == "graphsage":
                self.conv_mu = SAGEConv(self.gnn.node_attr_dim, self.gnn.node_attr_dim)
                self.conv_logstd = SAGEConv(self.gnn.node_attr_dim, self.gnn.node_attr_dim)
                self.node_rep_dim = self.gnn.node_attr_dim


   

    def forward(self, *argv):
        if len(argv) == 5:
            x, edge_index, edge_attr, batch, ins_length = argv[0], argv[1], argv[2], argv[3], argv[4]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch, ins_length = data.x, data.edge_index, data.edge_attr, data.batch, data.ins_length
        else:
            raise ValueError("unmatched number of arguments.")
        

        node_representation, x = self.gnn(x, edge_index, edge_attr, ins_length)
        if self.task == "context":
            return node_representation
        if self.task in [ "javasmall", "java250"]:
            seq_x = global_add_pool(x, batch)
            if self.repWay == "seq":
               # print("Only use seq")
                return self.prediction_linear( seq_x )
            elif self.repWay == "append":    
                return self.prediction_linear( torch.cat( ( seq_x, self.pool(node_representation, batch) ), dim=1) )
            elif self.repWay == "graph":
                return self.prediction_linear(self.pool(node_representation, batch) )
            elif self.repWay == "alpha":
                #print(self.alpha)
                rep = seq_x * self.weight_inp1 + self.pool(node_representation, batch) * self.weight_inp2
                return  self.prediction_linear( rep )
            else:
                assert False, f"Invalid repway {self.repWay}"

        if self.task in ["node_class"]:
            return self.prediction_linear( torch.cat( ( x, node_representation ), dim=1) )
        if self.task == "VGAE":
            return self.conv_mu(node_representation, edge_index), self.conv_logstd(node_representation, edge_index)
        if self.task == "GAE":
            return self.conv_mu(node_representation, edge_index)
        
        if self.task == "codeclone":
            seq_x = global_add_pool(x, batch)
            if self.repWay == "seq":
               # print("Only use seq")
                return seq_x 
            elif self.repWay == "append":    
                return  torch.cat( ( seq_x, self.pool(node_representation, batch) ), dim=1) 
            elif self.repWay == "graph":
                return self.pool(node_representation, batch) 
            elif self.repWay == "alpha":
                #print(self.alpha)
                rep = seq_x * self.weight_inp1 + self.pool(node_representation, batch) * self.weight_inp2
                return   rep 
            else:
                assert False, f"Invalid repway {self.repWay}"

    def from_pretrained(self, model_file, device):
        if "gnn" in model_file:
            # layers = torch.load(model_file, map_location="cpu")
            # for k in layers:
            #     print(k)
            self.gnn.load_state_dict(torch.load(model_file, map_location="cpu"))
            print("load gnn")
        else:
            data = torch.load(model_file, map_location="cpu")
            if "model_substruct_state_dict" in data:
                gnn_weights = data["model_substruct_state_dict"]
            else:
                gnn_weights = data["model_state_dict"]
            # for k in gnn_weights:
            #      print(k)
            #gnn_weights = torch.load(model_file,  map_location="cpu")
            if "prediction_linear.weight" in gnn_weights:
                del gnn_weights["prediction_linear.weight"]
                del gnn_weights["prediction_linear.bias"]
            for d in ["graph_pred_linear", "link_prediction"]:
                     if  f"{d}.weight" in  gnn_weights:
                         del gnn_weights[f"{d}.weight"]
                     if  f"{d}.bias" in  gnn_weights:
                         del gnn_weights[f"{d}.bias"]

            self.load_state_dict(gnn_weights, strict=False)
            
    def loadWholeModel(self, model_file, device, maps={}):
        gnn_weights = torch.load(model_file,  map_location="cpu")
        self.load_state_dict(gnn_weights)
    def getVector(self, *argv):
        if len(argv) == 5:
            x, edge_index, edge_attr, batch, ins_length = argv[0], argv[1], argv[2], argv[3], argv[4]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch, ins_length = data.x, data.edge_index, data.edge_attr, data.batch, data.ins_length
        else:
            raise ValueError("unmatched number of arguments.")
      
        node_representation, x = self.gnn(x, edge_index, edge_attr, ins_length)
        seq_x = global_add_pool(x, batch) #torch.cat( ( seq_x, self.pool(node_representation, batch) ), dim=1)
        return  torch.cat( ( seq_x, self.pool(node_representation, batch) ), dim=1), self.pool(node_representation, batch)  , seq_x

    def prediction(self, x_s, x_t):
        return self.prediction_linear( torch.cat( (x_s, x_t) , dim=1) )
    
    def compute(self, batch):
        return self.forward(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.ins_length)

if __name__ == "__main__":
    pass

