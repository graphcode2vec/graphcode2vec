import argparse
import json
from random import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from tqdm import tqdm
import numpy as np
from utils.model import GNN_encoder
from utils.AverageMeter import AverageMeter
from utils.dataset import UnsupervisedDataset
from utils.extract_substructure import ExtractSubstructureContextPair, DataLoaderSubstructContext
from utils.tools import  TokenIns, performance
from utils.dataset import UnsupervisedDataset
from torch_geometric.transforms.remove_isolated_nodes import RemoveIsolatedNodes
import random
from decouple import config
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from utils.tools import TokenIns,inverse_eage
from torchvision import transforms
DEBUG=config('DEBUG', default=False, cast=bool)
step_eval=1 if DEBUG else 200

def pool_func(x, batch, mode = "sum"):
    if mode == "sum":
        return global_add_pool(x, batch)
    elif mode == "mean":
        return global_mean_pool(x, batch)
    elif mode == "max":
        return global_max_pool(x, batch)

def cycle_index(num, shift):
    arr = torch.arange(num) + shift
    arr[-shift:] = torch.arange(shift)
    return arr

criterion = nn.BCEWithLogitsLoss()

    
def train(args, model_context, model_substruct, device, optimizer_substruct, optimizer_context, loader):
    model_context.train()
    model_substruct.train()
    trainloss = AverageMeter()
    balanced_loss_accum = 0
    acc_accum = 0

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        if batch is None:
            continue
        batch = batch.to(device)
       # print(batch.center_substruct_idx.dtype)
        substruct_rep = model_substruct(batch.x_substruct, batch.edge_index_substruct, batch.edge_attr_substruct, \
            batch.batch, batch.ins_length_substruct)[ batch.center_substruct_idx ]
       # print(substruct_rep.dtype)
        ### creating context representations
        # print(batch.center_substruct_idx)
        # print( batch.overlap_context_substruct_idx )
        overlapped_node_rep = model_context(batch.x_context, batch.edge_index_context, batch.edge_attr_context, \
             batch.batch, batch.ins_length_context)[batch.overlap_context_substruct_idx]
       # print(substruct_rep.shape  )
       # print(overlapped_node_rep.shape  )
        #Contexts are represented by 
        if args.mode == "cbow":
            # positive context representation
            context_rep = pool_func(overlapped_node_rep, batch.batch_overlapped_context, mode = args.context_pooling)
            #print(context_rep.shape  )
            # negative contexts are obtained by shifting the indicies of context embeddings
            neg_context_rep = torch.cat([context_rep[cycle_index(len(context_rep), i+1)] for i in range(args.neg_samples)], dim = 0)
            
            pred_pos = torch.sum(substruct_rep * context_rep, dim = 1)
            pred_neg = torch.sum(substruct_rep.repeat((args.neg_samples, 1))*neg_context_rep, dim = 1)

        elif args.mode == "skipgram":

            expanded_substruct_rep = torch.cat([substruct_rep[i].repeat((batch.overlapped_context_size[i],1)) for i in range(len(substruct_rep))], dim = 0)
            pred_pos = torch.sum(expanded_substruct_rep * overlapped_node_rep, dim = 1)

            #shift indices of substructures to create negative examples
            shifted_expanded_substruct_rep = []
            for i in range(args.neg_samples):
                shifted_substruct_rep = substruct_rep[cycle_index(len(substruct_rep), i+1)]
                shifted_expanded_substruct_rep.append(torch.cat([shifted_substruct_rep[i].repeat((batch.overlapped_context_size[i],1)) for i in range(len(shifted_substruct_rep))], dim = 0))

            shifted_expanded_substruct_rep = torch.cat(shifted_expanded_substruct_rep, dim = 0)
            pred_neg = torch.sum(shifted_expanded_substruct_rep * overlapped_node_rep.repeat((args.neg_samples, 1)), dim = 1)

        else:
            raise ValueError("Invalid mode!")

        loss_pos = criterion(pred_pos.double(), torch.ones(len(pred_pos)).to(pred_pos.device).double())
        loss_neg = criterion(pred_neg.double(), torch.zeros(len(pred_neg)).to(pred_neg.device).double())

        
        optimizer_substruct.zero_grad()
        optimizer_context.zero_grad()

        loss = loss_pos + args.neg_samples*loss_neg
        loss.backward()
        #To write: optimizer
        optimizer_substruct.step()
        optimizer_context.step()

        balanced_loss_accum += float(loss_pos.detach().cpu().item() + loss_neg.detach().cpu().item())
        acc_accum += 0.5* (float(torch.sum(pred_pos > 0).detach().cpu().item())/len(pred_pos) + float(torch.sum(pred_neg < 0).detach().cpu().item())/len(pred_neg))

    return balanced_loss_accum/step, acc_accum/step

def eval(args, model_context, model_substruct, device, loader):
    trainloss = AverageMeter()
    balanced_loss_accum = 0
    acc_accum = 0
    with torch.no_grad():
        for step, batch in enumerate(tqdm(loader, desc="Iteration")):
            batch = batch.to(device)
            substruct_rep = model_substruct(batch.x_substruct, batch.edge_index_substruct, batch.edge_attr_substruct, \
                batch.batch, batch.ins_length_substruct)[ batch.center_substruct_idx ]
            overlapped_node_rep = model_context(batch.x_context, batch.edge_index_context, batch.edge_attr_context, \
                batch.batch, batch.ins_length_context)[batch.overlap_context_substruct_idx]
            if args.mode == "cbow":
                # positive context representation
                context_rep = pool_func(overlapped_node_rep, batch.batch_overlapped_context, mode = args.context_pooling)
                #print(context_rep.shape  )
                # negative contexts are obtained by shifting the indicies of context embeddings
                neg_context_rep = torch.cat([context_rep[cycle_index(len(context_rep), i+1)] for i in range(args.neg_samples)], dim = 0)
                
                pred_pos = torch.sum(substruct_rep * context_rep, dim = 1)
                pred_neg = torch.sum(substruct_rep.repeat((args.neg_samples, 1))*neg_context_rep, dim = 1)

            elif args.mode == "skipgram":

                expanded_substruct_rep = torch.cat([substruct_rep[i].repeat((batch.overlapped_context_size[i],1)) for i in range(len(substruct_rep))], dim = 0)
                pred_pos = torch.sum(expanded_substruct_rep * overlapped_node_rep, dim = 1)

                #shift indices of substructures to create negative examples
                shifted_expanded_substruct_rep = []
                for i in range(args.neg_samples):
                    shifted_substruct_rep = substruct_rep[cycle_index(len(substruct_rep), i+1)]
                    shifted_expanded_substruct_rep.append(torch.cat([shifted_substruct_rep[i].repeat((batch.overlapped_context_size[i],1)) for i in range(len(shifted_substruct_rep))], dim = 0))

                shifted_expanded_substruct_rep = torch.cat(shifted_expanded_substruct_rep, dim = 0)
                pred_neg = torch.sum(shifted_expanded_substruct_rep * overlapped_node_rep.repeat((args.neg_samples, 1)), dim = 1)

            else:
                raise ValueError("Invalid mode!")

            loss_pos = criterion(pred_pos.double(), torch.ones(len(pred_pos)).to(pred_pos.device).double())
            loss_neg = criterion(pred_neg.double(), torch.zeros(len(pred_neg)).to(pred_neg.device).double())

        
            balanced_loss_accum += float(loss_pos.detach().cpu().item() + loss_neg.detach().cpu().item())
            acc_accum += 0.5* (float(torch.sum(pred_pos > 0).detach().cpu().item())/len(pred_pos) + float(torch.sum(pred_neg < 0).detach().cpu().item())/len(pred_neg))

    return balanced_loss_accum/step, acc_accum/step

def create_model(args, num_class=1):
    l1 = args.num_layer - 1
    l2 = l1 + args.csize
    #set up model
    tokenizer_word2vec = TokenIns( 
        word2vec_file=os.path.join(args.sub_token_path, args.emb_file),
        tokenizer_file=os.path.join(args.sub_token_path, "fun.model")
    )
    embeddings, word_emb_dim, vocab_size = tokenizer_word2vec.load_word2vec_embeddings()
    model_substruct = GNN_encoder(args.num_layer,vocab_size, word_emb_dim, args.lstm_emb_dim, num_class=num_class, JK = args.JK, drop_ratio = args.dropout_ratio, 
                        graph_pooling = args.graph_pooling, gnn_type = args.gnn_type, subword_emb=args.subword_embedding, bidrection=args.bidirection, task=args.task)
    model_substruct.gnn.embedding.init_embeddings(embeddings)
    model_substruct.gnn.embedding.fine_tune_embeddings(False)

    
    model_context = GNN_encoder(int(l2 - l1), vocab_size, word_emb_dim, args.lstm_emb_dim, num_class=num_class, JK = args.JK, drop_ratio = args.dropout_ratio, 
                        graph_pooling = args.graph_pooling, gnn_type = args.gnn_type, subword_emb=args.subword_embedding, bidrection=args.bidirection, task=args.task)
    model_context.gnn.embedding.init_embeddings(embeddings)
    model_context.gnn.embedding.fine_tune_embeddings(False)
    
    return model_substruct, model_context

import gc

def train_mode(args):
    os.makedirs( args.saved_model_path, exist_ok=True)
    if args.graph_pooling == "set2set":
        args.graph_pooling = [2, args.graph_pooling]

    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    #set up dataset
    # using jars dataset to pretrain
    l1 = args.num_layer - 1
    l2 = l1 + args.csize

    if DEBUG:
        dataset = UnsupervisedDataset( args.dataset_path , dataname=args.dataset, load_index=0, check_all=False,  \
            transform = transforms.Compose([RemoveIsolatedNodes(), ExtractSubstructureContextPair(args.num_layer, l1, l2)]) )
        
        check_all = False
        data_partitions = dataset.get_partitions()
    else:
        dataset = UnsupervisedDataset( args.dataset_path , dataname=args.dataset, load_index=0, check_all=args.check_all, \
            transform = transforms.Compose([RemoveIsolatedNodes(), ExtractSubstructureContextPair(args.num_layer, l1, l2)])  )
        check_all = args.check_all
        data_partitions = dataset.get_partitions()

    num_class=1
    if args.task == "node_class":
        num_class = 20
    #set up model
    tokenizer_word2vec = TokenIns( 
        word2vec_file=os.path.join(args.sub_token_path, args.emb_file),
        tokenizer_file=os.path.join(args.sub_token_path, "fun.model")
    )
    embeddings, word_emb_dim, vocab_size = tokenizer_word2vec.load_word2vec_embeddings()
    model_substruct = GNN_encoder(args.num_layer,vocab_size, word_emb_dim, args.lstm_emb_dim, num_class=num_class, JK = args.JK, drop_ratio = args.dropout_ratio, 
                        graph_pooling = args.graph_pooling, gnn_type = args.gnn_type, subword_emb=args.subword_embedding, bidrection=args.bidirection, task=args.task)
    model_substruct.gnn.embedding.init_embeddings(embeddings)
    model_substruct.gnn.embedding.fine_tune_embeddings(False)

    
    model_context = GNN_encoder(int(l2 - l1), vocab_size, word_emb_dim, args.lstm_emb_dim, num_class=num_class, JK = args.JK, drop_ratio = args.dropout_ratio, 
                        graph_pooling = args.graph_pooling, gnn_type = args.gnn_type, subword_emb=args.subword_embedding, bidrection=args.bidirection, task=args.task)
    model_context.gnn.embedding.init_embeddings(embeddings)
    model_context.gnn.embedding.fine_tune_embeddings(False)

    #set up dataset and transform function.
    loader = DataLoaderSubstructContext(dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
  
    pytorch_total_params = sum(p.numel() for p in model_context.parameters() if p.requires_grad)
    print(f"\nTotal Number of Parameters of model_context, {pytorch_total_params}")
    pytorch_total_params = sum(p.numel() for p in model_substruct.parameters() if p.requires_grad)
    print(f"\nTotal Number of Parameters of model_substruct, {pytorch_total_params}")
    model_substruct.to(device)
    model_context.to(device)

    #set up optimizer
    #set up optimizer for the two GNNs
    optimizer_substruct = optim.Adam(model_substruct.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer_context = optim.Adam(model_context.parameters(), lr=args.lr, weight_decay=args.decay)
    iters = [ i for i in range( args.epochs  )]
    if os.path.isfile(os.path.join( args.saved_model_path , f"partition_index.json") ):
        split_index = json.load(open(os.path.join( args.saved_model_path , f"partition_index.json"), "r"))
    else:
        split_index = {}
    for epoch in iters:
        for p in data_partitions:
            dataset = UnsupervisedDataset( args.dataset_path , dataname=args.dataset, load_index=p, check_all=check_all, \
                         transform = transforms.Compose([ExtractSubstructureContextPair(args.num_layer, l1, l2)])  )
               
            #dataset.shuffle()
            if p not in split_index:
                index = [i for i in range(len(dataset ))]
                random.shuffle(index)
                test_size = int(0.2*len(dataset))
                val_size = int(0.2* (len(dataset) -test_size ))
                train_size = len(dataset) - test_size -val_size
                train_index = index[: train_size]
                test_index = index[train_size : train_size+test_size]
                val_index = index[ train_size+test_size: ]
                split_index[p] = { "train":train_index, "val":val_index, "test":test_index }
                json.dump(split_index, open(os.path.join( args.saved_model_path , f"partition_index.json"), "w"))
            if DEBUG:
                loader = DataLoaderSubstructContext(dataset[split_index[p]["train"]][:300], batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
                loader_test = DataLoaderSubstructContext(dataset[split_index[p]["test"]][:300], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
            else:
                loader = DataLoaderSubstructContext(dataset[split_index[p]["train"]], batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
                loader_test = DataLoaderSubstructContext(dataset[split_index[p]["test"]], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
           
           
           
            print("==== epoch {}, data Partition {}  ".format( str(epoch) , p))
            trainloss, trainacc = train(args, model_context, model_substruct, device, optimizer_substruct, optimizer_context, loader) 
            model_context.eval()
            model_substruct.eval()
            evalloss, evalacc = eval(args, model_context, model_substruct, device, loader_test)
            torch.save(model_context.gnn.state_dict(), os.path.join( args.saved_model_path , f"model_contextgnn_{epoch}_{p}_roc{evalacc:.3f}.pth"))
            torch.save(model_substruct.gnn.state_dict(), os.path.join( args.saved_model_path , f"model_structuregnn_{epoch}_{p}_roc{evalacc:.3f}.pth"))
            #torch.save(model.state_dict(), os.path.join( args.saved_model_path , f"model.pth"))
            checkpointfile = os.path.join( args.saved_model_path , f"model_{epoch}.pth")
            torch.save({
            'epoch': epoch,
            'model_context_state_dict': model_context.state_dict(),
            'model_substruct_state_dict': model_substruct.state_dict(),
            'optimizer_substruct_state_dict': optimizer_substruct.state_dict(),
            'optimizer_context_state_dict': optimizer_context.state_dict(),
            }, checkpointfile)
            print(f"epoch {epoch}, data Partition {p} Train loss {trainloss}, Train acc {trainacc}, Test loss {evalloss}, Test acc {evalacc}")
            del dataset
            del loader
            del model_context
            del model_substruct
            del optimizer_substruct
            del optimizer_context
            gc.collect()
            torch.cuda.empty_cache()
            model_substruct, model_context = create_model(args, num_class)
            checkpoint = torch.load( checkpointfile )
            model_context.load_state_dict(checkpoint["model_context_state_dict"])
            model_substruct.load_state_dict(checkpoint["model_substruct_state_dict"])
            model_context.to(device)
            model_substruct.to(device)
            optimizer_substruct = optim.Adam(model_substruct.parameters(), lr=args.lr, weight_decay=args.decay)
            optimizer_context = optim.Adam(model_context.parameters(), lr=args.lr, weight_decay=args.decay)
            optimizer_substruct.load_state_dict(checkpoint['optimizer_substruct_state_dict'])
            optimizer_context.load_state_dict(checkpoint['optimizer_context_state_dict'])
            

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--recover', type=int, default=-1,
                        help='recover model')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 3).')
    parser.add_argument('--dropout_ratio', type=float, default=0.2,
                        help='dropout ratio (default: 0.2)')
    parser.add_argument('--evaluation', dest='evaluation', action='store_true', default=False) 
    parser.add_argument('--test', type=str, dest='test', default="") 
    

    parser.add_argument('--subword_embedding', type=str, default="lstm",
                        help='embed  (bag, lstmbag, gru, lstm, attention, selfattention)')
    parser.add_argument('--bidirection', dest='bidirection', action='store_true', default=True) 

    parser.add_argument('--lstm_emb_dim', type=int, default=150,
                        help='lstm embedding dimensions (default: 300)')
    parser.add_argument('--loaded_index', type=int, default=0,
                        help='lstm embedding dimensions (default: 300)')
   

    parser.add_argument('--graph_pooling', type=str, default="attention",
                        help='graph level pooling (sum, mean, max, set2set, attention)')
    
    parser.add_argument('--JK', type=str, default="sum",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--gnn_type', type=str, default="graphsage")

    parser.add_argument('--dataset', type=str, default = 'DV_PDG', help='root directory of dataset. For now, only classification.')
    parser.add_argument('--dataset_path', type=str, default = 'dataset/pretraining', help='root directory of dataset. For now, only classification.')
    parser.add_argument('--input_model_file', type=str, default = '', help='filename to read the model (if there is any)')
    parser.add_argument('--target_token_path', type=str, default = 'dataset/downstream/java-small/target_word_to_index.json', help='Target Vocab')
    parser.add_argument('--saved_model_path', type = str, default = './models', help='filename to output the pre-trained model')
    parser.add_argument('--num_workers', type=int, default = 8, help='number of workers for dataset loading')
    parser.add_argument('--sub_token_path', type=str, default = './tokens/jars', help='sub tokens vocab and embedding path')
    parser.add_argument('--emb_file', type=str, default = 'emb_100.txt', help='embedding txt path')
    parser.add_argument('--mode', type=str, default = "cbow", help = "cbow or skipgram")
    parser.add_argument('--context_pooling', type=str, default="mean",
                        help='how the contexts are pooled (sum, mean, or max)')
    parser.add_argument('--neg_samples', type=int, default=1,
                        help='number of negative contexts per positive context (default: 1)')
    parser.add_argument('--log_file', type = str, default = 'log.txt', help='log file')
    parser.add_argument('--task', type = str, default = 'context', help='Auto Encoder,VGAE or GAE')
    parser.add_argument('--check_all', dest='check_all', action='store_true', default=False) 
    parser.add_argument('--csize', type=int, default=3,
                        help='context size (default: 3).')
    args = parser.parse_args()
    assert os.path.isdir(args.dataset_path)
    train_mode(args)

    


if __name__ == "__main__":
    main()
