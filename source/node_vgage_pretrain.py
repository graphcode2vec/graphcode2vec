import argparse
import json
from random import random
from torch_geometric.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from tqdm import tqdm
import numpy as np
from utils.model import GNN_encoder
from torch_geometric.utils import negative_sampling
from utils.tools import  TokenIns, inverse_eage, performance
from utils.AverageMeter import AverageMeter
from torch_geometric.nn import  GAE, VGAE
from utils.dataset import UnsupervisedDataset
from torchvision import transforms
from torch_geometric.transforms.remove_isolated_nodes import RemoveIsolatedNodes
import random
from decouple import config

DEBUG=config('DEBUG', default=False, cast=bool)
step_eval=1 if DEBUG else 200
criterion = nn.CrossEntropyLoss()

def train_gae(args, model, device, loader, optimizer, loader_val, loader_test, epoch, variational=False):
    model.train()
    trainloss = AverageMeter()
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        optimizer.zero_grad()
        if args.subword_embedding == "selfattention":
           batch.x = batch.x[:, :50] 
        z = model.encode(batch.x, batch.edge_index,  batch.edge_attr, batch.batch, batch.ins_length)
        loss = model.recon_loss(z, batch.edge_index)
        if variational:
            loss = loss + (1 / batch.num_nodes) * model.kl_loss()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        optimizer.step()
        trainloss.update(loss.item())
    model.eval()
    #model.training = False
    print("Evaluation ")
    eval_loss, roc, precsion = eval_gae(args, model, device, loader_val, variational)
    print(f"Epoch {epoch}, Validation, Eval loss {eval_loss},  Eval ROC {roc}, Precision {precsion}")
    test_loss, roc_test, precsion_test = eval_gae(args, model, device, loader_test, variational)
    print(f"Epoch {epoch}, Test, Test loss {test_loss} Test ROC {roc_test},Test Precision {precsion_test}")
    return roc_test
    
def train(args, model, device, loader, optimizer, loader_val, loader_test, epoch):
    model.train()
    trainloss = AverageMeter()
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        optimizer.zero_grad()
        if args.subword_embedding == "selfattention":
           batch.x = batch.x[:, :50] 
        outputs = model(batch.x, batch.edge_index,  batch.edge_attr, batch.batch, batch.ins_length)
        loss = criterion( outputs, batch.node_type) 
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        optimizer.step()
        trainloss.update(loss.item())
    model.eval()
    #model.training = False
    print("Evaluation ")
    eval_loss, accuracy_val, precision_val, recall_val, f1_val= eval(args, model, device, loader_val)
    print(f"Epoch {epoch}, Validation, Eval loss {eval_loss},  Eval Acc {accuracy_val}, Precision {precision_val}, Recall {recall_val}, F1 {f1_val} ")
    test_loss, accuracy_test, precision_test, recall_test, f1_test = eval(args, model, device, loader_test)
    print(f"Epoch {epoch}, Test, Test loss {test_loss} Eval Acc {accuracy_test}, Precision {precision_test}, Recall {recall_test}, F1 {f1_test}")
    return f1_test

def eval(args, model, device, loader):
    y_true = []
    y_prediction = []
    evalloss = AverageMeter()
    for step, batch in enumerate(loader):
        batch = batch.to(device)
        with torch.no_grad():
            outputs = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.ins_length)
            loss = criterion( outputs, batch.node_type)
            evalloss.update( loss.item() )         
        y_true.append(batch.node_type.cpu())
        _, actual_labels = torch.max( outputs, dim=1 )
        y_prediction.append(actual_labels.cpu())
    gc.collect()    
    y_true = torch.cat(y_true, dim = 0)
    y_prediction = torch.cat(y_prediction, dim = 0)
    accuracy, precision, recall, f1 = performance( y_true,y_prediction)
    return evalloss.avg, accuracy, precision, recall, f1

def eval_gae(args, model, device, loader, variational):
    eval_roc = AverageMeter()
    eval_precision = AverageMeter()
    eval_loss = AverageMeter()
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        with torch.no_grad():
            z  = model.encode(batch.x, batch.edge_index,  batch.edge_attr, batch.batch, batch.ins_length)
            neg_edge_index = negative_sampling(edge_index=batch.edge_index, num_nodes=batch.num_nodes, num_neg_samples=batch.edge_index.size(1))
            loss = model.recon_loss(z, batch.edge_index)
            if variational:
                loss = loss + (1 / batch.num_nodes) * model.kl_loss()
            eval_loss.update(loss.item())
            roc, avg_precision = model.test(z, batch.edge_index, neg_edge_index)     
            eval_roc.update(roc)
            eval_precision.update(avg_precision)
    return eval_loss.avg, eval_roc.avg, eval_precision.avg

import gc
def create_model(args, num_class=1):
    #set up model
    tokenizer_word2vec = TokenIns( 
        word2vec_file=os.path.join(args.sub_token_path, args.emb_file),
        tokenizer_file=os.path.join(args.sub_token_path, "fun.model")
    )
    embeddings, word_emb_dim, vocab_size = tokenizer_word2vec.load_word2vec_embeddings()
    encoder = GNN_encoder(args.num_layer,vocab_size, word_emb_dim, args.lstm_emb_dim,num_class=num_class, JK = args.JK, drop_ratio = args.dropout_ratio, 
                        graph_pooling = args.graph_pooling, gnn_type = args.gnn_type, subword_emb=args.subword_embedding, bidrection=args.bidirection, task=args.task)
    encoder.gnn.embedding.init_embeddings(embeddings)
    encoder.gnn.embedding.fine_tune_embeddings(False)
    
    if args.task == "GAE":
        print("GAE encoder")
        model = GAE(encoder)
    elif args.task == "VGAE":
        print("VGAE encoder")
        model = VGAE(encoder)
    else:
        model = encoder
    return model

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
    if DEBUG:
        dataset = UnsupervisedDataset( args.dataset_path , dataname=args.dataset, load_index=0, check_all=False)
        check_all = False
        data_partitions = dataset.get_partitions()
    else:
        dataset = UnsupervisedDataset( args.dataset_path , dataname=args.dataset, load_index=0, check_all=args.check_all)
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
    encoder = GNN_encoder(args.num_layer,vocab_size, word_emb_dim, args.lstm_emb_dim, num_class=num_class, JK = args.JK, drop_ratio = args.dropout_ratio, 
                        graph_pooling = args.graph_pooling, gnn_type = args.gnn_type, subword_emb=args.subword_embedding, bidrection=args.bidirection, task=args.task)
    encoder.gnn.embedding.init_embeddings(embeddings)
    encoder.gnn.embedding.fine_tune_embeddings(False)
    
    if args.task == "GAE":
        print("GAE encoder")
        model = GAE(encoder)
    elif args.task == "VGAE":
        print("VGAE encoder")
        model = VGAE(encoder)
    else:
        model = encoder
    
    iters = [ i for i in range( args.epochs  )]
    if args.recover != -1:
        start_epoch =  args.recover
        checkpointfile = os.path.join( args.saved_model_path , f"model_{start_epoch}.pth")
        checkpoint = torch.load( checkpointfile )
        model.load_state_dict(checkpoint["model_state_dict"])
        iters = [ i for i in range( start_epoch+1, args.epochs ) ]

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal Number of Parameters of Model, {pytorch_total_params}")
    model.to(device)

    #set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)  
    if args.recover != -1:
       optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print(optimizer)
    variational = False
    if args.task =="VGAE":
        print("Using KL Loss")
        variational = True
    if os.path.isfile(os.path.join( args.saved_model_path , f"partition_index.json") ):
        split_index = json.load(open(os.path.join( args.saved_model_path , f"partition_index.json"), "r"))
    else:
        split_index = {}
    for epoch in iters:
        for p in data_partitions:
            dataset = UnsupervisedDataset( args.dataset_path , dataname=args.dataset, load_index=p, check_all=check_all)
            if DEBUG:
                dataset = dataset[:200]
            dataset.transform = transforms.Compose([ lambda data: inverse_eage(data),RemoveIsolatedNodes()])
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
            loader = DataLoader(dataset[split_index[p]["train"]], batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
            loader_test = DataLoader(dataset[split_index[p]["test"]], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
            loader_val = DataLoader(dataset[split_index[p]["val"] ], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)

            print("==== epoch {}, data Partition {}  ".format( str(epoch) , p))
            if args.task == "node_class":
                roc_test = train(args, model, device, loader, optimizer, loader_val,loader_test, epoch)  
                torch.save(model.gnn.state_dict(), os.path.join( args.saved_model_path , f"model_gnn_{epoch}_{p}_roc{roc_test:.3f}.pth"))
            if args.task in ["GAE", "VGAE"]:
                roc_test = train_gae(args, model, device, loader, optimizer, loader_val,loader_test, epoch, variational=variational)
                torch.save(model.encoder.gnn.state_dict(), os.path.join( args.saved_model_path , f"model_gnn_{epoch}_{p}_roc{roc_test:.3f}.pth"))
            #torch.save(model.state_dict(), os.path.join( args.saved_model_path , f"model.pth"))
            checkpointfile = os.path.join( args.saved_model_path , f"model_{epoch}.pth")
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, checkpointfile)
            print(f"epoch {epoch}, data Partition {p} val loss {roc_test}")
            del dataset
            del loader
            del loader_test
            del loader_val
            del model
            del optimizer
            gc.collect()
            torch.cuda.empty_cache()
            model = create_model(args, num_class)
            checkpoint = torch.load( checkpointfile )
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(device)
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay) 
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            

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
    parser.add_argument('--log_file', type = str, default = 'log.txt', help='log file')
    parser.add_argument('--task', type = str, default = 'VGAE', help='Auto Encoder,VGAE or GAE')
    parser.add_argument('--check_all', dest='check_all', action='store_true', default=False) 
    args = parser.parse_args()
    assert os.path.isdir(args.dataset_path)
    train_mode(args)

    


if __name__ == "__main__":
    main()
