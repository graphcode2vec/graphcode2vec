import pickle
from utils.dataset import ProgramDataset
import argparse
import json
from torch_geometric.data import DataLoader
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from tqdm import tqdm
import numpy as np
from torchvision import transforms
from utils.model import  GNN_encoder
from utils.tools import subtoken_match, Vocab, TokenIns,inverse_eage
from utils.pytorchtools import EarlyStopping
from utils.AverageMeter import AverageMeter
from decouple import config

DEBUG=config('DEBUG', default=False, cast=bool)
step_eval=1 if DEBUG else 200
criterion = nn.CrossEntropyLoss()
best_f1 = 0

import gc
def eval(args, model, device, loader, subtokens_lookup):
    y_true = []
    y_prediction = []
    evalloss = AverageMeter()
    for step, batch in enumerate(loader):
        batch = batch.to(device)
        with torch.no_grad():
            outputs = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.ins_length)
            loss = criterion( outputs, batch.y)
            evalloss.update( loss.item() )         
        y_true.append(batch.y.cpu())
        _, actual_labels = torch.max( outputs, dim=1 )
        y_prediction.append(actual_labels.cpu())
    gc.collect()    
    y_true = torch.cat(y_true, dim = 0)
    y_prediction = torch.cat(y_prediction, dim = 0)

    accuracy, precision, recall, f1, pre_res = subtoken_match( y_true,y_prediction, subtokens_lookup)
        
    return evalloss.avg, accuracy, precision, recall, f1, pre_res

def eval_mode(args):
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    #set up dataset
    dataset_test = ProgramDataset( args.dataset_path + "/test", dataname=args.dataset)

    if DEBUG:
       dataset_test = dataset_test[:100]


    dataset_test.transform = transforms.Compose([lambda data: inverse_eage(data)])

    print(f"Test Data Size {len(dataset_test)}")

   # dataset_test = None

    loader_test = DataLoader(dataset_test, batch_size=int(args.batch_size/2), shuffle=False, num_workers = args.num_workers)
    
    #set up target vocab
    target_vocab_lookup = Vocab()
    normalized_names = json.load( open( args.target_token_path ) )
    counter  = 0
    for n, k in normalized_names.items():
        subtokens = n.split("|")
        target_vocab_lookup.append( n, index=k, subtokens=subtokens )
        counter += 1
    num_class = len(normalized_names) #num class
    print(f"Target Size {num_class}")
    #set up model
    tokenizer_word2vec = TokenIns( 
        word2vec_file=os.path.join(args.sub_token_path, args.emb_file),
        tokenizer_file=os.path.join(args.sub_token_path, "fun.model")
    )
    embeddings, word_emb_dim, vocab_size = tokenizer_word2vec.load_word2vec_embeddings()
    model = GNN_encoder(args.num_layer,vocab_size, word_emb_dim, args.lstm_emb_dim, num_class, JK = args.JK, drop_ratio = args.dropout_ratio, 
                        graph_pooling = args.graph_pooling, gnn_type = args.gnn_type, subword_emb=args.subword_embedding,
                         bidrection=args.bidirection, task="javasmall", repWay = args.repWay)
    
        
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal Number of Parameters of Model, {pytorch_total_params}")
    model.to(device)

    #set up optimizer
  
    best_f1_test = 0
    assert os.path.isfile(args.saved_model_path )
    print(f"load model {args.saved_model_path}\n")
    model.loadWholeModel( args.saved_model_path ,device )
    model.eval()
    testloss, accuracy_test, precision_test, recall_test, f1_test,_ = eval(args, model, device, loader_test, target_vocab_lookup)
    print(f"Test,  Loss {testloss}, Accuracy {accuracy_test}, Precision {precision_test}, Recall {recall_test}, F1 {f1_test}"  )
    json.dump( {"accuracy_test":accuracy_test, "precision_test":precision_test,
        "recall_test": recall_test, "f1_test":f1_test}, open(args.savedperformacenfile, "w"))
  


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 200)')
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
   

    parser.add_argument('--graph_pooling', type=str, default="attention",
                        help='graph level pooling (sum, mean, max, set2set, attention)')
    
    parser.add_argument('--JK', type=str, default="sum",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--gnn_type', type=str, default="gat")
    parser.add_argument('--repWay', type=str, default="append", help='seq, append, graph, alpha')

    parser.add_argument('--dataset', type=str, default = 'DV_PDG', help='root directory of dataset. For now, only classification.')
    parser.add_argument('--dataset_path', type=str, default = 'dataset/downstream/java-small', help='root directory of dataset. For now, only classification.')
    parser.add_argument('--input_model_file', type=str, default = '', help='filename to read the model (if there is any)')
    parser.add_argument('--target_token_path', type=str, default = 'dataset/downstream/java-small/target_word_to_index.json', help='Target Vocab')
    parser.add_argument('--saved_model_path', type = str, default = './models', help='filename to output the pre-trained model')
    parser.add_argument('--num_workers', type=int, default = 8, help='number of workers for dataset loading')
    parser.add_argument('--sub_token_path', type=str, default = './tokens/jars', help='sub tokens vocab and embedding path')
    parser.add_argument('--emb_file', type=str, default = 'emb_100.txt', help='embedding txt path')
    parser.add_argument('--log_file', type = str, default = 'log.txt', help='log file')

    parser.add_argument('--savedperformacenfile', type = str, default = 'log.txt', help='log file')
    args = parser.parse_args()
 
    
    eval_mode(args)

    


if __name__ == "__main__":
    main()
