from utils.mutantsdataset import MutantsSingleDataset
import argparse
import json
from torch_geometric.data import DataLoader
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm
import numpy as np
from torchvision import transforms
from utils.model import  GNN_encoder
from utils.tools import performance, TokenIns,inverse_eage
from utils.pytorchtools import EarlyStopping
from utils.AverageMeter import AverageMeter
from decouple import config
DEBUG=config('DEBUG', default=False, cast=bool)
best_f1 = 0
criterion = nn.CrossEntropyLoss()

   
def vectorize(model, device, loader):
    graph_seq, graph, seq = [], [], []
    id_list = []
    for _, batch in enumerate(loader):
        batch = batch.to(device)
        with torch.no_grad():
            bgraph_seq, bgraph, bseq = model.getVector(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.ins_length)  
        id_list.append(batch.graphID)    
        graph_seq.append(bgraph_seq.cpu())
        graph.append(bgraph.cpu())
        seq.append(bseq.cpu())
    
    graph_seq = torch.cat(graph_seq, dim=0)
    graph = torch.cat(graph, dim=0)
    seq = torch.cat(seq, dim=0)
    id_list = torch.cat(id_list, dim=0)
    return graph_seq, graph, seq, id_list
   
    

def compute_vector(args, project_dir, outputfolder, model=None):
    if args.graph_pooling == "set2set":
        args.graph_pooling = [2, args.graph_pooling]

    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    #set up mutants dataset
    dataset_mutants = MutantsSingleDataset(  project_dir , dataname=args.dataset, mutants=True)
    dataset_mutants.transform = transforms.Compose([lambda data: inverse_eage(data)])
    loader = DataLoader(dataset_mutants, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
    num_class = dataset_mutants.classes
    if model is None:
        #set up model
        tokenizer_word2vec = TokenIns( 
            word2vec_file=os.path.join(args.sub_token_path, args.emb_file),
            tokenizer_file=os.path.join(args.sub_token_path, "fun.model")
        )
        embeddings, word_emb_dim, vocab_size = tokenizer_word2vec.load_word2vec_embeddings()
        model = GNN_encoder(args.num_layer,vocab_size, word_emb_dim, args.lstm_emb_dim, num_class, JK = args.JK, drop_ratio = args.dropout_ratio, 
                            graph_pooling = args.graph_pooling, gnn_type = args.gnn_type, subword_emb=args.subword_embedding,
                            bidrection=args.bidirection, task="defects4j", repWay=args.repWay)
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable Parameters {pytorch_total_params}\n")
        
        model.gnn.embedding.fine_tune_embeddings(True)
        if not args.input_model_file == "-1":
            model.gnn.embedding.init_embeddings(embeddings)
            print(f"Load Pretraning model {args.input_model_file}")
            model.from_pretrained(args.input_model_file + ".pth", device)
        
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nTotal Number of Parameters of Model, {pytorch_total_params}")
        model.to(device)

    graph_seq_tensor, graph_tensor, seq_tensor,mutantgraphID = vectorize(model, device, loader)

    #set up original dataset
    dataset_original = MutantsSingleDataset( project_dir, dataname=args.dataset, mutants=False)
    dataset_original.transform = transforms.Compose([ lambda data: inverse_eage(data) ])
    loader_original = DataLoader(dataset_original, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    graph_seq_org_tensor, graph_org_tensor, seq_org_tensor,orggraphID = vectorize( model, device, loader_original )
    #[ mutant2_orggraph_id, original_graph_id_list , mutant_graph_labels, original_graph_labels ] = dataset_original.dataInfo
    # killlabel = {"LIVE":0, "FAIL":1, "EXC":2, "TIME":3, "Original":4}

    torch.save({"mutant_vector":[graph_seq_tensor, graph_tensor, seq_tensor], "dataInfo":dataset_original.dataInfo,"dataset_original":orggraphID,"dataset_mutants":mutantgraphID,"original_vector":[graph_seq_org_tensor, graph_org_tensor, seq_org_tensor] }, \
        os.path.join(outputfolder, "vector.pt"))
    return model


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--dropout_ratio', type=float, default=0.2,
                        help='dropout ratio (default: 0.2)')
    parser.add_argument('--evaluation', dest='evaluation', action='store_true', default=False) 
    #parser.add_argument('--onlyseq', dest='onlyseq', action='store_true', default=False) 
    #parser.add_argument('--reshuffle', dest='reshuffle', action='store_true', default=False) 
    parser.add_argument('--remove_gnn_attention', dest='remove_gnn_attention', action='store_true', default=False) 
    parser.add_argument('--test', type=str, dest='test', default="") 
    

    parser.add_argument('--subword_embedding', type=str, default="lstm",
                        help='embed  (bag, lstmbag, gru, lstm, attention, selfattention)')
    parser.add_argument('--bidirection', dest='bidirection', action='store_true', default=True) 

    parser.add_argument('--lstm_emb_dim', type=int, default=150,
                        help='lstm embedding dimensions (default: 512)')
   

    parser.add_argument('--graph_pooling', type=str, default="attention",
                        help='graph level pooling (sum, mean, max, set2set, attention)')
    
    parser.add_argument('--JK', type=str, default="sum",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--gnn_type', type=str, default="gat")
    parser.add_argument('--repWay', type=str, default="append", help='seq, append, graph, alpha')
    parser.add_argument('--nonodeembedding', dest='nonodeembedding', action='store_true', default=False)
    parser.add_argument('--dataset', type=str, default = 'DV_PDG', help='root directory of dataset. For now, only classification.')
    parser.add_argument('--dataset_path', type=str, default = 'dataset/mutants', help='root directory of dataset. For now, only classification.')
    parser.add_argument('--input_model_file', type=str, default = '', help='filename to read the model (if there is any)')
    parser.add_argument('--target_token_path', type=str, default = 'dataset/downstream/java-small/target_word_to_index.json', help='Target Vocab')
    parser.add_argument('--saved_data_path', type = str, default = 'dataset/mutants', help='filename to output the pre-trained model')
    parser.add_argument('--num_workers', type=int, default = 8, help='number of workers for dataset loading')
    parser.add_argument('--sub_token_path', type=str, default = './tokens/jars', help='sub tokens vocab and embedding path')
    parser.add_argument('--emb_file', type=str, default = 'emb_100.txt', help='embedding txt path')
    parser.add_argument('--log_file', type = str, default = 'log.txt', help='log file')
    args = parser.parse_args()
    with open(args.saved_data_path+'/commandline_args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    model = None
    
    for pname in os.listdir(args.dataset_path):
        if not os.path.isdir( os.path.join(args.dataset_path, pname) ):
            continue
        outputfolder = os.path.join(args.saved_data_path, f"{pname}"  )
        print(outputfolder)
        print(f"{pname}")
        os.makedirs(outputfolder, exist_ok=True )
        model = compute_vector(args, os.path.join(args.dataset_path, pname) , outputfolder, model)




if __name__ == "__main__":
    main()
