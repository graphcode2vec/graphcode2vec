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
def train(args, model, device, loader, optimizer, loader_val, loader_test, subtokens_lookup, epoch, saved_model_path,earlystopping=None):
    global best_f1
    model.train()
    trainloss = AverageMeter()
    res = []
    res_epcoh = []
    y_true = []
    y_pred = []
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        optimizer.zero_grad()
        if args.subword_embedding == "selfattention":
           batch.x = batch.x[:, :50] 
       # print(batch.x.shape)
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.ins_length)   
        #print(torch.max( batch.y)) 
        loss = criterion( pred, batch.y)            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        optimizer.step()
        trainloss.update(loss.item())
        y_true.extend( batch.y.detach().cpu().tolist())
        _, actual_labels = torch.max( pred, dim=1 )
        y_pred.extend(actual_labels.detach().cpu().tolist())
        if DEBUG or ( step%step_eval == 0 and step !=0 ):
            print("Evaluation ")
            model.eval()
            evalloss, accuracy_val, precision_val, recall_val, f1_val, _ = eval(args, model, device, loader_val, subtokens_lookup)
            model.train()
            print(f"\nEpoch {step}/{epoch}, Best F1 {best_f1}, Validation, Eval Loss {evalloss}, Accuracy {accuracy_val}, Precision {precision_val}, Recall {recall_val}, F1 {f1_val}"  )
            res.append([accuracy_val, precision_val, recall_val, f1_val])
            if earlystopping is not None:
                earlystopping(f1_val, model, performance={"val_f1":f1_val})
            else:
                if f1_val > best_f1 :
                    best_f1 = f1_val
                    torch.save(model.state_dict(), os.path.join(saved_model_path,  f"best_epoch{epoch}_.pth"))
            if earlystopping.early_stop:
                break
            
    model.eval()
    print("Evaluation ")
    accuracy, precision, recall, f1, _ = subtoken_match( y_true,y_pred, subtokens_lookup)
    res_epcoh.append([trainloss.avg, accuracy, precision, recall, f1 ])
    print(f"\nEpoch {epoch}, Train,  Loss {trainloss.avg}, Accuracy {accuracy}, Precision {precision}, Recall {recall}, F1 {f1}"  )
    evalloss, accuracy_val, precision_val, recall_val, f1_val,_ = eval(args, model, device, loader_val, subtokens_lookup)
    res_epcoh.append([evalloss, accuracy_val, precision_val, recall_val, f1_val ])
    print(f"\nEpoch {epoch}, Valid,  Loss {evalloss}, Accuracy {accuracy_val}, Precision {precision_val}, Recall {recall_val}, F1 {f1_val}"  )
    testloss, accuracy_test, precision_test, recall_test, f1_test,_ = eval(args, model, device, loader_test, subtokens_lookup)
    res_epcoh.append([ testloss, accuracy_test, precision_test, recall_test, f1_test ])
    print(f"\nEpoch {epoch}, Test,  Loss {testloss}, Accuracy {accuracy_test}, Precision {precision_test}, Recall {recall_test}, F1 {f1_test}"  )
    return res_epcoh, res
   
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

def train_mode(args):
    os.makedirs( args.saved_model_path, exist_ok=True)
    if args.graph_pooling == "set2set":
        args.graph_pooling = [2, args.graph_pooling]

    # torch.manual_seed(0)
    # np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    #set up dataset
    dataset_val = ProgramDataset( args.dataset_path + "/val", dataname=args.dataset)
    dataset_test = ProgramDataset( args.dataset_path + "/test", dataname=args.dataset)
    dataset = ProgramDataset(  args.dataset_path + "/train", dataname=args.dataset)
    if DEBUG:
       dataset = dataset[:100]
       dataset_val = dataset_val[:100]
       dataset_test = dataset_test[:100]

    dataset.transform = transforms.Compose([lambda data: inverse_eage(data)])
    dataset_val.transform = transforms.Compose([lambda data: inverse_eage(data)])
    dataset_test.transform = transforms.Compose([lambda data: inverse_eage(data)])
    dataset = dataset.shuffle()
    dataset = dataset[:876773]
    print(f"Train Data Size {len(dataset)}")
    print(f"Test Data Size {len(dataset_test)}")
    print(f"Val Data Size {len(dataset_val)}")
   # dataset_test = None

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
    loader_val = DataLoader(dataset_val, batch_size=int(args.batch_size/2), shuffle=False, num_workers = args.num_workers)
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
    
        
    model.gnn.embedding.fine_tune_embeddings(True)
    
    if not args.input_model_file == "-1":
        model.gnn.embedding.init_embeddings(embeddings)
        print(f"Load Pretraning model {args.input_model_file}")
        model.from_pretrained(args.input_model_file + ".pth", device)
    
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal Number of Parameters of Model, {pytorch_total_params}")
    model.to(device)

    #set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)  
    print(optimizer)
    f0 = open(args.log_file, "w")
    f = csv.writer( f0 )
    res = [ ]
    f.writerow(["Accuracy", "Precsion", "Recall", "F1"])
    earlystopping = EarlyStopping(monitor="f1", patience=20, verbose=True, path=args.saved_model_path)
    best_f1_test = 0
    for epoch in range(1, args.epochs+1):
            print("====epoch " + str(epoch))
            eval_epoch, evalstepres  = train(args, model, device, loader, optimizer, loader_val, loader_test, target_vocab_lookup, epoch, args.saved_model_path, earlystopping)
            res.append( eval_epoch )
            for r in evalstepres:
                f.writerow(r)
            if earlystopping.early_stop:
                print(f"Best Val F1 {earlystopping.best_score}")
                model.loadWholeModel( os.path.join(args.saved_model_path, "saved_model.pt") ,device )
                model.eval()
                testloss, accuracy_test, precision_test, recall_test, f1_test,_ = eval(args, model, device, loader_test, target_vocab_lookup)
                print(f"Best Test F1 {f1_test}")
                best_f1_test = f1_test
                break
    f0.close()
    with open(args.log_file+"_res.pt", "wb"  ) as f:
            pickle.dump(res, f)
    with open(args.log_file+"_besttest.txt", "w"  ) as f:
            f.write(f"{best_f1_test}\n")   

def prediction( args ):
    os.makedirs( args.saved_model_path, exist_ok=True)
    if args.graph_pooling == "set2set":
        args.graph_pooling = [2, args.graph_pooling]

    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    #set up dataset
    dataset = ProgramDataset(args.dataset_path + "/test" , dataname=args.dataset)
    dataset.transform = transforms.Compose([lambda data: inverse_eage(data)])
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
    #set up target vocab
    target_vocab_lookup = Vocab()
    normalized_names = json.load( open( args.target_token_path ) )
    counter  = 0
    for n, k in normalized_names.items():
        subtokens = n.split("|")
        target_vocab_lookup.append( n, index=k, subtokens=subtokens )
        counter += 1
    num_class = len(normalized_names) #num class

    #set up model
    tokenizer_word2vec = TokenIns( 
        word2vec_file=os.path.join(args.sub_token_path, args.emb_file),
        tokenizer_file=os.path.join(args.sub_token_path, "fun.model")
    )
    _, word_emb_dim, vocab_size = tokenizer_word2vec.load_word2vec_embeddings()
    model = GNN_encoder(args.num_layer,vocab_size, word_emb_dim, args.lstm_emb_dim, num_class, JK = args.JK, drop_ratio = args.dropout_ratio, 
                        graph_pooling = args.graph_pooling, gnn_type = args.gnn_type, subword_emb=args.subword_embedding,
                         bidrection=args.bidirection,  task="javasmall", repWay = args.repWay)
    
    print(f"Load model from file {args.input_model_file}")
    model.from_pretrained(args.input_model_file, device, maps={"graph_pred_linear.weight":"prediction_linear.weight", "graph_pred_linear.bias":"prediction_linear.bias"})
    
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal Number of Parameters of Model, {pytorch_total_params}")
    model.to(device)
    model.eval()
    evalloss, accuracy_val, precision_val, recall_val, f1_val, pre_res_val = eval(args, model, device, loader, target_vocab_lookup)
    print(f"Test, Eval Loss {evalloss}, Accuracy {accuracy_val}, Precision {precision_val}, Recall {recall_val}, F1 {f1_val}"  )
    with open(args.log_file+"_besttest.txt", "w" ) as f:
        f.write(f"{accuracy_val}, {precision_val},{recall_val},{f1_val}\n")
 

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
    args = parser.parse_args()
 
    with open(args.saved_model_path+'/commandline_args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    train_mode(args)

    


if __name__ == "__main__":
    main()
