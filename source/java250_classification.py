from utils.dataset import Java250Dataset
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

def train(args, model, device, loader, optimizer, loader_val, loader_test, epoch, saved_model_path):
    global best_f1
    model.train()
    trainloss = AverageMeter()
    res = []
    y_true = []
    y_pred = []
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        optimizer.zero_grad()
        if args.subword_embedding == "selfattention":
           batch.x = batch.x[:, :50] 
        
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.ins_length)    
        loss = criterion( pred, batch.y)            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        optimizer.step()
        trainloss.update(loss.item())
        y_true.extend( batch.y.detach().cpu())
        _, actual_labels = torch.max( pred, dim=1 )
        y_pred.extend(actual_labels.detach().cpu())
        if step%2000 == 0 and step !=0 :
            print("Evaluation ")
            model.eval()
            evalloss, accuracy_val, precision_val, recall_val, f1_val = eval(args, model, device, loader_val)
            model.train()
            print(f"\nEpoch {step}/{epoch}, Valid, Eval Loss {evalloss}, Accuracy {accuracy_val}, Precision {precision_val}, Recall {recall_val}, F1 {f1_val}"  )
            if f1_val > best_f1 :
                best_f1 = f1_val
                torch.save(model.state_dict(), os.path.join(saved_model_path,  f"best_epoch{epoch}_.pth"))
            res.append([accuracy_val, precision_val, recall_val, f1_val])
    model.eval()
    print("Evaluation ")
    epcoh_res = []
    accuracy_train, precision_train, recall_train, f1_train, = performance(y_true, y_pred)
    print(f"\nEpoch {epoch}, Train,  Loss {trainloss.avg}, Accuracy {accuracy_train}, Precision {precision_train}, Recall {recall_train}, F1 {f1_train}"  )
    epcoh_res.extend( [accuracy_train, precision_train, recall_train, f1_train ] )
    evalloss, accuracy_val, precision_val, recall_val, f1_val, = eval(args, model, device, loader_val)
    print(f"\nEpoch {epoch}, Valid,  Loss {evalloss}, Accuracy {accuracy_val}, Precision {precision_val}, Recall {recall_val}, F1 {f1_val}"  )
    epcoh_res.extend( [ evalloss, accuracy_val, precision_val, recall_val, f1_val ] )
    testloss, accuracy_test, precision_test, recall_test, f1_test, = eval(args, model, device, loader_test)
    epcoh_res.extend( [testloss, accuracy_test, precision_test, recall_test, f1_test])
    print(f"\nEpoch {epoch}, Test,  Loss {testloss}, Accuracy {accuracy_test}, Precision {precision_test}, Recall {recall_test}, F1 {f1_test}"  )
    return epcoh_res, res, accuracy_val, accuracy_test
   

import gc
def eval(args, model, device, loader):
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
        _, predicted_label = torch.max( outputs, dim=1 )
        y_prediction.append(predicted_label.cpu())
    gc.collect()    
    y_true = torch.cat(y_true, dim = 0)
    y_prediction = torch.cat(y_prediction, dim = 0)
    accuracy, precision, recall, f1 = performance( y_true,y_prediction)    
    return evalloss.avg, accuracy, precision, recall, f1

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
    dataset = Java250Dataset( args.dataset_path , dataname=args.dataset)
    dataset.transform = transforms.Compose([lambda data: inverse_eage(data)])
    split_dict = dataset.split(reshuffle=False)  
  
    if DEBUG:
        loader = DataLoader(dataset[:100], batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
        loader_test = DataLoader(dataset[100:200], batch_size=int(args.batch_size/2), shuffle=False, num_workers = args.num_workers)
        loader_val = DataLoader(dataset[200:300], batch_size=int(args.batch_size/2), shuffle=False, num_workers = args.num_workers)
    else:
        loader = DataLoader(dataset[split_dict["train"]], batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
        loader_test = DataLoader(dataset[split_dict["test"]], batch_size=int(args.batch_size/2), shuffle=False, num_workers = args.num_workers)
        loader_val = DataLoader(dataset[split_dict["valid"]], batch_size=int(args.batch_size/2), shuffle=False, num_workers = args.num_workers)
    num_class = 250 

    #set up model
    tokenizer_word2vec = TokenIns( 
        word2vec_file=os.path.join(args.sub_token_path, args.emb_file),
        tokenizer_file=os.path.join(args.sub_token_path, "fun.model")
    )
    embeddings, word_emb_dim, vocab_size = tokenizer_word2vec.load_word2vec_embeddings()
    model = GNN_encoder(args.num_layer,vocab_size, word_emb_dim, args.lstm_emb_dim, num_class, JK = args.JK, drop_ratio = args.dropout_ratio, 
                        graph_pooling = args.graph_pooling, gnn_type = args.gnn_type, subword_emb=args.subword_embedding,
                         bidrection=args.bidirection, task="java250", repWay=args.repWay)
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

    #set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay) 
    #print(list(model.parameters()) )
    #print( model.alpha.requires_grad )
    print(optimizer)
    f0 = open(args.log_file, "w")
    f1 = open( args.log_file+"_epoch" , "w")
    f = csv.writer( f0 )
    ef = csv.writer( f1 )
    f.writerow(["Accuracy", "Precsion", "Recall", "F1"])
    ef.writerow(["Accuracy", "Precsion", "Recall", "F1", "Val Loss","Val Accuracy", "Val Precsion", "Val Recall", "Val F1", "Test Loss",
                    "Test Accuracy", "Test Precsion", "Test Recall", "Test F1"])
    epoch_res = []
    earlystopping = EarlyStopping(monitor="acc", patience=10, verbose=True, path=args.saved_model_path)
    for epoch in range(1, args.epochs+1):
            print("====epoch " + str(epoch))
            performance_model, evalstepres,accuracy_val, accuracy_test  = train(args, model, device, loader, optimizer, loader_val, loader_test,epoch, args.saved_model_path)
            #res.extend( pre_res_val )
            epoch_res.append( performance_model )
            for r in evalstepres:
                f.writerow(r)
            earlystopping(accuracy_val, model, performance={"val_acc":accuracy_val, "test_acc":accuracy_test, "all":performance_model})
            if earlystopping.early_stop:
                print("Reach Patience, and Stop training")
                print(f"Best Val Acc {earlystopping.best_score}")
                model.loadWholeModel( os.path.join(args.saved_model_path, "saved_model.pt"), device  )
                model.eval()
                testloss, accuracy_test, precision_test, recall_test, f1_test, = eval(args, model, device, loader_test)
                print(f"Best Test,  Loss {testloss}, Accuracy {accuracy_test}, Precision {precision_test}, Recall {recall_test}, F1 {f1_test}"  )
                break
    f0.close()
    for r in epoch_res:
        ef.writerow(r)
    f1.close()
    # with open(args.log_file+"_res.txt", "w"  ) as f:
    #         f.writelines(res)

 

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
    parser.add_argument('--dataset_path', type=str, default = 'dataset/downstream/java-250-call', help='root directory of dataset. For now, only classification.')
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
