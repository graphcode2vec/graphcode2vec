from utils.mutantsdataset import MutantsDataset
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
from utils.tools import performance, TokenIns, inverse_eage
from utils.pytorchtools import EarlyStopping
from utils.AverageMeter import AverageMeter
from decouple import config
from utils.probing_classifier import PredictionLinearModelFineTune
try:
    from transformers import get_linear_schedule_with_warmup as linear_schedule
except:
    print("import WarmupLinearSchedule instead of get_linear_schedule_with_warmup")
    from transformers import WarmupLinearSchedule as get_linear_schedule_with_warmup
    def linear_schedule(optimizer, num_warmup_steps=100, num_training_steps=100):
        return get_linear_schedule_with_warmup(optimizer, warmup_steps=num_warmup_steps,
                                                    t_total=num_training_steps)
DEBUG=config('DEBUG', default=False, cast=bool)
best_f1 = 0
view_test_f1 = 0
criterion = nn.CrossEntropyLoss()

def train(args, model, device, loader, optimizer, loader_val, loader_test, epoch, saved_model_path, earlystopping, scheduler):
    global best_f1
    global view_test_f1
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
        
        pred = model(batch)    
        if args.num_class == 2:
            batch.y[ batch.y != 0 ] = 1
        else:
            batch.y[ batch.y == 4 ] = 1
        loss = criterion( pred, batch.y)      
         
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        if args.warmup_schedule:
            scheduler.step()  
        trainloss.update(loss.item())
        y_true.extend( batch.y.detach().cpu())
        _, predicted_labels = torch.max( pred, dim=1 )
        y_pred.extend(predicted_labels.detach().cpu())
        if step%args.save_steps == 0 and step !=0 :
            print("Evaluation ")
            model.eval()
            evalloss, accuracy_val, precision_val, recall_val, f1_val, result = eval(args, model, device, loader_val)
            testloss, accuracy_test, precision_test, recall_test, f1_test, result_test = eval(args, model, device, loader_test)
            earlystopping(f1_val, model, performance={"val_f1":f1_val, "test_f1":f1_test,"epoch":epoch, "all":[evalloss, accuracy_val, precision_val, recall_val, f1_val, testloss, accuracy_test, precision_test, recall_test, f1_test,
            result, result_test]})
            model.train()
            print(f"Best Test {view_test_f1}")
            print(f"\nEpoch {step}/{epoch}, Valid, Eval Loss {evalloss}, Accuracy {accuracy_val}, Precision {precision_val}, Recall {recall_val}, F1 {f1_val}"  )
            print(f"\nEpoch {step}/{epoch}, Test, Eval Loss {testloss}, Accuracy {accuracy_test}, Precision {precision_test}, Recall {recall_test}, F1 {f1_test}"  )
            if f1_val > best_f1 :
                best_f1 = f1_val
                view_test_f1 = f1_test
                if earlystopping.save_model:
                    torch.save(model.state_dict(), os.path.join(saved_model_path,  f"best_epoch{epoch}_.pth"))
            res.append([accuracy_val, precision_val, recall_val, f1_val])
    model.eval()
    print("Evaluation ")
    epcoh_res = []
    accuracy_train, precision_train, recall_train, f1_train, = performance(y_true, y_pred, average="macro")
    print(f"\nEpoch {epoch}, Train,  Loss {trainloss.avg}, Accuracy {accuracy_train}, Precision {precision_train}, Recall {recall_train}, F1 {f1_train}"  )
    epcoh_res.extend( [accuracy_train, precision_train, recall_train, f1_train ] )
    evalloss, accuracy_val, precision_val, recall_val, f1_val,_ = eval(args, model, device, loader_val)
    print(f"\nEpoch {epoch}, Valid,  Loss {evalloss}, Accuracy {accuracy_val}, Precision {precision_val}, Recall {recall_val}, F1 {f1_val}"  )
    epcoh_res.extend( [ evalloss, accuracy_val, precision_val, recall_val, f1_val ] )
    testloss, accuracy_test, precision_test, recall_test, f1_test,_ = eval(args, model, device, loader_test)
    epcoh_res.extend( [testloss, accuracy_test, precision_test, recall_test, f1_test])
    print(f"\nEpoch {epoch}, Test,  Loss {testloss}, Accuracy {accuracy_test}, Precision {precision_test}, Recall {recall_test}, F1 {f1_test}"  )
    return epcoh_res, res, f1_val, f1_test
   

import gc
def eval(args, model, device, loader):
    y_true = []
    y_prediction = []
    evalloss = AverageMeter()
    for step, batch in enumerate(loader):
        batch = batch.to(device)
        with torch.no_grad():
            outputs = model(batch)
            if args.num_class == 2:
                batch.y[ batch.y!= 0 ] = 1
            else:
                batch.y[ batch.y == 4 ] = 1
            loss = criterion( outputs, batch.y)
            evalloss.update( loss.item() )         
        y_true.append(batch.y.cpu())
        _, predicted_label = torch.max( outputs, dim=1 )
        y_prediction.append(predicted_label.cpu())
    gc.collect()    
    y_true = torch.cat(y_true, dim = 0)
    y_prediction = torch.cat(y_prediction, dim = 0)
    accuracy, precision, recall, f1 = performance( y_true,y_prediction, average="macro")
    accuracy_macro, precision_macro, recall_macro, f1_macro = performance( y_true,y_prediction, average="macro")
    accuracy_weighted, precision_weighted, recall_weighted, f1_weighted = performance( y_true,y_prediction, average="weighted")
    accuracy_micro, precision_micro, recall_micro, f1_micro = performance( y_true,y_prediction, average="micro") 
    result = {"eval_accuracy":accuracy, "eval_precision":precision, "eval_recall":recall,"eval_f1": f1, "macro":[accuracy_macro, precision_macro, recall_macro, f1_macro],
    "weighted":[accuracy_weighted, precision_weighted, recall_weighted, f1_weighted], "micro":[accuracy_micro, precision_micro, recall_micro, f1_micro]}
    return evalloss.avg, accuracy, precision, recall, f1, result

#Data 1 17159, Data 0 7153
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
    dataset_inmemory = MutantsDataset( args.dataset_path , dataname=args.dataset)
    #dataset.transform = transforms.Compose([lambda data: inverse_eage(data)])
    split_dict = dataset_inmemory.split(reshuffle=False, hits=args.hits)  
    dataset = dataset_inmemory.data
    
    #if args.grid_search == "no":
        #print("Grid Search No=============")
    loader = DataLoader([ dataset[i] for i in split_dict["train"] ], batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers,follow_batch=['x_s', 'x_t'])
    loader_test = DataLoader( [ dataset[i] for i in split_dict["test"] ], batch_size=int(args.batch_size/2), shuffle=False, num_workers = args.num_workers,follow_batch=['x_s', 'x_t'])
    loader_val = DataLoader( [ dataset[i] for i in split_dict["valid"] ], batch_size=int(args.batch_size/2), shuffle=False, num_workers = args.num_workers,follow_batch=['x_s', 'x_t'])
    #else:
    #    print("Grid Search Yes=============")
    #    loader = DataLoader([ dataset[i] for i in split_dict["train"][: int(0.5*len(split_dict["train"])) ] ], batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers,follow_batch=['x_s', 'x_t'])
    #    loader_test = DataLoader( [ dataset[i] for i in split_dict["test"][: int(0.5*len(split_dict["test"])) ] ], batch_size=int(args.batch_size/2), shuffle=False, num_workers = args.num_workers,follow_batch=['x_s', 'x_t'])
    #    loader_val = DataLoader( [ dataset[i] for i in split_dict["valid"][: int(0.5*len(split_dict["valid"])) ] ], batch_size=int(args.batch_size/2), shuffle=False, num_workers = args.num_workers,follow_batch=['x_s', 'x_t'])
    num_class = args.num_class

    #set up model
    tokenizer_word2vec = TokenIns( 
        word2vec_file=os.path.join(args.sub_token_path, args.emb_file),
        tokenizer_file=os.path.join(args.sub_token_path, "fun.model")
    )
    embeddings, word_emb_dim, vocab_size = tokenizer_word2vec.load_word2vec_embeddings()
    encoder = GNN_encoder(args.num_layer,vocab_size, word_emb_dim, args.lstm_emb_dim, num_class, JK = args.JK, drop_ratio = args.dropout_ratio, 
                        graph_pooling = args.graph_pooling, gnn_type = args.gnn_type, subword_emb=args.subword_embedding,
                         bidrection=args.bidirection, task="mutants", repWay=args.repWay)
    pytorch_total_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"Trainable Parameters Encoder {pytorch_total_params}\n")
    
    encoder.gnn.embedding.fine_tune_embeddings(True)

    if not args.input_model_file == "-1":
        encoder.gnn.embedding.init_embeddings(embeddings)
        print(f"Load Pretraning model {args.input_model_file}")
        encoder.from_pretrained(args.input_model_file + ".pth", device)

       
    
    pytorch_total_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"\nTotal Number of Parameters of Model, {pytorch_total_params}")
    model = PredictionLinearModelFineTune(600, num_class,encoder,False if args.mutant_type == "no" else True,args.dropratio)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Parameters Model {pytorch_total_params}\n")
    model.to(device)

    #set up optimizer
    optimizer = optim.Adam( model.parameters(), lr=args.lr, weight_decay=args.decay ) 
    args.max_steps=args.epochs*len( loader)
    args.save_steps=len( loader)//10
    args.warmup_steps=args.max_steps//5
    scheduler = linear_schedule(optimizer, num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=args.max_steps)
    args.warmup_schedule = False if args.warmup_schedule == "no" else True
    save_model = False if args.lazy == "yes" else True
    #print(optimizer)
    # # Prepare optimizer and schedule (linear warmup and decay)
    # no_decay = ['bias', 'LayerNorm.weight']
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
    #      'weight_decay': args.decay},
    #     {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    # ]
    # optimizer = AdamW(optimizer_grouped_parameters, lr=1e-4, eps=1e-8)
    f0 = open(args.log_file, "w")
    f1 = open( args.log_file+"_epoch" , "w")
    f = csv.writer( f0 )
    ef = csv.writer( f1 )
    f.writerow(["Accuracy", "Precsion", "Recall", "F1"])
    ef.writerow(["Accuracy", "Precsion", "Recall", "F1", "Val Loss","Val Accuracy", "Val Precsion", "Val Recall", "Val F1", "Test Loss",
                    "Test Accuracy", "Test Precsion", "Test Recall", "Test F1"])
    epoch_res = []
    earlystopping = EarlyStopping(monitor="f1", patience=50, verbose=True, path=args.saved_model_path, save_model=save_model)
    for epoch in range(1, args.epochs+1):
            print("====epoch " + str(epoch))
            performance_model, evalstepres,f1_val, f1_test  = train(args, model, device, loader, optimizer, loader_val, loader_test,epoch, args.saved_model_path, earlystopping, scheduler)
            #res.extend( pre_res_val )
            epoch_res.append( performance_model )
            for r in evalstepres:
                f.writerow(r)
            #earlystopping(f1_val, model, performance={"val_f1":f1_val, "test_f1":f1_test, "all":performance_model})
            if earlystopping.early_stop:
                print("Reach Patience, and Stop training")
                print(f"Best Val Acc {earlystopping.best_score}")
                model.loadWholeModel( os.path.join(args.saved_model_path, "saved_model.pt"), device  )
                model.eval()
                testloss, accuracy_test, precision_test, recall_test, f1_test,result = eval(args, model, device, loader_test)
                print(f"Best Test,  Loss {testloss}, Accuracy {accuracy_test}, Precision {precision_test}, Recall {recall_test}, F1 {f1_test}, {result}"  )
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
    parser.add_argument('--dataset_path', type=str, default = 'dataset/mutants_small/', help='root directory of dataset. For now, only classification.')
    parser.add_argument('--input_model_file', type=str, default = 'pretrained_models/context/gat/model_0', help='filename to read the model (if there is any)')
    parser.add_argument('--target_token_path', type=str, default = 'dataset/downstream/java-small/target_word_to_index.json', help='Target Vocab')
    parser.add_argument('--saved_model_path', type = str, default = 'results/mutants_class_2/context', help='filename to output the pre-trained model')
    parser.add_argument('--num_workers', type=int, default = 8, help='number of workers for dataset loading')
    parser.add_argument('--sub_token_path', type=str, default = './tokens/jars', help='sub tokens vocab and embedding path')
    parser.add_argument('--emb_file', type=str, default = 'emb_100.txt', help='embedding txt path')
    parser.add_argument('--log_file', type = str, default = 'log.txt', help='log file')
    parser.add_argument('--num_class', type = int, default =2, help='num_class')
    parser.add_argument('--dropratio', type = float, default =0.25, help='drop_ratio')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--warmup_schedule', type=str, default="no",
                        help='warmup')
    parser.add_argument('--mutant_type', type=str, default="no",
                        help='mutantype')
    parser.add_argument('--lazy', type=str, default="no",
                        help='save model')
    parser.add_argument('--grid_search', type=str, default="no",
                        help='grid_search')
    parser.add_argument('--hits', type=int, default=-1)
    
    args = parser.parse_args()
    with open(args.saved_model_path+'/commandline_args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    train_mode(args)

    


if __name__ == "__main__":
    main()
