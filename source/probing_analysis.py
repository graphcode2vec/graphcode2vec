import argparse
import pickle
from numpy.random import seed
from sklearn.utils import shuffle
from torch._C import device
from torch_geometric.data import DataLoader
import torch
import os
import numpy as np
from utils.model import GNN_encoder
from utils.tools import  TokenIns, inverse_eage, accuracy, f1
from utils.dataset import ProbingDataset, GoogleCodeCloneClassifier, ProbingCorruptedDataset, DeadlockDataset
from torchvision import transforms
from torch_geometric.transforms.remove_isolated_nodes import RemoveIsolatedNodes
import numpy as np
import torch.optim as optim
from utils.AverageMeter import AverageMeter
import pandas as pd
from utils.probing_classifier import PredictionLinearModel
from tqdm import tqdm
from skorch import NeuralNetClassifier
from sklearn.model_selection import cross_val_score
cc = [  'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
            'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
            'hot', 'afmhot', 'gist_heat', 'copper',  'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu' ]

def get_k_fold_data(k, i, X, y): 
    # 返回第i折交叉验证时所需要的训练和测试数据，分开放，X_train为训练数据，X_test为验证数据
    assert k > 1
    fold_size = X.shape[0] // k  # 每份的个数:数据总条数/折数（向下取整）
    
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)  # slice(start,end,step)切片函数 得到测试集的索引
        
        X_part, y_part = X[idx, :], y[idx]			# 只对第一维切片即可
        if j == i: 									# 第i折作test
            X_test, y_test = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat((X_train, X_part), dim=0) # 其他剩余折进行拼接 也仅第一维
            y_train = torch.cat((y_train, y_part), dim=0)
    return X_train, y_train, X_test, y_test 

import random
def k_fold(args,k, device, X, Y, input_dim, num_class):
    index = [i for i in range(len(X))] 
    random.shuffle(index)
    X = torch.tensor(X[index])
    Y = torch.tensor(Y[index])
    acc = []
    f1 = []
    for i in tqdm(range(k)):
        X_train, y_train, X_valid,y_valid = get_k_fold_data(k, i, X, Y)
        dataset_train = torch.utils.data.TensorDataset( X_train ,  y_train )
        dataset_test = torch.utils.data.TensorDataset( X_valid,  y_valid )
        best_acc, best_f1 = train(args, device,input_dim, dataset_train, dataset_test, num_class )
        acc.append( best_acc )
        f1.append( best_f1 )
    print(f"Acc {acc}")
    print(f"F1 {f1}")
    print(f"Avg Acc {np.mean(acc)}")
    print(f"Avg F1 {np.mean(f1)}")
    return np.mean(acc),  np.std(acc), np.mean(f1), np.std(f1)


def train(args, device,input_dim, train_dataset, test_dataset, num_class ):
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False)
    model = PredictionLinearModel(input_dim, num_class)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)  

    optimizer = optim.Adam(model.parameters())
    optimizer.zero_grad()
    criterion = torch.nn.CrossEntropyLoss().to(device)
    num_epochs = args.epochs
    losses = AverageMeter()
    best_acc = 0
    best_f1 = 0 
    for epoch_idx in range(num_epochs):
        running_loss = 0.0
        for i, (gr, y) in enumerate(train_loader):
            # print("Batch shape {}".format(gr.shape) )
            gr = gr.to(device)
            y = y.to(device)
            model.train()
            optimizer.zero_grad()
            #( x1, x2 ) = torch.chunk( gr, 2, dim=1 )
            prey = model(gr)#model(x1, x2)
            loss = criterion(prey, y)
            loss.backward()
            optimizer.step()
            losses.update(loss.item())
            # print statistics
            running_loss += loss.item()
            
        evaloss, accuracy, f = evaluate_loss(model, device,test_loader, criterion)
        # print(
        #             "Epoch %d Eval F1 : %.3f, Accuracy :%.3f"
        #             % (epoch_idx + 1, f, accuracy)
        #         )
        if accuracy > best_acc:
            best_acc = accuracy
            torch.save(model.state_dict(), os.path.join(args.saved_model_path, "saved_model_acc.pt"))
        if f > best_f1:
            best_f1 = f
    
    #print(f"Accuracy {best_acc}, F1 {best_f1}")
    return best_acc, best_f1

def evaluate_loss(model,device, dataloader_test, criterion):
    model.eval()
    losses = AverageMeter()
    ys =[]
    aas =[]
    with torch.no_grad():
        for ( gr, y ) in dataloader_test:
            gr = gr.to(device)
            y = y.to(device)
           # (x1, x2) = torch.chunk( gr, 2, dim=1 )
            prey = model(gr)#model(x1, x2)
            loss = criterion(prey, y)
            _, actual_labels = torch.max(prey, dim=1)
            losses.update(loss.item())
            ys.append(y)
            aas.append(actual_labels)
    y = torch.cat(ys, dim= 0 )
    actual_labels = torch.cat( aas, dim=0 )
    acc, f = accuracy(y.cpu().numpy(), actual_labels.cpu().numpy()), f1(y.cpu().numpy(), actual_labels.cpu().numpy()) 
  
    return losses.avg, acc, f


def visiualize_TSNE(x, y, savepath):
    from openTSNE import TSNE
    from utils import tsne_utils
    import matplotlib.pyplot as plt
    #from matplotlib import colors
    tsne = TSNE(
    perplexity=10,
    metric="euclidean",
    n_jobs=8,
    random_state=42,
    verbose=False,
    )
    embedding_train = tsne.fit(x)
    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    pickle.dump([embedding_train, y], open(savepath+".pk", "wb"))
    tsne_utils.plot(embedding_train, y,ax=ax, colors=tsne_utils.MOUSE_10X_COLORS, s=120 ,alpha=0.8)
    # num_class = len(list( set(y) ))
   
    # classes = [i for i in range(num_class ) ]
    # category_to_color = { i: cc[i] for i in range(num_class )  }
    # category_to_label = { i: f'P{i}' for i in range(num_class )}
    # plt.figure(figsize=(8, 8))
    # for i in classes:    
    #         plt.scatter(z[y==i, 0], z[y==i, 1],  c=category_to_color[i], label=category_to_label[i] ,alpha=0.8)
    plt.axis('off')
    #plt.legend(fontsize=15) # using a size in points
    plt.savefig( savepath )
    #plt.show()
    plt.close()

def load_model(args, device, num_class=1):
    tokenizer_word2vec = TokenIns( 
        word2vec_file=os.path.join(args.sub_token_path, args.emb_file),
        tokenizer_file=os.path.join(args.sub_token_path, "fun.model")
    )
    _, word_emb_dim, vocab_size = tokenizer_word2vec.load_word2vec_embeddings()
    encoder = GNN_encoder(args.num_layer,vocab_size, word_emb_dim, args.lstm_emb_dim,num_class=num_class, JK = args.JK, drop_ratio = args.dropout_ratio, 
                        graph_pooling = args.graph_pooling, gnn_type = args.gnn_type, subword_emb=args.subword_embedding, bidrection=args.bidirection, task=args.task)

    print(f"Load Pretraning model {args.input_model_file}")
    encoder.from_pretrained(args.input_model_file, device)

    return encoder

def getDataset(args, reshuffle=False, name="google"):
    if name=="google":
       dataset = GoogleCodeCloneClassifier( args.dataset_path , dataname=args.dataset) 
       idxlist=dataset.split(reshuffle)
       num_class = 12
       label_name = "y"
    if name=="leetcode":
        dataset = ProbingDataset(args.dataset_path , dataname=args.dataset  )
        dataset.transform = transforms.Compose([ lambda data: inverse_eage(data),RemoveIsolatedNodes()])
        idxlist=dataset.split(reshuffle)
        num_class=10
        label_name = "label"
    if name=="leetcode_corrupted_20":
        dataset = ProbingCorruptedDataset(args.dataset_path , dataname=args.dataset  )
        dataset.transform = transforms.Compose([ lambda data: inverse_eage(data),RemoveIsolatedNodes()])
        idxlist=dataset.split(reshuffle)
        num_class=20
        label_name = "label"
    if name=="leetcode_corrupted_2":
        dataset = ProbingCorruptedDataset(args.dataset_path , dataname=args.dataset  )
        dataset.transform = transforms.Compose([ lambda data: inverse_eage(data),RemoveIsolatedNodes()])
        idxlist=dataset.split(reshuffle)
        num_class=2
        label_name = "y1"
    if name=="deadlock":
        dataset = DeadlockDataset(args.dataset_path , dataname=args.dataset  )
        dataset.transform = transforms.Compose([ lambda data: inverse_eage(data),RemoveIsolatedNodes()])
        idxlist=dataset.split(True)
        num_class=7
        label_name = "y"
    
    return dataset, idxlist,num_class, label_name

def probing_analysis(args, reshuffle=False):
    os.makedirs( args.saved_model_path, exist_ok=True)
    if args.graph_pooling == "set2set":
        args.graph_pooling = [2, args.graph_pooling]

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using Device {device}")
    dataset, idxlist,num_class, label_name = getDataset( args, reshuffle ,args.data)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)

    model = load_model(args, device, num_class)
    model.to(device)
   
    method_label = []
    mReps_cat = []
    mReps_g = []
    seqReps = []
    model.eval()
    for step, batch in enumerate(loader):
        batch = batch.to(device)
        with torch.no_grad():
            method_rep_cat, method_rep_g,  seq_x = model.getVector(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.ins_length)        
       
        method_label.append(  batch[label_name].cpu() )
        mReps_cat.append( method_rep_cat.cpu() )
        mReps_g.append( method_rep_g.cpu() )
        seqReps.append( seq_x.cpu() )
    
    method_label = torch.cat(method_label, dim = 0).numpy()
    mReps_cat = torch.cat(mReps_cat, dim = 0).numpy()
    mReps_g = torch.cat(mReps_g, dim = 0).numpy()
    seqReps = torch.cat(seqReps, dim = 0).numpy()
    print("Data Info")
    print(f"Number of method cluster {len(set(method_label.tolist()))}")
    print(f"Number of Data {mReps_cat.shape[0]}")
    print(mReps_cat.shape)
    print(method_label.shape)
    acc_c, acc_cstd, f1_c, f1_cstd = k_fold(args, args.k_fold, device, mReps_cat, method_label, mReps_cat.shape[1], num_class)

    # model = PredictionLinearModel(int(mReps_cat.shape[1]/2), num_class)
    # model.load_state_dict(torch.load(os.path.join(args.saved_model_path, "saved_model_acc.pt"), map_location="cpu"))
    # tf_rep = torch.tensor( mReps_cat )
    #( x1, x2 ) = torch.chunk( tf_rep, 2, dim=1 )
    #alpha_rep = model(x1, x2).detach().numpy()
    
    acc_g, acc_gstd, f1_g, f1_gstd = k_fold(args, args.k_fold, device,mReps_g, method_label, mReps_g.shape[1], num_class)  
    acc_s, acc_sstd, f1_s, f1_sstd = k_fold(args, args.k_fold, device,seqReps, method_label, seqReps.shape[1], num_class)
    with open(args.saved_model_path+"/k_fold.txt", "w") as f:
        f.write(f"{acc_c},{acc_cstd},{f1_c},{f1_cstd}\n")
        f.write(f"{acc_g},{acc_gstd},{f1_g},{f1_gstd}\n")
        f.write(f"{acc_s},{acc_sstd},{f1_s},{f1_sstd}\n")
    f.close()
    # with open(args.saved_model_path+"/k_fold_alpha.txt", "w") as f:
    #     f.write(f"{model.alpha.item()}\n")
    visiualize_TSNE(seqReps, method_label , args.saved_model_path+f"/tokens_tsne.pdf" )
    visiualize_TSNE(mReps_g, method_label , args.saved_model_path+f"/method_grap_tsne.pdf" )
    visiualize_TSNE(mReps_cat, method_label , args.saved_model_path+f"/method_cat_tsne.pdf" )
    #visiualize_TSNE(alpha_rep, method_label , args.saved_model_path+f"/method_alpha_tsne_${load_emb}_${load_weights}.png" )
 
    

    
    
            

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--k_fold', type=int, default=5,
                        help='number of epochs to train (default: 5)')                    
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
    parser.add_argument('--random', dest='random', action='store_true', default=False) 
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
    parser.add_argument('--gnn_type', type=str, default="gat")
    parser.add_argument('--data', type=str, default = 'DV_PDG', help='root directory of dataset. For now, only classification.')
    parser.add_argument('--dataset', type=str, default = 'DV_PDG', help='root directory of dataset. For now, only classification.')
    parser.add_argument('--dataset_path', type=str, default = 'dataset/probing/', help='root directory of dataset. For now, only classification.')
    parser.add_argument('--input_model_file', type=str, default = 'experiments_res/pretraining_weights/vgae_pretraining_attention/gat/model_gnn_9_0_roc0.960.pth', help='filename to read the model (if there is any)')
    parser.add_argument('--target_token_path', type=str, default = 'dataset/downstream/java-small/target_word_to_index.json', help='Target Vocab')
    parser.add_argument('--saved_model_path', type = str, default = 'RQ3/vgae_gat', help='filename to output the pre-trained model')
    parser.add_argument('--num_workers', type=int, default = 8, help='number of workers for dataset loading')
    parser.add_argument('--sub_token_path', type=str, default = './tokens/jars', help='sub tokens vocab and embedding path')
    parser.add_argument('--emb_file', type=str, default = 'emb_100.txt', help='embedding txt path')
    parser.add_argument('--log_file', type = str, default = 'log.txt', help='log file')
    parser.add_argument('--task', type = str, default = 'probing', help='probing task')
 
    args = parser.parse_args()
    if "leetcode" in args.data:
        args.dataset_path = "dataset/probing"
    if args.data == "jam":
        args.dataset_path = "dataset/downstream/jam"
    assert os.path.isdir(args.dataset_path)
    
    # if args.random:
    #     assert False,"not random"
    #     probing_analysis(args, False, False, False)
    # else:
    assert os.path.isfile(args.input_model_file)
    probing_analysis(args)
   


if __name__ == "__main__":
    main()
