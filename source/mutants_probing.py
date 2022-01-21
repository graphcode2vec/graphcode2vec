import argparse
from torch._C import device
from torch_geometric.data import DataLoader
import torch
import os
from utils.model import GNN_encoder
from utils.tools import  TokenIns, inverse_eage, accuracy, f1
from utils.mutantsdataset import MutantsDataset
from torchvision import transforms
from torch_geometric.transforms.remove_isolated_nodes import RemoveIsolatedNodes
import torch.optim as optim
from utils.AverageMeter import AverageMeter
from utils.probing_classifier import PredictionLinearModel
from decouple import config

DEBUG=config('DEBUG', default=False, cast=bool)
criterion = torch.nn.CrossEntropyLoss()
def train(args, device,input_dim, train_dataset, valid_dataset, num_class ):
   # print(args.batch_size)
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=args.batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    model = PredictionLinearModel(input_dim, num_class)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)  

    optimizer = optim.Adam(model.parameters())
    optimizer.zero_grad()
    
    num_epochs = args.epochs
    losses = AverageMeter()
    best_acc = 0
    best_f1 = 0 
    least_loss = 0
    for _ in range(num_epochs):
        running_loss = 0.0
        for i, (gr, y) in enumerate(train_loader):
            # print("Batch shape {}".format(gr.shape) )
            gr = gr.to(device)
            y = y.to(device)
            model.train()
            optimizer.zero_grad()
            prey = model(gr)
            loss = criterion(prey, y)
            loss.backward()
            optimizer.step()
            losses.update(loss.item())
            # print statistics
            running_loss += loss.item()
            
        evaloss, accuracy, f = evaluate_model(model, device,valid_loader, criterion)
      
            
        if accuracy > best_acc:
            print(f"Best Acc: {best_acc} -> {accuracy}")
            best_acc = accuracy
            torch.save(model.state_dict(), os.path.join(args.saved_model_path, "saved_model_acc.pt"))
        if f > best_f1:
            print(f"Best F1: {best_f1} -> {f}")
            best_f1 = f
            torch.save(model.state_dict(), os.path.join(args.saved_model_path, "saved_model_f1.pt"))

    return best_acc, best_f1, least_loss

def evaluate_model(model,device, dataloader_test, criterion):
    model.eval()
    losses = AverageMeter()
    ys =[]
    aas =[]
    with torch.no_grad():
        for ( gr, y ) in dataloader_test:
            gr = gr.to(device)
            y = y.to(device)
            prey = model(gr)
            loss = criterion(prey, y)
            _, actual_labels = torch.max(prey, dim=1)
            losses.update(loss.item())
            ys.append(y)
            aas.append(actual_labels)
    y = torch.cat(ys, dim= 0 )
    actual_labels = torch.cat( aas, dim=0 )
    acc, f = accuracy(y.cpu().numpy(), actual_labels.cpu().numpy()), f1(y.cpu().numpy(), actual_labels.cpu().numpy(), average="macro")   
    #acc, f = accuracy(y.cpu().numpy(), actual_labels.cpu().numpy()), f1(y.cpu().numpy(), actual_labels.cpu().numpy(), average="weighted")   
    #print(f"{wf}")
    return losses.avg, acc, f


def load_model(args, device, num_class=1):
    tokenizer_word2vec = TokenIns( 
        word2vec_file=os.path.join(args.sub_token_path, args.emb_file),
        tokenizer_file=os.path.join(args.sub_token_path, "fun.model")
    )
    _, word_emb_dim, vocab_size = tokenizer_word2vec.load_word2vec_embeddings()
    encoder = GNN_encoder( args.num_layer,vocab_size, word_emb_dim, args.lstm_emb_dim,\
                            num_class=num_class, JK = args.JK, drop_ratio = args.dropout_ratio, \
                        graph_pooling = args.graph_pooling, gnn_type = args.gnn_type,\
                             subword_emb=args.subword_embedding, bidrection=args.bidirection, task=args.task )

    print(f"Load Pretraning model {args.input_model_file}")
    encoder.from_pretrained(args.input_model_file, device)

    return encoder

def getDataset(args):
    dataset = MutantsDataset(args.dataset_path , dataname=args.dataset  )
    dataset.transform = transforms.Compose([ lambda data: inverse_eage(data),RemoveIsolatedNodes()])
    label_name = "y"
    
    return dataset, label_name

def generate_vector(args, reshuffle=False, num_class=2):
    os.makedirs( args.saved_model_path, exist_ok=True)
    if not os.path.isfile( os.path.join(args.saved_model_path, "data.pt") ):
        if args.graph_pooling == "set2set":
            args.graph_pooling = [2, args.graph_pooling]
        device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
        print(f"Using Device {device}")
        dataset,  label_name = getDataset( args)
        loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers = args.num_workers)
        model = load_model(args, device, num_class)
        model.to(device)
    
        label_list = []
        mReps_cat = []
        mReps_g = []
        seqReps = []
        model.eval()

        for _, batch in enumerate(loader):
            batch = batch.to(device)
            with torch.no_grad():
                method_rep_cat, method_rep_g,  seq_x = model.getVector(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.ins_length)       
            label_list.append(  batch[label_name].cpu() )
            mReps_cat.append( method_rep_cat.cpu() )
            mReps_g.append( method_rep_g.cpu() )
            seqReps.append( seq_x.cpu() )
        
        label_list = torch.cat(label_list, dim = 0).numpy()
        mReps_cat = torch.cat(mReps_cat, dim = 0).numpy()
        mReps_g = torch.cat(mReps_g, dim = 0).numpy()
        seqReps = torch.cat(seqReps, dim = 0).numpy()
        print("Data Info")
        print(f"Number of method cluster {len(set(label_list.tolist()))}")
        print(f"Number of Data {mReps_cat.shape[0]}")
        print(mReps_cat.shape)
        print(label_list.shape)
        split_dict = dataset.split(reshuffle=True)  
        torch.save([ label_list, mReps_cat, mReps_g, seqReps, split_dict  ], \
            os.path.join(args.saved_model_path, "data.pt") )
    else:
        [ label_list, mReps_cat, mReps_g, seqReps, split_dict  ] = torch.load( os.path.join(args.saved_model_path, "data.pt") )
    return label_list, mReps_cat, mReps_g, seqReps, split_dict


def probing_analysis(args, reshuffle=False):
    num_class = args.num_class
    label_list, mReps_cat, _, _, split_dict = generate_vector(args, reshuffle, num_class)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    mReps_cat = torch.tensor(mReps_cat)
    label_list = torch.tensor(label_list)
    input_dim = mReps_cat.shape[1]
    X_train = mReps_cat[split_dict["train"]]
    y_train = label_list[split_dict["train"]]
    X_test = mReps_cat[split_dict["test"]]
    y_test = label_list[split_dict["test"]]
    X_valid = mReps_cat[split_dict["valid"]]
    y_valid = label_list[split_dict["valid"]]
    
    if num_class == 2:
        y_train[y_train==2] = 1
        y_train[y_train==3] = 1
        y_train[y_train==4] = 1
        y_test[y_test==2] = 1
        y_test[y_test==3] = 1
        y_test[y_test==4] = 1
        y_valid[y_valid==2] = 1
        y_valid[y_valid==3] = 1
        y_valid[y_valid==4] = 1
        h = torch.sum(y_train) + torch.sum(y_test) + torch.sum(y_valid)
        print(f"{mReps_cat.shape},{y_train.shape},{y_test.shape},{y_valid.shape} {h}")

    dataset_train = torch.utils.data.TensorDataset( X_train ,  y_train )
    dataset_test = torch.utils.data.TensorDataset( X_test,  y_test )
    dataset_valid = torch.utils.data.TensorDataset( X_valid,  y_valid )

    train(args, device, input_dim, dataset_train, dataset_valid, num_class )
    model = PredictionLinearModel(input_dim, num_class)
    model.to(device)

    test_loader = torch.utils.data.DataLoader(dataset_test, shuffle=False)
  
    model.load_state_dict( torch.load( os.path.join(args.saved_model_path, "saved_model_f1.pt") ) )
    evaloss1, accuracy1, f1 = evaluate_model(model, device, test_loader, criterion)
    print(f"Best F Model, {evaloss1} {accuracy1} {f1} \n")
    model.load_state_dict( torch.load( os.path.join(args.saved_model_path, "saved_model_acc.pt") ) )
    evaloss2, accuracy2, f2 = evaluate_model(model, device, test_loader, criterion)
    print(f"Best Acc Model, {evaloss2} {accuracy2} {f2} \n")
    # model = PredictionLinearModel(int(mReps_cat.shape[1]/2), num_class)
    # alpha_rep = model(x1, x2).detach().numpy()
        
    # with open(args.saved_model_path+"/k_fold_alpha.txt", "w") as f:
    #     f.write(f"{model.alpha.item()}\n")
    # visiualize_TSNE(seqReps, method_label , args.saved_model_path+f"/tokens_tsne.pdf" )
    # visiualize_TSNE(mReps_g, method_label , args.saved_model_path+f"/method_grap_tsne.pdf" )
    # visiualize_TSNE(mReps_cat, method_label , args.saved_model_path+f"/method_cat_tsne.pdf" )
    # visiualize_TSNE(alpha_rep, method_label , args.saved_model_path+f"/method_alpha_tsne_${load_emb}_${load_weights}.png" )
           

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=2048*4,
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
    parser.add_argument('--dataset', type=str, default = 'DV_PDG', help='root directory of dataset. For now, only classification.')
    parser.add_argument('--dataset_path', type=str, default = 'dataset/mutants/', help='root directory of dataset. For now, only classification.')
    parser.add_argument('--input_model_file', type=str, default = 'pretrained_models/context/gat/model_0.pth', help='filename to read the model (if there is any)')
    parser.add_argument('--target_token_path', type=str, default = 'dataset/downstream/java-small/target_word_to_index.json', help='Target Vocab')
    parser.add_argument('--saved_model_path', type = str, default = 'mutants_probing_2/mutants/context/gat', help='filename to output the pre-trained model')
    parser.add_argument('--num_workers', type=int, default = 8, help='number of workers for dataset loading')
    parser.add_argument('--sub_token_path', type=str, default = './tokens/jars', help='sub tokens vocab and embedding path')
    parser.add_argument('--emb_file', type=str, default = 'emb_100.txt', help='embedding txt path')
    parser.add_argument('--log_file', type = str, default = 'mutants_probing/mutants/vge/gat/log.txt', help='log file')
    parser.add_argument('--task', type = str, default = 'probing', help='probing task')
    parser.add_argument('--num_class', type = int, default =2, help='num_class')
 
    args = parser.parse_args()
    assert os.path.isfile(args.input_model_file)
    probing_analysis(args)
   


if __name__ == "__main__":
    main()
