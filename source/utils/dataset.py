import collections
from itertools import repeat
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
import torch
import os
from pathlib import Path
import tqdm 
import pandas as pd
import json
from sklearn.model_selection import train_test_split
import numpy as np
import random
import multiprocessing
from functools import reduce
from utils.tools import inverse_eage


class TestProgramDataset(InMemoryDataset):
    def __init__(self, root, dataname,transform=None, pre_transform=None, pre_filter=None, empty=False):
        self.root = root
        #print(dataname)
        self.dataname = dataname
        super(TestProgramDataset, self).__init__(root=root, transform=transform,  pre_transform=pre_transform, pre_filter=pre_filter)
        #print(self.dataname)
        if not empty:
            print("Load Data")
            self.data, self.slices = torch.load(self.processed_paths[0])
      
    
    def get(self, idx):
        data = Data()
        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(slices[idx],
                                                    slices[idx + 1])
            data[key] = item[s]
        return data

    
    @property
    def raw_file_names(self):
        file_name_list = [ str(f) for f in Path(self.root).rglob(f"{self.dataname}.pt") ]
        # assert len(file_name_list) == 1     # currently assume we have a
        # # single raw file
        return file_name_list
    
    @property
    def processed_file_names(self):
        return f'geometric_data_processed_{self.dataname}_test_metrics.pt'
    
    def download(self):
        pass
    
    def process(self):
        data_list = []
        graph_labels = []
        graph_ids = []                                     #target_index_to_word
        tkc = json.load( open("dataset/downstream/java-small/target_index_to_word.json", "r") )
        kc = {}
        for k in tkc:
            kc[int(k)] = tkc[k]
        counter1 = json.load( open("dataset/downstream/java-small/test/processed/counter.json") )
        counter2 = json.load( open("dataset/downstream/java-small/train/processed/counter.json") )
        common = set(list(counter1.keys())).intersection( set(list(counter2.keys())) )
        keepm = {}
        for i in common:
            if int(counter1[i])>100 and int(counter2[i]) > 100:
                keepm[i] = int(counter1[i])

        for file in tqdm.tqdm( self.raw_file_names ):
            raw_data = torch.load(os.path.join( file ))
            raw_data_list, raw_graph_labels, raw_graph_id = raw_data[0], raw_data[1], raw_data[2]
           # print(f"{file}, {max(raw_graph_labels)}")
            for i, k in enumerate(raw_graph_labels):
                if kc[k] in keepm:
                    data_list = data_list + [raw_data_list[i]]
                    graph_labels = graph_labels + [raw_graph_labels[i]]
                    graph_ids = graph_ids + [raw_graph_id[i]]
                

        graph_labels_series = pd.Series( graph_labels )
        graph_ids_series = pd.Series( graph_ids )
        graph_labels_series.to_csv(os.path.join(self.processed_dir,
                                               'graph_labels_test_metric.csv'), index=False,
                                  header=False)
        graph_ids_series.to_csv(os.path.join(self.processed_dir,
                                               'graph_ids_test_metric.csv'), index=False,
                                  header=False)
        

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
       

class ProgramDataset(InMemoryDataset):
    def __init__(self, root, dataname,transform=None, pre_transform=None, pre_filter=None, empty=False):
        self.root = root
        #print(dataname)
        self.dataname = dataname
        super(ProgramDataset, self).__init__(root=root, transform=transform,  pre_transform=pre_transform, pre_filter=pre_filter)
        #print(self.dataname)
        if not empty:
            print("Load Data")
            self.data, self.slices = torch.load(self.processed_paths[0])
      
    
    def get(self, idx):
        data = Data()
        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(slices[idx],
                                                    slices[idx + 1])
            data[key] = item[s]
        return data

    
    @property
    def raw_file_names(self):
        file_name_list = [ str(f) for f in Path(self.root).rglob(f"{self.dataname}.pt") ]
        # assert len(file_name_list) == 1     # currently assume we have a
        # # single raw file
        return file_name_list
    
    @property
    def processed_file_names(self):
        return f'geometric_data_processed_{self.dataname}.pt'
    
    def download(self):
        pass
    
    def process(self):
        data_list = []
        graph_labels = []
        graph_ids = []                                     #target_index_to_word
        tkc = json.load( open("dataset/downstream/java-small/target_index_to_word.json", "r") )
        kc = {}
        for k in tkc:
            kc[int(k)] = tkc[k]
        counter = collections.defaultdict(int)
        for file in tqdm.tqdm( self.raw_file_names ):
            raw_data = torch.load(os.path.join( file ))
            raw_data_list, raw_graph_labels, raw_graph_id = raw_data[0], raw_data[1], raw_data[2]
           # print(f"{file}, {max(raw_graph_labels)}")
            data_list = data_list + raw_data_list
            graph_labels = graph_labels + raw_graph_labels
            graph_ids = graph_ids + raw_graph_id
            for k in raw_graph_labels:
                counter[kc[k]] = counter[kc[k]] + 1

        graph_labels_series = pd.Series( graph_labels )
        graph_ids_series = pd.Series( graph_ids )
        graph_labels_series.to_csv(os.path.join(self.processed_dir,
                                               'graph_labels.csv'), index=False,
                                  header=False)
        graph_ids_series.to_csv(os.path.join(self.processed_dir,
                                               'graph_ids.csv'), index=False,
                                  header=False)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print(f"Dict Size {len(counter)}")
        json.dump( counter, open(os.path.join(self.processed_dir,
                                               'counter.json'), "w"))


class DemoExampleDataset(InMemoryDataset):
    def __init__(self, root, dataname,transform=None, pre_transform=None, pre_filter=None):
        self.root = root
        #print(dataname)
        self.dataname = dataname
        super(DemoExampleDataset, self).__init__(root=root, transform=transform,  pre_transform=pre_transform, pre_filter=pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
      
    
    def get(self, idx):
        data = Data()
        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            #print(f"{key},{item}")
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(slices[idx],
                                                    slices[idx + 1])
            data[key] = item[s]
        return data

    
    @property
    def raw_file_names(self):
        file_name_list = [ str(f) for f in Path(self.root).rglob(f"{self.dataname}.pt") ]
        # assert len(file_name_list) == 1     # currently assume we have a
        # # single raw file
        return file_name_list
    
    @property
    def processed_file_names(self):
        return f'geometric_data_processed_{self.dataname}.pt'
    
    def download(self):
        pass
    
    def process(self):
        counter = collections.defaultdict(int)
        #for file in tqdm.tqdm( self.raw_file_names ):
        raw_data = torch.load(os.path.join( self.raw_file_names[0] ))
        data_list, graph_labels, graph_ids = raw_data[0], raw_data[1], raw_data[2]
        print(data_list)
        graph_labels_series = pd.Series( graph_labels )
        graph_ids_series = pd.Series( graph_ids )
        graph_labels_series.to_csv(os.path.join(self.processed_dir,
                                               'graph_labels.csv'), index=False,
                                  header=False)
        graph_ids_series.to_csv(os.path.join(self.processed_dir,
                                               'graph_ids.csv'), index=False,
                                  header=False)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print(f"Dict Size {len(counter)}")
        json.dump( counter, open(os.path.join(self.processed_dir,
                                               'counter.json'), "w"))

class GoogleCodeCloneClassifier(InMemoryDataset):
    def __init__(self, root, dataname,transform=None, pre_transform=None, pre_filter=None):
        self.root = root
        #print(dataname)
        self.dataname = dataname
        self.numclasses = 12
        super(GoogleCodeCloneClassifier, self).__init__(root=root, transform=transform,  pre_transform=pre_transform, pre_filter=pre_filter)
      
        self.data, self.slices = torch.load(self.processed_paths[0])
        
        
    def split(self, reshuffle=False):
        import pickle
        if reshuffle:
            train_idx =  pickle.load( open( os.path.join(self.root, "train.csv.npy") , "rb") )
            test_idx = pickle.load( open( os.path.join(self.root, "test.csv.npy") , "rb") )
            total_size = test_idx.size + train_idx.size
            total_size = len(self)
            indexes = np.arange(total_size)
            np.random.shuffle(indexes)
            test_size = int(0.2*total_size)
            train_size = total_size  - test_size
            test_idx = indexes[:test_size]
            train_idx = indexes[test_size:]
            pickle.dump(test_idx, open( os.path.join(self.root, "test.csv.npy") , "wb"))
            pickle.dump(train_idx, open( os.path.join(self.root, "train.csv.npy") , "wb"))
        else:
            print("Load from the previsou splitting")
            train_idx =  pickle.load( open( os.path.join(self.root, "train.csv.npy") , "rb") )#pd.read_csv(osp.join(path, 'train.csv'),  header = None).values.T[0]
            #valid_idx = pickle.load( open( os.path.join(self.root, "valid.csv.npy") , "rb") )#pd.read_csv(osp.join(path, 'valid.csv'),  header = None).values.T[0]
            test_idx = pickle.load( open( os.path.join(self.root, "test.csv.npy") , "rb") )#pd.read_csv(osp.join(path, 'test.csv'),  header = None).values.T[0]
        
        
        return {'train': torch.tensor(train_idx, dtype = torch.long), 'test': torch.tensor(test_idx, dtype = torch.long)}

        

    def get(self, idx):
        data = Data()
        for key in self.data.keys:
           # print(key)
            item, slices = self.data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(slices[idx],
                                                    slices[idx + 1])
            data[key] = item[s]
        return data

    
    @property
    def raw_file_names(self):
        file_name_list = [ str(f) for f in Path(self.root).rglob(f"{self.dataname}.pt") ]
        # assert len(file_name_list) == 1     # currently assume we have a
        # # single raw file
        return file_name_list
    
    @property
    def processed_file_names(self):
        return [f'geometric_data_{self.dataname}_processed_classifier.pt']    
            #[f'geometric_data_processed_{self.dataname}_train.pt', f'geometric_data_processed_{self.dataname}_test.pt']
    
    def download(self):
        pass
    
    def process(self):
        data_list = []
        graph_labels = []
        graph_ids = []    
       
        #for file in tqdm.tqdm( self.raw_file_names ):
       # print(self.raw_file_names)
        raw_data = torch.load(os.path.join( self.raw_file_names[0] ))
        data_list, graph_labels, _ = raw_data[0], raw_data[1], raw_data[2]
        data_list = [ inverse_eage(data) for data in data_list ]
       
        indexes = np.arange(len(graph_labels ))
        np.random.shuffle(indexes)
        test_size = int(0.2*len(graph_labels))
        train_size = len(graph_labels)  - test_size
        text_idx = indexes[:test_size]
        train_idx = indexes[test_size:]
        import pickle
        print(f"{len(train_idx)}, {len(text_idx)}")
        pickle.dump(text_idx, open( os.path.join(self.root, "test.csv.npy") , "wb"))
        pickle.dump(train_idx, open( os.path.join(self.root, "train.csv.npy") , "wb"))
 

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

        
        print("finishing processing")
   


class GoogleCodeClone(InMemoryDataset):
    def __init__(self, root, dataname,transform=None, pre_transform=None, pre_filter=None, dumpy=False):
        self.root = root
        #print(dataname)
        self.dataname = dataname
        self.numclasses = 12
        self.dumpy = dumpy
        super(GoogleCodeClone, self).__init__(root=root, transform=transform,  pre_transform=pre_transform, pre_filter=pre_filter)
        #print(self.dataname)
        
        if dumpy:
            self.data, self.slices = torch.load(self.processed_paths[1])
        else:    
            self.data, self.slices = torch.load(self.processed_paths[0])
        
        
    def split(self, reshuffle=False):
        import pickle
        if reshuffle:
            train_idx =  pickle.load( open( os.path.join(self.root, "train.csv.npy") , "rb") )
            test_idx = pickle.load( open( os.path.join(self.root, "test.csv.npy") , "rb") )
            total_size = test_idx.size + train_idx.size
            
            assert total_size == len(self), f"{len(self)}, {total_size}"
          
            indexes = np.arange(total_size)
            np.random.shuffle(indexes)
            test_size = int(0.2*total_size)
            train_size = total_size  - test_size
            test_idx = indexes[:test_size]
            train_idx = indexes[test_size:]
            pickle.dump(test_idx, open( os.path.join(self.root, "test.csv.npy") , "wb"))
            pickle.dump(train_idx, open( os.path.join(self.root, "train.csv.npy") , "wb"))
        else:
            print("Load from the previsou splitting")
            train_idx =  pickle.load( open( os.path.join(self.root, "train.csv.npy") , "rb") )#pd.read_csv(osp.join(path, 'train.csv'),  header = None).values.T[0]
            #valid_idx = pickle.load( open( os.path.join(self.root, "valid.csv.npy") , "rb") )#pd.read_csv(osp.join(path, 'valid.csv'),  header = None).values.T[0]
            test_idx = pickle.load( open( os.path.join(self.root, "test.csv.npy") , "rb") )#pd.read_csv(osp.join(path, 'test.csv'),  header = None).values.T[0]
        
        
        return {'train': torch.tensor(train_idx, dtype = torch.long), 'test': torch.tensor(test_idx, dtype = torch.long)}

        

    def get(self, idx):
        if self.dumpy:
            data = PairData()
        else:
            data = Data()
        for key in self.data.keys:
           # print(key)
            item, slices = self.data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(slices[idx],
                                                    slices[idx + 1])
            data[key] = item[s]
        return data

    
    @property
    def raw_file_names(self):
        file_name_list = [ str(f) for f in Path(self.root).rglob(f"{self.dataname}.pt") ]
        # assert len(file_name_list) == 1     # currently assume we have a
        # # single raw file
        return file_name_list
    
    @property
    def processed_file_names(self):
        if self.dumpy:
            return [f'geometric_data_{self.dataname}_processed.pt', f'dumpy_geometric_data_{self.dataname}_processed.pt'] 
        else:
             return [f'geometric_data_{self.dataname}_processed.pt']    
            #[f'geometric_data_processed_{self.dataname}_train.pt', f'geometric_data_processed_{self.dataname}_test.pt']
    
    def download(self):
        pass
    
    def process(self):
        data_list = []
        graph_labels = []
        graph_ids = []                                     #target_index_to_word
       
        #for file in tqdm.tqdm( self.raw_file_names ):
       # print(self.raw_file_names)
        raw_data = torch.load(os.path.join( self.raw_file_names[0] ))
        data_list, graph_labels, _ = raw_data[0], raw_data[1], raw_data[2]
        data_list = [ inverse_eage(data) for data in data_list ]
        if self.dumpy:
            dumpy_data_list, dumpy_graph_labels = self.make_pairs( data_list, graph_labels,pos_ratio = 1.0, neg_ratio=1.0, add_all_neg=True )
            
            print(f"{np.sum(np.asarray(dumpy_graph_labels)==1)/len(dumpy_graph_labels)}")
            indexes = np.arange(len(dumpy_graph_labels ))
            np.random.shuffle(indexes)
            test_size = int(0.2*len(dumpy_graph_labels))
            train_size = len(dumpy_graph_labels)  - test_size
            text_idx = indexes[:test_size]
            train_idx = indexes[test_size:]
            import pickle
            print(f"{len(train_idx)}, {len(text_idx)}")
            pickle.dump(text_idx, open( os.path.join(self.root, "test.csv.npy") , "wb"))
            pickle.dump(train_idx, open( os.path.join(self.root, "train.csv.npy") , "wb"))
            dumpy_data, dumpy_slices = self.collate(dumpy_data_list)
            torch.save((dumpy_data, dumpy_slices), self.processed_paths[1])

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

        
        print("finishing processing")
   
    def make_pairs(self, data_list, graph_labels, pos_ratio = 1.0, neg_ratio=1.0, add_all_neg=True):
        #indices = np.random.permutation(len(graph_labels)[0])
        y_dist = np.bincount(graph_labels)
        positive_count = reduce(lambda n1, n2: n1+n2, map(lambda num: num*num/2,
                                            y_dist.tolist()))
        pair_data_list = []
        #X_right = []
        pairlable_y = []
        p = positive_count * neg_ratio * pos_ratio / (len(data_list) * len(data_list) / 2)
        for i in tqdm.tqdm( range(len(data_list))):
            for j in range(i + 1, len(data_list)):
                if graph_labels[i] == graph_labels[j] and np.random.rand(1)[0] <= pos_ratio:
                    #X_left.append(data_list[i])
                    #X_right.append(data_list[j])
                    pair_data_list.append(PairData(data_list[i].edge_index, torch.ones( data_list[i].x.size()[0]),data_list[i].edge_attr,
                             data_list[j].edge_index,  torch.ones( data_list[j].x.size()[0]), data_list[j].edge_attr, torch.tensor(1), torch.tensor([i]),  torch.tensor([j])))
                    pairlable_y.append(1)
                elif np.random.rand(1)[0] <= p or add_all_neg:
                    pair_data_list.append(PairData(data_list[i].edge_index, torch.ones( data_list[i].x.size()[0]),  data_list[i].edge_attr,
                             data_list[j].edge_index,  torch.ones( data_list[j].x.size()[0]),data_list[j].edge_attr, torch.tensor(0),  torch.tensor([i]),  torch.tensor([j])))
                    pairlable_y.append(0)
        return pair_data_list, pairlable_y
        
class PairData(Data):
    
    def __init__(self, edge_index_s=None, x_s=None, edge_attr_s=None, edge_index_t=None, x_t=None,  edge_attr_t=None, y=None, s_i=None, t_j=None):
        super(PairData, self).__init__()
        self.edge_index_s = edge_index_s
        self.edge_attr_s = edge_attr_s
        self.x_s = x_s
        self.s_i = s_i
      
        self.edge_index_t = edge_index_t
        self.edge_attr_t = edge_attr_t
        self.x_t = x_t
        self.t_j = t_j
        self.y = y

    def __inc__(self, key, value):
        if key == 'edge_index_s':
            return self.x_s.size(0)
        if key == 'edge_index_t':
            return self.x_t.size(0)
        else:
            return super().__inc__(key, value)

class Java250Dataset(InMemoryDataset):
    def __init__(self, root, dataname,transform=None, pre_transform=None, pre_filter=None, train=False, test=False):
        self.root = root
        #print(dataname)
        self.dataname = dataname
        self.numclasses = 250
        super(Java250Dataset, self).__init__(root=root, transform=transform,  pre_transform=pre_transform, pre_filter=pre_filter)
        #print(self.dataname)
        
        self.data, self.slices = torch.load(self.processed_paths[0])
        
        
    def split(self, reshuffle=True):
        import pickle
        if reshuffle:
            raw_data = torch.load(os.path.join( self.raw_file_names[0] ))
            graph_labels = raw_data[1]
            indexes = np.arange(len(graph_labels ))
            np.random.shuffle(indexes)
            test_size = int(0.2*len(graph_labels))
            val_size = int ( 0.2*( len(graph_labels) - test_size ) )
            train_size = len(graph_labels) - val_size - test_size
            test_idx = indexes[:test_size]
            valid_idx = indexes[test_size: test_size+val_size]
            train_idx = indexes[test_size+val_size:]
            pickle.dump(test_idx, open( os.path.join(self.root, "test.csv.npy") , "wb"))
            pickle.dump(valid_idx, open( os.path.join(self.root, "valid.csv.npy") , "wb"))
            pickle.dump(train_idx, open( os.path.join(self.root, "train.csv.npy") , "wb"))
        else:
            print("Load from the previsou splitting")
            train_idx =  pickle.load( open( os.path.join(self.root, "train.csv.npy") , "rb") )#pd.read_csv(osp.join(path, 'train.csv'),  header = None).values.T[0]
            valid_idx = pickle.load( open( os.path.join(self.root, "valid.csv.npy") , "rb") )#pd.read_csv(osp.join(path, 'valid.csv'),  header = None).values.T[0]
            test_idx = pickle.load( open( os.path.join(self.root, "test.csv.npy") , "rb") )#pd.read_csv(osp.join(path, 'test.csv'),  header = None).values.T[0]

        return {'train': torch.tensor(train_idx, dtype = torch.long), 'valid': torch.tensor(valid_idx, dtype = torch.long), 'test': torch.tensor(test_idx, dtype = torch.long)}

        

    def get(self, idx):
        data = Data()
        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(slices[idx],
                                                    slices[idx + 1])
            data[key] = item[s]
        return data

    
    @property
    def raw_file_names(self):
        file_name_list = [ str(f) for f in Path(self.root).rglob(f"{self.dataname}.pt") ]
        # assert len(file_name_list) == 1     # currently assume we have a
        # # single raw file
        return file_name_list
    
    @property
    def processed_file_names(self):
        return [f'geometric_data_{self.dataname}_processed.pt'] #[f'geometric_data_processed_{self.dataname}_train.pt', f'geometric_data_processed_{self.dataname}_test.pt']
    
    def download(self):
        pass
    
    def process(self):
        data_list = []
        graph_labels = []
        graph_ids = []                                     #target_index_to_word
       
        #for file in tqdm.tqdm( self.raw_file_names ):
       # print(self.raw_file_names)
        raw_data = torch.load(os.path.join( self.raw_file_names[0] ))
        data_list, graph_labels, graph_ids = raw_data[0], raw_data[1], raw_data[2]

        indexes = np.arange(len(graph_labels ))
        np.random.shuffle(indexes)
        test_size = int(0.2*len(graph_labels))
        val_size = int ( 0.2*( len(graph_labels) - test_size ) )
        train_size = len(graph_labels) - val_size - test_size
        text_idx = indexes[:test_size]
        val_idx = indexes[test_size: test_size+val_size]
        train_idx = indexes[test_size+val_size:]
        import pickle
        print(f"{len(train_idx)}, {len(val_idx)}, {len(text_idx)}")
        pickle.dump(text_idx, open( os.path.join(self.root, "test.csv.npy") , "wb"))
        pickle.dump(val_idx, open( os.path.join(self.root, "valid.csv.npy") , "wb"))
        pickle.dump(train_idx, open( os.path.join(self.root, "train.csv.npy") , "wb"))

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
     
        

class UnsupervisedDataset(InMemoryDataset):
    def __init__(self, root, dataname,num_partition=10, load_index=0, check_all=False,transform=None, pre_transform=None, pre_filter=None):
        self.root = root
        #print(dataname)
        self.dataname = dataname
        self.num_partition = num_partition
        self.check_all = check_all
        self.load_index = load_index
        super(UnsupervisedDataset, self).__init__(root=root, transform=transform,  pre_transform=pre_transform, pre_filter=pre_filter)
        
        self.data, self.slices = torch.load(self.processed_paths[load_index] if len(self.processed_paths)>1 else self.processed_paths[0])
    
    def get_partitions(self):
        if self.check_all:
            chunks = int(len(  self.raw_file_names )/self.num_partition)
            partitions = [self.raw_file_names[i:i+chunks] for i in range(0, len(self.raw_file_names), chunks)]
            return [i for i in range(len(partitions)) ]
        else:
            return [ 0 ]
    
    def get(self, idx):
        data = Data()
        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(slices[idx],
                                                    slices[idx + 1])
            data[key] = item[s]
        return data 
    
    @property
    def raw_file_names(self):
        #print(f"====================={self.root}")
        file_name_list = [ str(f) for f in Path(self.root).rglob(f"{self.dataname}.pt") ]
        # assert len(file_name_list) == 1     # currently assume we have a
        # # single raw file
        return file_name_list
    
    @property
    def processed_file_names(self):
        if self.check_all:
            chunks = int(len(  self.raw_file_names )/self.num_partition)
            partitions = [self.raw_file_names[i:i+chunks] for i in range(0, len(self.raw_file_names), chunks)]
            return [f'geometric_data_{self.dataname}_processed_partition_{i}.pt' for i in range(len(partitions))]
        else:
            return [f'geometric_data_{self.dataname}_processed_partition_{self.load_index}.pt' ]
    
    def download(self):
        pass
    
    def task(self, data):
        index = data[0]
        pieces = data[1]
        data_list = []
        graph_labels = []
        graph_ids = []
        for file in tqdm.tqdm(pieces):
                raw_data = torch.load(os.path.join( file ))
                raw_data_list, raw_graph_labels, raw_graph_id = raw_data[0], raw_data[1], raw_data[2]
                data_list = data_list + raw_data_list
                graph_labels = graph_labels + raw_graph_labels
                graph_ids = graph_ids + raw_graph_id
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[index])

    def process(self):
        random.shuffle( self.raw_file_names )
        chunks = int(len(  self.raw_file_names )/self.num_partition)
        partitions = [self.raw_file_names[i:i+chunks] for i in range(0, len(self.raw_file_names), chunks)]
        params = []
        for i, pieces in tqdm.tqdm( enumerate( partitions ) ):
            params.append([i, pieces])
            # data_list = []
            # graph_labels = []
            # graph_ids = []
            # for file in pieces:
            #     raw_data = torch.load(os.path.join( file ))
            #     raw_data_list, raw_graph_labels, raw_graph_id = raw_data[0], raw_data[1], raw_data[2]
            #     data_list = data_list + raw_data_list
            #     graph_labels = graph_labels + raw_graph_labels
            #     graph_ids = graph_ids + raw_graph_id
            # data, slices = self.collate(data_list)
            # torch.save((data, slices), self.processed_paths[i])
        print("Process")
        pool = multiprocessing.Pool(processes=10)
        pool.map(self.task, params)



class ProbingDataset(InMemoryDataset):
    def __init__(self, root, dataname,transform=None, pre_transform=None, pre_filter=None):
        self.root = root
        #print(dataname)
        self.dataname = dataname
        super(ProbingDataset, self).__init__(root=root, transform=transform,  pre_transform=pre_transform, pre_filter=pre_filter)
        self.data, self.slices = torch.load( self.processed_paths[0] )
    
    def split(self, reshuffle=False):
        import pickle
        if reshuffle:
            train_idx =  pickle.load( open( os.path.join(self.root, "train.csv.npy") , "rb") )
            test_idx = pickle.load( open( os.path.join(self.root, "test.csv.npy") , "rb") )
            total_size = test_idx.size + train_idx.size
            total_size = len(self)
            indexes = np.arange(total_size)
            np.random.shuffle(indexes)
            test_size = int(0.2*total_size)
            train_size = total_size  - test_size
            test_idx = indexes[:test_size]
            train_idx = indexes[test_size:]
            pickle.dump(test_idx, open( os.path.join(self.root, "test.csv.npy") , "wb"))
            pickle.dump(train_idx, open( os.path.join(self.root, "train.csv.npy") , "wb"))
        else:
            print("Load from the previsou splitting")
            train_idx =  pickle.load( open( os.path.join(self.root, "train.csv.npy") , "rb") )#pd.read_csv(osp.join(path, 'train.csv'),  header = None).values.T[0]
            #valid_idx = pickle.load( open( os.path.join(self.root, "valid.csv.npy") , "rb") )#pd.read_csv(osp.join(path, 'valid.csv'),  header = None).values.T[0]
            test_idx = pickle.load( open( os.path.join(self.root, "test.csv.npy") , "rb") )#pd.read_csv(osp.join(path, 'test.csv'),  header = None).values.T[0]
        
        
        return {'train': torch.tensor(train_idx, dtype = torch.long), 'test': torch.tensor(test_idx, dtype = torch.long)}
    
    def get(self, idx):
        data = Data()
        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(slices[idx],
                                                    slices[idx + 1])
            data[key] = item[s]
        return data 
    
    @property
    def raw_file_names(self):
        #print(f"====================={self.root}")
        file_name_list = [ str(f) for f in Path(self.root).rglob(f"{self.dataname}.pt") ]
        # assert len(file_name_list) == 1     # currently assume we have a
        # # single raw file
        return file_name_list
    
    @property
    def processed_file_names(self):
        return [f'geometric_data_{self.dataname}_processed_partition.pt' ]
    
    def download(self):
        pass
    
    def process(self):
        data_list = []
        graph_labels = []
        graph_ids = []                                     
        
        counter = collections.defaultdict(int)
        for file in tqdm.tqdm( self.raw_file_names ):
            raw_data = torch.load(os.path.join( file ))
            raw_data_list, raw_graph_labels, raw_graph_id = raw_data[0], raw_data[1], raw_data[2]
            data_list = data_list + raw_data_list
            graph_labels = graph_labels + raw_graph_labels
            graph_ids = graph_ids + raw_graph_id

        indexes = np.arange(len(graph_labels ))
        np.random.shuffle(indexes)
        test_size = int(0.3*len(graph_labels))
        train_size = len(graph_labels)  - test_size
        text_idx = indexes[:test_size]
        train_idx = indexes[test_size:]
        import pickle
        print(f"{len(train_idx)}, {len(text_idx)}")
        pickle.dump(text_idx, open( os.path.join(self.root, "test.csv.npy") , "wb"))
        pickle.dump(train_idx, open( os.path.join(self.root, "train.csv.npy") , "wb"))   
        graph_labels_series = pd.Series( graph_labels )
        graph_ids_series = pd.Series( graph_ids )
        graph_labels_series.to_csv(os.path.join(self.processed_dir,
                                               'graph_labels.csv'), index=False,
                                  header=False)
        graph_ids_series.to_csv(os.path.join(self.processed_dir,
                                               'graph_ids.csv'), index=False,
                                  header=False)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print(f"Dict Size {len(counter)}")
        json.dump( counter, open(os.path.join(self.processed_dir,
                                               'counter.json'), "w"))

    

class ProbingCorruptedDataset(InMemoryDataset):
    def __init__(self, root, dataname,transform=None, pre_transform=None, pre_filter=None):
        self.root = root
        #print(dataname)
        self.dataname = dataname
        super(ProbingCorruptedDataset, self).__init__(root=root, transform=transform,  pre_transform=pre_transform, pre_filter=pre_filter)
        self.data, self.slices = torch.load( self.processed_paths[0] )
    
    def split(self, reshuffle=False):
        import pickle
        if reshuffle:
            train_idx =  pickle.load( open( os.path.join(self.root, "train.csv.npy") , "rb") )
            test_idx = pickle.load( open( os.path.join(self.root, "test.csv.npy") , "rb") )
            total_size = test_idx.size + train_idx.size
            total_size = len(self)
            indexes = np.arange(total_size)
            np.random.shuffle(indexes)
            test_size = int(0.2*total_size)
            train_size = total_size  - test_size
            test_idx = indexes[:test_size]
            train_idx = indexes[test_size:]
            pickle.dump(test_idx, open( os.path.join(self.root, "test.csv.npy") , "wb"))
            pickle.dump(train_idx, open( os.path.join(self.root, "train.csv.npy") , "wb"))
        else:
            print("Load from the previsou splitting")
            train_idx =  pickle.load( open( os.path.join(self.root, "train.csv.npy") , "rb") )#pd.read_csv(osp.join(path, 'train.csv'),  header = None).values.T[0]
            #valid_idx = pickle.load( open( os.path.join(self.root, "valid.csv.npy") , "rb") )#pd.read_csv(osp.join(path, 'valid.csv'),  header = None).values.T[0]
            test_idx = pickle.load( open( os.path.join(self.root, "test.csv.npy") , "rb") )#pd.read_csv(osp.join(path, 'test.csv'),  header = None).values.T[0]
        
        print(f"Train {train_idx.shape} Test {test_idx.shape}")
        return {'train': torch.tensor(train_idx, dtype = torch.long), 'test': torch.tensor(test_idx, dtype = torch.long)}
    
    def get(self, idx):
        data = Data()
        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(slices[idx],
                                                    slices[idx + 1])
            data[key] = item[s]
        return data 
    
    @property
    def raw_file_names(self):
        #print(f"====================={self.root}")
        file_name_list = [ f"{self.root}/raw/{self.dataname}.pt", f"{self.root}/raw/d{self.dataname}.pt" ]
       
        # assert len(file_name_list) == 1     # currently assume we have a
        # # single raw file
        return file_name_list
    
    @property
    def processed_file_names(self):
        return [f'geometric_data_{self.dataname}_processed_corrupted.pt' ]
    
    def download(self):
        pass
    
    def process(self):
        data_list = []
        graph_labels = []
        graph_ids = []                                     
        
        counter = collections.defaultdict(int)
        # for file in tqdm.tqdm( self.raw_file_names ):
        raw_data = torch.load(os.path.join( self.raw_file_names[0] ))
        raw_data_list, raw_graph_labels, raw_graph_id = raw_data[0], raw_data[1], raw_data[2]
        for d in raw_data_list:
            d.y1 = 1
        
        data_list = data_list + raw_data_list
        graph_labels = graph_labels + raw_graph_labels
        graph_ids = graph_ids + raw_graph_id

        raw_data = torch.load(os.path.join( self.raw_file_names[1] ))
        raw_data_list, raw_graph_labels, raw_graph_id = raw_data[0], raw_data[1], raw_data[2]
        for d in raw_data_list:
            d.y1 = 0
            d.label = d.label + 10
        raw_graph_labels = [ i + 10  for i in raw_graph_labels]
        data_list = data_list + raw_data_list
        graph_labels = graph_labels + raw_graph_labels
        graph_ids = graph_ids + raw_graph_id

        indexes = np.arange(len(graph_labels ))
        np.random.shuffle(indexes)
        test_size = int(0.2*len(graph_labels))
        train_size = len(graph_labels)  - test_size
        text_idx = indexes[:test_size]
        train_idx = indexes[test_size:]
        import pickle
        print(f"{len(train_idx)}, {len(text_idx)}")
        pickle.dump(text_idx, open( os.path.join(self.root, "test.csv.npy") , "wb"))
        pickle.dump(train_idx, open( os.path.join(self.root, "train.csv.npy") , "wb"))   
        graph_labels_series = pd.Series( graph_labels )
        graph_ids_series = pd.Series( graph_ids )
        graph_labels_series.to_csv(os.path.join(self.processed_dir,
                                               'graph_labels.csv'), index=False,
                                  header=False)
        graph_ids_series.to_csv(os.path.join(self.processed_dir,
                                               'graph_ids.csv'), index=False,
                                  header=False)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print(f"Dict Size {len(counter)}")
        json.dump( counter, open(os.path.join(self.processed_dir,
                                               'counter.json'), "w"))


   

class DeadlockDataset(InMemoryDataset):
    def __init__(self, root, dataname,transform=None, pre_transform=None, pre_filter=None):
        self.root = root
        #print(dataname)
        self.dataname = dataname
        super(DeadlockDataset, self).__init__(root=root, transform=transform,  pre_transform=pre_transform, pre_filter=pre_filter)
        self.data, self.slices = torch.load( self.processed_paths[0] )
    
    def split(self, reshuffle=False):
        import pickle
        if reshuffle:
            train_idx =  pickle.load( open( os.path.join(self.root, "train.csv.npy") , "rb") )
            test_idx = pickle.load( open( os.path.join(self.root, "test.csv.npy") , "rb") )
            total_size = test_idx.size + train_idx.size
            total_size = len(self)
            indexes = np.arange(total_size)
            np.random.shuffle(indexes)
            test_size = int(0.3*total_size)
            train_size = total_size  - test_size
            test_idx = indexes[:test_size]
            train_idx = indexes[test_size:]
            pickle.dump(test_idx, open( os.path.join(self.root, "test.csv.npy_1") , "wb"))
            pickle.dump(train_idx, open( os.path.join(self.root, "train.csv.npy_1") , "wb"))
        else:
            print("Load from the previsou splitting")
            train_idx =  pickle.load( open( os.path.join(self.root, "train.csv.npy") , "rb") )#pd.read_csv(osp.join(path, 'train.csv'),  header = None).values.T[0]
            #valid_idx = pickle.load( open( os.path.join(self.root, "valid.csv.npy") , "rb") )#pd.read_csv(osp.join(path, 'valid.csv'),  header = None).values.T[0]
            test_idx = pickle.load( open( os.path.join(self.root, "test.csv.npy") , "rb") )#pd.read_csv(osp.join(path, 'test.csv'),  header = None).values.T[0]
        
        print(f"Train {train_idx.shape} Test {test_idx.shape}")
        return {'train': torch.tensor(train_idx, dtype = torch.long), 'test': torch.tensor(test_idx, dtype = torch.long)}
    
    def get(self, idx):
        data = Data()
        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(slices[idx],
                                                    slices[idx + 1])
            data[key] = item[s]
        return data 
    
    @property
    def raw_file_names(self):
        #print(f"====================={self.root}")
        file_name_list = [ f"{self.root}/raw/{self.dataname}.pt" ]
       
        # assert len(file_name_list) == 1     # currently assume we have a
        # # single raw file
        return file_name_list
    
    @property
    def processed_file_names(self):
        return [f'geometric_data_{self.dataname}_processed_corrupted.pt' ]
    
    def download(self):
        pass
    
    def process(self):
                                    
        
        counter = collections.defaultdict(int)
        # for file in tqdm.tqdm( self.raw_file_names ):
        raw_data = torch.load(os.path.join( self.raw_file_names[0] ))
        data_list, _, _ = raw_data[0], raw_data[1], raw_data[2]
        dead_num = 0
        #dead_data = []
        #non_dead_data = []
        for d in data_list:
            if d.y == 0:
                d.y1 = 1
                dead_num += 1
                #dead_data.append(d)
            else:
                d.y1 = 0
                #non_dead_data.append(d)
        #random.shuffle(dead_data)
        #data_list =  dead_data[: len(non_dead_data) ] + non_dead_data
        print(f"Dead Number {dead_num}")      
        indexes = np.arange(len(data_list ))
        np.random.shuffle(indexes)
        test_size = int(0.5*len(data_list))
        train_size = len(data_list)  - test_size
        text_idx = indexes[:test_size]
        train_idx = indexes[test_size:]
        import pickle
        print(f"{len(train_idx)}, {len(text_idx)}")
        pickle.dump(text_idx, open( os.path.join(self.root, "test.csv.npy") , "wb"))
        pickle.dump(train_idx, open( os.path.join(self.root, "train.csv.npy") , "wb"))   
        # graph_labels_series = pd.Series( graph_labels )
        # graph_ids_series = pd.Series( graph_ids )
        # graph_labels_series.to_csv(os.path.join(self.processed_dir,
        #                                        'graph_labels.csv'), index=False,
        #                           header=False)
        # graph_ids_series.to_csv(os.path.join(self.processed_dir,
        #                                        'graph_ids.csv'), index=False,
        #                           header=False)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print(f"Dict Size {len(counter)}")
        json.dump( counter, open(os.path.join(self.processed_dir,
                                               'counter.json'), "w"))
