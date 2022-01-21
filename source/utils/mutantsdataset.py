import collections
from itertools import repeat
from networkx.algorithms.core import k_core
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
import torch
import os
import tqdm 
import numpy as np
import pickle
from itertools import compress
from utils.tools import  inverse_eage
from collections import Counter
import json
class MutantsDataset(InMemoryDataset):
    def __init__(self, root, dataname, project="", transform=None, pre_transform=None, pre_filter=None):
        self.root = root
        self.dataname = dataname
        self.project = project
        super(MutantsDataset, self).__init__(root=root, transform=transform,  pre_transform=pre_transform, pre_filter=pre_filter)
        self.pair_data = torch.load(self.processed_paths[0])
        [ self.pairgraph_labels, self.mutants_splitting ] =  torch.load(self.processed_paths[1])

     

    def split(self, reshuffle=False, binary=False, hits=1):
        import pickle
        if reshuffle or ( not os.path.isfile( os.path.join(self.root,  "data", "train.csv.npy") )):
            data = []
            label=[]
            mid = []
            for l in self.pair_data:
                data.extend(self.pair_data[l])
                mid.extend(self.mutants_splitting[l])
                if int(l) == 0:
                    label.extend([0]*len(self.pair_data[l]))
                    print("Label 0 {len(self.pair_data[l])}")
                else:
                    if binary:
                        label.extend([1]*len(self.pair_data[l]))
                        print("Label 1 {len(self.pair_data[l])}")
                    else:
                        label.extend([int(l)]*len(self.pair_data[l]))
                        print("Label {l} {len(self.pair_data[l])}")
            
            #self.data, self.y, self.mids = balanced_subsample(data, label, mid)
            self.data_size = len(data)
            self.data = data
            self.y = label
            statistic = Counter(label)
            print(statistic)
            self.mids = mid
            indexes = np.arange( len(self.data) )
            np.random.shuffle(indexes)
            test_size = int(0.1*self.data_size)
            val_size =  int(0.1*self.data_size)
            train_size = self.data_size - val_size - test_size
            test_idx = indexes[:test_size]
            valid_idx = indexes[test_size: test_size+val_size]
            train_idx = indexes[test_size+val_size:]
            train_y = [ self.y[i] for i in train_idx ]
            #train_idx = balanced_oversample(train_idx, train_y) # balanced_subsample(data, label, mid)
            pickle.dump(test_idx, open( os.path.join(self.root, "data","test.csv.npy") , "wb"))
            pickle.dump(valid_idx, open( os.path.join(self.root, "data","valid.csv.npy") , "wb"))
            pickle.dump(train_idx, open( os.path.join(self.root, "data", "train.csv.npy") , "wb"))
            pickle.dump(self.mids, open( os.path.join(self.root,"data", "mids.csv.npy") , "wb"))
            pickle.dump([self.data, self.y, self.mids], open( os.path.join(self.root,"data", "sampled_mids.csv.npy") , "wb"))
            json.dump(statistic, open( os.path.join(self.root,"data", "statistic.json") , "w"))
        else:
            print("Load from the previsou splitting")
            [self.data, self.y, self.mids] = pickle.load( open( os.path.join(self.root,"data", "sampled_mids.csv.npy") , "rb") )
            mutants_type = json.load(open(os.path.join(self.root,"data", "mutants_type.json")  ))
            print(os.path.join(self.root,"data", "mutants_type.json") )
            operand_type = json.load(open(os.path.join(self.root, "data", "operand_id_mapping.json")))
            operand_type = {k:int(v) for k,v in operand_type.items() }
            typeid={"STD":1, "LVR":2, "ORU":3, "ROR":4, "SOR":5, "LOR":6, "COR":7, "AOR":8}  
            type_one = [ operand_type[mutants_type[mid][2]] for mid in self.mids ]
            type_two = [ operand_type[mutants_type[mid][3]] for mid in self.mids ]
            for i,d in enumerate(self.data):
                #print(f"check {d.type}, {mutants_type[self.mids[i]][0]}")
                #print(typeid[mutants_type[self.mids[i]][0]])
                d.operand1 = type_one[i]
                d.operand2 = type_two[i]
                assert d.type == typeid[mutants_type[self.mids[i]][0]]

            train_idx =  pickle.load( open( os.path.join(self.root,"data", "train.csv.npy") , "rb") )#pd.read_csv(osp.join(path, 'train.csv'),  header = None).values.T[0]
            valid_idx = pickle.load( open( os.path.join(self.root,"data", "valid.csv.npy") , "rb") )#pd.read_csv(osp.join(path, 'valid.csv'),  header = None).values.T[0]
            test_idx = pickle.load( open( os.path.join(self.root,"data", "test.csv.npy") , "rb") )#pd.read_csv(osp.join(path, 'test.csv'),  header = None).values.T[0]
            #train_idx = balanced_oversample(train_idx,  self.y)
        mutant_hits_info = json.load(open(os.path.join(self.root,"data", "mutant_hits_info.json")  ))
        hist_number =  [ mutant_hits_info[mid] for mid in self.mids ]
        labels = [ d.y.item() for d in self.data ]
        fileter_train_idx = []
        fileter_valid_idx = []
        fileter_test_idx = []
        live_hist = []
        killed_hist = []
        for i in train_idx:
            if labels[i] == 0:
                live_hist.append(hist_number[i])
            if labels[i] == 0 and hist_number[i] <= hits:
                continue
            fileter_train_idx.append(i)
            killed_hist.append(hist_number[i])

        for i in valid_idx:
            if labels[i] == 0:
                live_hist.append(hist_number[i])
            if labels[i] == 0 and hist_number[i] <= hits:
                continue
            fileter_valid_idx.append(i)
            killed_hist.append(hist_number[i])
        
        for i in test_idx:
            if labels[i] == 0:
                live_hist.append(hist_number[i])
            if labels[i] == 0 and hist_number[i] <= hits:
                continue
            fileter_test_idx.append(i)
            killed_hist.append(hist_number[i])
        json.dump(live_hist, open( os.path.join(self.root,"data", "live_hist.json") , "w"))
        json.dump(hist_number, open( os.path.join(self.root,"data", "hist_number.json") , "w"))
        json.dump(killed_hist, open( os.path.join(self.root,"data", "killed_hist.json") , "w"))
        print(f"Train Data Size {len(train_idx)}, Valid Data Size {len(valid_idx)}, Test Data Size {len(test_idx)}, Num {len(self.mids)}")
        statistic = Counter(labels)
        print(statistic)
        train_idx = fileter_train_idx
        valid_idx = fileter_valid_idx
        test_idx = fileter_test_idx
        print(f"Train Data Size {len(train_idx)}, Valid Data Size {len(valid_idx)}, Test Data Size {len(test_idx)}, Num {len(self.mids)}")
        return {'train': torch.tensor(train_idx, dtype = torch.long), 'valid': torch.tensor(valid_idx, dtype = torch.long), 'test': torch.tensor(test_idx, dtype = torch.long)}
       # return {'train': torch.tensor(train_idx, dtype = torch.long),  'test': torch.tensor(test_idx, dtype = torch.long)}

           
    
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
    def raw_dir(self):
        return os.path.join(self.root)
    @property
    def raw_paths(self):
        r"""The filepaths to find in order to skip the download."""
        
        return self.raw_file_names[0] + self.raw_file_names[1]

    @property
    def raw_file_names(self):
        mutant_file_name_list = []
        original_file_name_list = []
        for pname in os.listdir(self.root):
            if os.path.isfile(os.path.join( self.root ,f"{pname}" )) or pname == "processed" or pname=="data":
                continue
            if self.project.strip() and  self.project not in pname:
                continue
            mutant_file_name_list.append(  os.path.join( self.root ,f"{pname}", "raw","graph", f"{self.dataname}.pt")   )
            original_file_name_list.append(  os.path.join( self.root ,f"{pname}", "raw","original", "graph", f"{self.dataname}.pt")   )
        
        return mutant_file_name_list, original_file_name_list
    
    @property
    def processed_file_names(self):
        return [ f'geometric_data_processed_{self.dataname}.pt', f"data_info.pt"]
    
    def download(self):
        pass
    
    def process(self):     
        import json  
        mutanttyper=json.load( open( os.path.join(os.path.join(self.root, "mutant_operator.json")) ) )
        typeid={"STD":1, "LVR":2, "ORU":3, "ROR":4, "SOR":5, "LOR":6, "COR":7, "AOR":8}       
        pairgraph_labels = []
        pair_data_list = collections.defaultdict(list)
        mutant_file_name_list, original_file_name_list = self.raw_file_names
        mutants_splitting = collections.defaultdict(list)
        for file_id in tqdm.tqdm( range(len(mutant_file_name_list)) ):
            mfile = mutant_file_name_list[file_id]
            pname = mfile.split("/")[-4]
            ofile = original_file_name_list[file_id]
            mutant_data = torch.load(os.path.join( mfile ))
            mutant_data_list, graph_labels, graph_ids = mutant_data[0], mutant_data[1], mutant_data[2]
            mutant_data_list = [ inverse_eage(d) for d in mutant_data_list ]
            org_data = torch.load(os.path.join( ofile ))
            org_data_list, _, org_data_id = org_data[0], org_data[1], org_data[2]
            org_data_list = [ inverse_eage(d) for d in org_data_list ]
            for k, orgid in enumerate(org_data_id):
                indx = [ id ==orgid for id in graph_ids ]
                graphs = list(compress(mutant_data_list, indx )) #mutant_data_list[graph_ids==orgid]
                mutant_data_labels = list(compress(graph_labels, indx ))  #graph_labels[graph_ids==orgid]
                orggraph = org_data_list[k]
                for i in range(len(graphs)):
                   # print(type(graphs[i]))
                   # print(x_s.size())
                    id=graphs[i].mutantID
                    pair_data_list[mutant_data_labels[i]].append(PairData(graphs[i].edge_index,  graphs[i].x, graphs[i].edge_attr, graphs[i].ins_length, 
                                                    orggraph.edge_index,  orggraph.x, orggraph.edge_attr, orggraph.ins_length, 
                                                    torch.tensor(mutant_data_labels[i]), torch.tensor([typeid[mutanttyper[f"{pname}_{id}"]]]) ))
                   # pairgraph_labels.append( int(mutant_data_labels[i]) if mutant_data_labels[i] !=0 else 0)
                   # pairgraph_labels.append( 1 if mutant_data_labels[i] !=0 else 0)
                    
                    mutants_splitting[mutant_data_labels[i]].append( f"{pname}_{id}"  )
                    
                   # pairgraph_ids.append()
                 
        #pair_data_list,pairgraph_labels = balanced_subsample(pair_data_list, pairgraph_labels)
        self.data_size = len(pair_data_list)                        
       # mutant_data, mutant_slices = self.collate(pair_data_list)
        torch.save(pair_data_list, self.processed_paths[0])
        torch.save( [ pairgraph_labels, mutants_splitting ], self.processed_paths[1] )

def balanced_subsample(x,y,mid_list,subsample_size=1.0):

    class_xs = []
    min_elems = None

    for yi in np.unique(y):
        idx =  [ id ==yi for id in y ]
        elems = list(compress(x, idx )) 
        mid =  list(compress(mid_list, idx )) 
        class_xs.append((yi, elems, mid))
        print(f"label {yi}, Number {len(elems)}")
        if min_elems == None or len(elems) < min_elems:
            min_elems = len(elems)

    use_elems = min_elems
    if subsample_size < 1:
        use_elems = int(min_elems*subsample_size)

    xs = []
    ys = []
    mids= []
    for ci,this_xs, this_mids in class_xs:
        index = [i for i in range(len(this_xs))]
        if len(this_xs) > use_elems:
            np.random.shuffle(index)

        x_ = [ this_xs[i] for i in index[:use_elems] ]  #this_xs[:use_elems]
        mid_ = [ this_mids[i] for i in index[:use_elems] ]
        y_ = np.empty(use_elems)
        y_.fill(ci)

        xs.extend(x_)
        ys.extend(y_.tolist())
        mids.extend(mid_)

   # xs = np.concatenate(xs)
   # ys = np.concatenate(ys)

    return xs       

import random
def balanced_oversample(x, y):
    class_xs = []
    max_elems=None
    for yi in np.unique(y):
        idx =  [ id ==yi for id in y ]
        elems = list(compress(x, idx )) 
        class_xs.append((yi, elems))
        print(f"label {yi}, Number {len(elems)}")
        if max_elems == None or len(elems) > max_elems:
            max_elems = len(elems)

    use_elems = max_elems
    # if subsample_size < 1:
    #     use_elems = int(min_elems*subsample_size)

    xs = []
    ys = []

    for ci,this_xs in class_xs:
        index = [i for i in range(len(this_xs))]
        if use_elems > len(this_xs):
            index = random.choices(index, k=use_elems)

        x_ = [ this_xs[i] for i in index ]  #this_xs[:use_elems]
        y_ = np.empty(use_elems)
        y_.fill(ci)
        
        xs.extend(x_)
        ys.extend(y_.tolist())
  
    return xs      

class PairData(Data):
    
    def __init__(self, edge_index_s=None, x_s=None, edge_attr_s=None, ins_length_s=None, edge_index_t=None, x_t=None,  edge_attr_t=None, ins_length_t=None,y=None, type=None, operand1=None, operand2=None):
        super(PairData, self).__init__()
        self.edge_index_s = edge_index_s
        self.x_s = x_s
        self.edge_attr_s = edge_attr_s
        self.ins_length_s = ins_length_s
            
        self.edge_index_t = edge_index_t
        self.edge_attr_t = edge_attr_t
        self.x_t = x_t
        self.ins_length_t = ins_length_t
        self.y = y
        self.type=type
        self.operand1=operand1
        self.operand2=operand2
        #print(x_s)
        if x_s is None:
             print("Debug")
        # print(self.x_s.size(0))
        # print(self.x_t.size(0))

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_s':
          #  print(self.x_s.size(0))
            return self.x_s.size(0)
        if key == 'edge_index_t':
           # print(self.x_s.size(0))
            return self.x_t.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)



class MutantsSingleDataset(InMemoryDataset):
    def __init__(self, root, dataname, transform=None, pre_transform=None, pre_filter=None, mutants=False):
        self.root = root
        self.dataname = dataname
        self.classes = 0
        super(MutantsSingleDataset, self).__init__(root=root, transform=transform,  pre_transform=pre_transform, pre_filter=pre_filter)
        if mutants:
            print("Load Mutants Data")
            self.data, self.slices = torch.load( self.processed_paths[0] )
            self.classes = 4
        else:
            print("Load Original Data")
            self.data, self.slices = torch.load( self.processed_paths[1] )
            self.classes = 2
        self.dataInfo = torch.load( self.processed_paths[2] )
      
    
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
        file_name_list = collections.defaultdict(list)
        p_data = os.path.join( self.raw_dir , "graph", f"{self.dataname}.pt")
        op_data = os.path.join( self.raw_dir , "original", "graph", f"{self.dataname}.pt")
        file_name_list = [ p_data, op_data ]
        return file_name_list
    
    @property
    def processed_file_names(self):
        return [ f'mutant_geometric_data_processed_{self.dataname}.pt',f'original_geometric_data_processed_{self.dataname}.pt', \
            f'mutant_2_org_graph_mappting_{self.dataname}.pt' ]
    
    def download(self):
        pass

    
    def process(self):
        #for pid in tqdm.tqdm( self.raw_file_names ):
        files = self.raw_file_names
        print("====================== ===========================")
        print(files)
        mutant_data = torch.load(os.path.join( files[0] ))
        mutant_data_list, mutant_graph_labels, mutant2_orggraph_id = mutant_data[0],  torch.tensor(mutant_data[1]),  torch.tensor(mutant_data[2])
        original_data = torch.load(os.path.join( files[1] ))
        original_data_list, original_graph_labels, original_graph_id_list = original_data[0],  torch.tensor(original_data[1]), torch.tensor(original_data[2])
                            
        mutant_data, mutant_slices = self.collate(mutant_data_list)
        torch.save((mutant_data, mutant_slices), self.processed_paths[0])
        org_data, org_slices = self.collate(original_data_list)
        torch.save((org_data, org_slices), self.processed_paths[1])
        torch.save( [ mutant2_orggraph_id, original_graph_id_list, mutant_graph_labels, original_graph_labels], self.processed_paths[2] )
  