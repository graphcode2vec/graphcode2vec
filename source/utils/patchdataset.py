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
class PatchDataset(InMemoryDataset):
    def __init__(self, root, dataname, project="", transform=None, pre_transform=None, pre_filter=None):
        self.root = root
        self.dataname = dataname
        self.project = project
        super(PatchDataset, self).__init__(root=root, transform=transform,  pre_transform=pre_transform, pre_filter=pre_filter)
        print(self.processed_paths[0])
        self.pair_data = torch.load(self.processed_paths[0])
        self.patch_name_list =  torch.load(self.processed_paths[1])

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
        data_file =  os.path.join( self.root ,"raw", "patch_data.pt") 
        return data_file
    
    @property
    def processed_file_names(self):
        return [ f'geometric_data_processed_{self.dataname}.pt', f"data_info.pt"]
    
    def download(self):
        pass
    
    def process(self): 
        pair_data_list = []
        data_file = self.raw_file_names
      
        patch_data = torch.load( os.path.join( data_file ))
        counter  = 0
        import pandas as pd
        patch_list = pd.read_csv("dataset/patch_data/ASE20-902-Patches.csv", index_col="id").index.tolist()

        with open("tensor_size.txt", "w") as f:
            for dataname in [ self.dataname ]: #[ "DV_CFG", "DV_PDG" , "ORG_PDG" , "ORG_CFG"]:
                pair_data_list = []
                patch_name_list = []
                for data in patch_data: #  graph_type: [data_geometric, label , graph_ids ]
                    d_new = inverse_eage( data[0][dataname] )
                    d_old = inverse_eage( data[1][dataname] )
                    patch_name = data[2]
                   # print(d_new.y)
                    if d_new is None:
                        print(patch_name)
                    if d_new.x.shape[0] == 0 or d_old.x.shape[0] ==0:
                        counter = counter + 1
                        f.write(f"{patch_name} {d_new.x.shape} {d_old.x.shape} \n")
                        continue
                    pair_data_list.append( PairData(d_new.edge_index,  d_new.x, d_new.edge_attr, d_new.ins_length, 
                                                        d_old.edge_index,  d_old.x, d_old.edge_attr, d_old.ins_length, 
                                                        torch.tensor(d_new.y) ) )
                    patch_name_list.append( patch_name )
                   # print(f"{d_new.x.shape} {d_old.x.shape} ")
                    
                torch.save(pair_data_list, self.processed_paths[0])
                    

            self.data_size = len(pair_data_list)     
            torch.save( patch_name_list, self.processed_paths[1] )
            # for p in patch_list:
            #     if p+".patch.dir" not in patch_name_list:
            #         print(f"remove {p}")
        print(f"{self.data_size} =========================== \n ")

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
    
    def __init__(self, edge_index_s=None, x_s=None, edge_attr_s=None, ins_length_s=None, edge_index_t=None,\
         x_t=None,  edge_attr_t=None, ins_length_t=None,y=None):
        # print(edge_index_s.type())
        # print(x_s.type())
        # print(edge_attr_s.type())
        # print(ins_length_s.type())
        # print(y.type())
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

