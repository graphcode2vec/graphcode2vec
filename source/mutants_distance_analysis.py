from math import isnan
from scipy import stats
import glob
import os
import torch
from numpy.linalg import inv
from scipy.spatial import distance
import numpy as np
import collections
import numpy as np
import matplotlib.pyplot as plt
from utils.tools import precision

killlabel = {"LIVE":0, "FAULT":1, "EXC":2, "TIME":3, "FAIL":4,"Original":5}
killlabel_reverse = {0:"LIVE", 1:"FAULT", 2:"EXC", 3:"TIME", 4:"FAIL",5:"Original"}
no_pass_label = {0:"LIVE", 1:"NoPass", 2:"NoPass", 3:"NoPass", 4:"NoPass",5:"Original"}

# check if there is bigger difference between living mutants and other types
def getMeanMedian(data):
    res_avg = {}
    res_median = {}
    for k, v in data.items():
        res_avg[k] = np.mean(v)
        res_median[k] = np.median(v)
    return res_avg, res_median



def plot_histogram(data, Project):
    import matplotlib.pyplot as plt
    # the histogram of the data
    for k, v in data.items():
        n, bins, patches = plt.hist(v, 100, density=True, alpha=0.5, label=k )
        #print(f"Number {k} = {len(v)}")
    # mu = np.mean( data )
    # sigma = np.std( data )
    plt.xlabel('Mahalanobis')
    plt.ylabel('Number')
    plt.title(f'{Project} Histogram of Mahalanobis Distance')
    # plt.text(60, .025, rf'$\mu={mu}, $\sigma={sigma}$')
    # plt.xlim(40, 160)
    # plt.ylim(0, 0.03)
    plt.legend(loc='upper right')
    plt.show()
    plt.grid(True)
   
   
def compute_mahalanobis(data_dir):
    project_name_list=["Chart", "Cli", "Closure", "Codec", "Collections", "Compress",
            "Csv", "Gson", "JacksonCore", "JacksonDatabind", "JacksonXml", "Jsoup", "JxPath", "Lang", "Math", "Mockito", "Time"]
    
    distance_data = collections.defaultdict( )
    for project_name in project_name_list:
        s = []
        for vfile in glob.glob( os.path.join(data_dir, f"{project_name}_*_*", "vector.pt") ):
            data = torch.load( vfile )
            graph_mutants = data["mutant_vector"][0]
            s.append(graph_mutants)
            original_graph = data["original_vector"][0]
            s.append(original_graph)
        if len(s) == 0:
            continue   
        S = torch.cat( s, dim=0 ).numpy().transpose() # if the number of sample is less than 600, convariance_matrix may not semi-positive
        if S.shape[1] < 600:
            print(f"================== {project_name}, {S.shape}")
            continue
        scov = np.cov(S)
        inverse_convariance_matrix = inv(scov)   
        dlist_mahalanobis = []
        label_list = []
        pro_num = 0
        for vfile in glob.glob( os.path.join(data_dir, f"{project_name}_*_*", "vector.pt") ):
            #print("=================================================")
            data = torch.load( vfile )
            x = data["mutant_vector"][0]
            y = data["original_vector"][0]
            label = data["dataInfo"][2] # mutant label
            orgIDlist = data["dataset_original"]
            mutant2org_id=data["dataInfo"][0]
            tmp = {}
            for j, org_id in enumerate(orgIDlist):
                tmp[org_id.item()] = j
            # print(orgIDlist)
            # print(tmp)
            mutant2org_index = list(map(lambda x: tmp[x],  mutant2org_id.tolist() ))
            num=x.shape[0]
            for i in range(num):
                res = distance.mahalanobis(x[i].numpy(), y[mutant2org_index[i]].numpy(), inverse_convariance_matrix)
                dlist_mahalanobis.append(res)
            label_list.extend(label.tolist())    
            pro_num += 1
        dlist_mahalanobis = torch.tensor( dlist_mahalanobis )
        distance_data[project_name] = [ dlist_mahalanobis,  torch.tensor( label_list ), pro_num ]
    return distance_data

from collections import Counter

def boxplot(data):
    ltotal=[]
    for p in data:
        d = data[p]
        distance, labels, num = d[0], d[1], d[2]
        mu = []
        for l in [ 0, 1, 2, 3, 4 ]:
            index = labels == l 
            di = torch.mean( distance[index] )
            mu.append( di.item() )
        id = np.argsort(mu)
        print(f"{np.argsort(mu)}, {p}, {num}, {distance.shape[0]}")

    for p in data:
        d = data[p]
        distance, labels, num = d[0], d[1], d[2]
        ltotal.extend( labels.tolist() )
        index_live = labels == 0
        index_nopass = labels != 0
        mu = [ torch.mean( distance[index_live] ).item(), torch.mean( distance[index_nopass] ).item() ]
        print(f"{np.argsort(mu)}, {p}, {num}")
    
    print(Counter(ltotal))
        

            
                   



killlabel_reverse = {0:"LIVE", 1:"FAULT", 2:"EXC", 3:"TIME", 4:"FAIL",5:"Original"}

if __name__ == "__main__":
    data = compute_mahalanobis("results/mutants_vector/context/gat")
    boxplot(data)
   # boxplot(data)


