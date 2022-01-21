import torch
import numpy as np
import sentencepiece as spm
from sklearn.metrics import (
    recall_score,
    precision_score,
    f1_score,
    accuracy_score,
    matthews_corrcoef,
)
from torch.functional import Tensor


def accuracy(y, pred):
    return accuracy_score(y, pred)


def f1(y, pred, average=None):
    if average:
        return f1_score(y, pred, average=average, zero_division=0)
    return f1_score(y, pred, average="weighted", zero_division=0)


def recall(y, pred, average=None):
    if average:
        return recall_score(y, pred, average=average, zero_division=0)
    return recall_score(y, pred, average="weighted", zero_division=0)


def precision(y, pred, average=None):
    if average:
        return precision_score(y, pred, average=average, zero_division=0)
    return precision_score(y, pred, average="weighted", zero_division=0)


def performance(y, pred, average=None):
    acc = accuracy(y, pred)
    f = f1(y, pred, average=average)
    re =recall(y, pred, average=average)
    pre = precision(y, pred, average=average)
    return acc, pre, re, f

import random
def seed_all_devices(seed=78934783):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(f"Use Device {device}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    return device

class Vocab(object):
    """vocabulary (terminal symbols or path names or label(method names))"""
    def __init__(self):
        self.stoi = {}
        self.itos = {}
        self.itosubtokens = {}
        self.freq = {}

    
    def append(self, name, index=None, subtokens=None):
        if name not in self.stoi:
            if index is None:
                index = len(self.stoi)
            if self.freq.get(index) is None:
                self.freq[index] = 0
            self.stoi[name] = index
            self.itos[index] = name
            if subtokens is not None:
                self.itosubtokens[index] = subtokens
            self.freq[index] += 1

    def get_freq_list(self):
        freq = self.freq
        freq_list = [0] * self.len()
        for i in range(self.len()):
            freq_list[i] = freq[i]
        return freq_list

    def len(self):
        return len(self.stoi)


class TokenIns:
    def __init__(
        self,
        word2vec_file="./tokens/emb.txt",
        tokenizer_file="./tokens/fun.model",
    ):
        # better way to make w2v not this global?
        # it is so messed up with the absolute path, preprocessing, and when creating node representations
        # word tokenizer
        # Load word tokenizer and word2vec model
        self.word_tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_file)
        self.parse_emb_txt(word2vec_file)
        #print("Load Subtokens Index of Statement")
        #self.statement_subtokens = pickle.load( open("./vocab/tokens/statement_subtokens.pt", "rb") )
        print("\nEmbedding length is %d.\n" % self.vector_size)
     
 

    def getVocabSize(self):
        return len(self.word2vec)

    def parse_emb_txt(self, embtxtfile):
        self.wordIndex = {"<pad>":0,"<unk>":1}
        self.word2vec = {}
        with open(embtxtfile,'r') as f:
            for line in f:
                #print(line)
                splitLines = line.split()
                if len(splitLines) == 2:
                    continue
                word = splitLines[0]
                wordEmbedding = np.array([float(value) for value in splitLines[1:]])
                self.vector_size = wordEmbedding.size
                self.word2vec[word] = wordEmbedding
                self.wordIndex[word] = len(self.wordIndex)
        self.word2vec.update( {"<pad>":np.zeros(self.vector_size), "<unk>":np.zeros(self.vector_size)} )

        print(len(self.word2vec)," words loaded!")

    def load_word2vec_embeddings(self):
        """
        Load pre-trained embeddings for words in the word map.
        :return: embeddings for words in the word map, embedding dim, vocab_size
        """

        # Create tensor to hold embeddings for words that are in-corpus
        embeddings = torch.FloatTensor(len(self.word2vec), self.vector_size)
        torch.nn.init.xavier_uniform_(embeddings)

        # Read embedding file
        # print("Loading embeddings...")
        for word in self.word2vec:
            embeddings[self.wordIndex[word]] = torch.FloatTensor(
                self.word2vec[word]
            )

        print("Done.\n Embedding vocabulary: %d.\n" % len(self.word2vec))
        return embeddings, self.vector_size, len(self.word2vec)


def subtoken_match(ground_truth_labels, actual_labels, label_vocab):
    """[summary]

    Args:
        expected_labels ([type]): [description]
        outputs ([type]): [description]
        K (int, optional): [description]. Defaults to 1.

    Returns:
        [type]: [description]
    """
    
    
    subtokens_match = 0.0
    subtokens_ground_truth_count = 0.0
    subtokens_actual_count = 0.0
    #print(len(label_vocab.itosubtokens))
    pre_res = []
    if isinstance(ground_truth_labels, Tensor ):
        ground_truth_labels = ground_truth_labels.tolist()
    if isinstance(actual_labels, Tensor ):
        actual_labels = actual_labels.tolist()    
        
    for ground_label, actual in zip(ground_truth_labels, actual_labels):
        ground_label_subtokens = label_vocab.itosubtokens[ground_label]
        actual_subtokens = label_vocab.itosubtokens[actual]
        res=f'{ground_label_subtokens} === {actual_subtokens}'
        pre_res.append(res)
        for subtoken in ground_label_subtokens:
            if subtoken in actual_subtokens:
                subtokens_match += 1
        subtokens_ground_truth_count += len(ground_label_subtokens)
        subtokens_actual_count += len(actual_subtokens)

    accuracy = subtokens_match / (subtokens_ground_truth_count + subtokens_actual_count - subtokens_match)
    precision = subtokens_match / subtokens_actual_count
    recall = subtokens_match / subtokens_ground_truth_count
    if precision + recall > 0:
        f1 = 2.0 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    return accuracy, precision, recall, f1, pre_res

def inverse_eage(data):
    '''
        Input:
            data: PyG data object
        Output:
            data (edges are augmented in the following ways):
                data.edge_index: Added next-token edge. The inverse edges were also added.
                data.edge_attr (torch.Long):
                    data.edge_attr[:,0]: whether it is a edge label
                    data.edge_attr[:,1]: whether it is original direction (0) or inverse direction (1)
                    IF IN DATA: n types for graph edges
                    data.edge_attr[:, 1+n]: types, n > 0

    '''
  
    if data.edge_index.nelement() == 0:
        edge_index = data.edge_index
        data.edge_attr = torch.zeros((edge_index.size(1), 2), dtype=torch.long)
        return data
    edge_index = data.edge_index
    edge_attr = torch.zeros((edge_index.size(1), 2), dtype=torch.long)
    #print( data.edge_attr.shape )
    edge_attr[:, 0] = data.edge_attr
   
    # print( data.edge_index.shape  )
    # print( data.edge_attr.shape  )
    ##### Inversed edge
    edge_index_inverse = torch.stack([edge_index[1], edge_index[0]], dim = 0)
    
    try:
        edge_attr_inverse = torch.cat([data.edge_attr.view(-1, 1), torch.ones(edge_index_inverse.size(1), 1)], dim = 1)
    except:
        print(data.edge_index)
        print(data.edge_attr.shape)
        print(edge_index_inverse.shape)
 
    
    data.edge_index = torch.cat([edge_index, edge_index_inverse], dim = 1).to(torch.long)
    data.edge_attr = torch.cat([edge_attr,   edge_attr_inverse], dim = 0).to(torch.long)
    #print( data.edge_attr.shape )
    # print( data.edge_index.shape  )
    # print( data.edge_attr.shape  )
    return data
