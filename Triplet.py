import os
import sys
import torch
from torch.utils.data import Dataset
import csv
import random
import numpy as np
import util

class Triplet(Dataset):
    def __init__(self, mode, args, dictionary):
        self.mode = mode
        self.dictionary = dictionary
        self.emb_dim = args.emb_dim
        self.max_dic_size = dictionary.size()
        self.max_q_len = args.max_q_len
        self.max_doc_len = args.max_doc_len
        self.len = 0

        if(self.mode == 'train'):
            data_file = args.train_file
        elif(self.mode == 'doc'):
            data_file = args.doc_file
        elif(self.mode == 'query'):
            data_file = args.query_file

        self.datalines = self.loadData(data_file)
        self.embeddings = self.readEmbedding(args.pre_trained_embedding_file)
        
    def loadData(self, fname):
        data = []
        with open(fname, 'r') as fp:
            for line in fp: 
                data.append(line)  
        self.len = len(data)
        print(self.len, ' lines in triplet file', file=sys.stderr)
        return data 

    def readEmbedding(self, pretrained_file):
        term_to_id, id_to_term, we_matrix = util.load_word_embeddings(pretrained_file, self.emb_dim)
        init_matrix = np.random.random((self.max_dic_size+1, self.emb_dim))
        for i in range(self.max_dic_size):
            if self.dictionary.id_to_term[i] in term_to_id:
                tid = term_to_id[self.dictionary.id_to_term[i]]
                init_matrix[i] = we_matrix[tid]
        init_matrix[i+1] = np.zeros([1, self.emb_dim])  ## zero padding
        print(i, ' embeddings are read from the pretrained file', file=sys.stderr)
        return torch.from_numpy(init_matrix)

    def len_check(self, ta, maxlen):  
        pad_arr = np.full(maxlen-len(ta), self.max_dic_size)
        return np.append(ta, pad_arr)

    def __getitem__(self, index):
        line = self.datalines[index]
        psd = (line.rstrip()).split('\t')    

        if(self.mode == 'train'):
            query = np.fromstring(psd[0], dtype=int, sep=',')
            doc1 = np.fromstring(psd[1], dtype=int, sep=',')
            doc2 = np.fromstring(psd[2], dtype=int, sep=',')
            label = np.fromstring(psd[3], dtype=int, sep=',')

            query = torch.from_numpy(self.len_check(query, self.max_q_len))
            doc1 = torch.from_numpy(self.len_check(doc1, self.max_doc_len))
            doc2 = torch.from_numpy(self.len_check(doc2, self.max_doc_len))
            #label = torch.from_numpy(np.concatenate([label, 0-label]))
            label = torch.from_numpy(np.concatenate([0-label, label]))
  
            #print(query)
            #print(query.shape)
            query_emb = torch.index_select(self.embeddings, 0, query).unsqueeze(1)
            doc1_emb = torch.index_select(self.embeddings, 0, doc1).unsqueeze(1)
            doc2_emb = torch.index_select(self.embeddings, 0, doc2).unsqueeze(1)
            #print(query_emb.shape)
            #print(doc1_emb.shape)
            #print(doc2_emb.shape)
    
            return query_emb.transpose(0,2), \
                   doc1_emb.transpose(0,2), \
                   doc2_emb.transpose(0,2), \
                   label
        elif(self.mode == 'doc'):
            doc_id  = np.fromstring(psd[0][1:], dtype=int, sep=',')
            doc = np.fromstring(psd[1], dtype=int, sep=',')

            doc_id = torch.from_numpy(doc_id)
            doc = torch.from_numpy(self.len_check(doc, self.max_doc_len))

            doc_emb = torch.index_select(self.embeddings, 0, doc).unsqueeze(1)
            
            return doc_id, doc_emb.transpose(0,2)

        elif(self.mode == 'query'):
            q_id  = np.fromstring(psd[0], dtype=int, sep=',')
            query = np.fromstring(psd[1], dtype=int, sep=',')

            query = torch.from_numpy(self.len_check(query, self.max_q_len))
            q_id = torch.from_numpy(q_id)

            query_emb = torch.index_select(self.embeddings, 0, query).unsqueeze(1)
            
            return q_id, query_emb.transpose(0,2)
           
        else:
            return None

    def __len__(self):
        return self.len


