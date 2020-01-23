"""
Training the SNRM model with PyTorch

coded by Jaekeol Choi ( jkchoi.naver@navercorp.com)
Authors: Hamed Zamani (zamani@cs.umass.edu)

"""

import torch
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torch import optim
import argparse

from dictionary import Dictionary
from Triplet import Triplet
from psnrm import SNRM
import sys

def train():
    #1. read dictionary
    dictionary = Dictionary()
    dictionary.load_from_galago_dump(args.dict_file, args.dict_min_freq)

    #2. make snrm instance
    device = torch.device('cpu')
    snrm = SNRM(args).to(device)

    #3. read train data
    train_data = Triplet('train', args, dictionary)
    
    #4. train
    db_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, num_workers=0)

    for epoch in range(args.epoch):
        for i, (query, doc1, doc2, label)  in enumerate(db_loader):
            query, doc1, doc2, label = query.to(device), doc1.to(device), doc2.to(device), label.to(device)
            assert(query.shape[1] == args.emb_dim)

            accs = snrm.model_train(query, doc1, doc2, label)
            if(i % 1 == 0):
                print('step:', i, '\ttraing cost:', accs.item(), '\r', file=sys.stderr, end='')  

    print('>trainin end. final cost = ', accs.item(), file=sys.stderr)  
 
    #5. save model
    torch.save(snrm.state_dict(), args.model_file)
    print('>save model : ', args.model_file, file=sys.stderr)  

def build_index():
    print('build index..', file=sys.stderr)
    #1. read dictionary
    dictionary = Dictionary()
    dictionary.load_from_galago_dump(args.dict_file, args.dict_min_freq)

    #2. make snrm instance & load weight
    device = torch.device('cpu')
    snrm = SNRM(args).to(device)
    snrm.load_state_dict(torch.load(args.model_file))  ## load model
    snrm.eval()      ## set inference mode

    #3. read train data
    train_data = Triplet('doc', args, dictionary)
    
    #4. train
    db_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, num_workers=0)

   

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()

    ## run mode
    argparser.add_argument('--mode', type=str, help='run mode', default='train')

    ## Hyper parameter
    argparser.add_argument('--epoch', type=int, help='epoch number', default=1000) ## 100000
    argparser.add_argument('--batch_size', type=int, help='batch size', default=128)  ## 512
    argparser.add_argument('--learning_rate', type=float, help='learning rate with ADAM', default=0.0001)
    argparser.add_argument('--dropout_parameter', type=float, help='dropout', default=0.0)
    argparser.add_argument('--regularization_term', type=float, help='regularization', default=0.0001)

    ## file name
    argparser.add_argument('--emb_dim', type=int, help='embedding dimension', default=100)
    argparser.add_argument('--dict_file', type=str, help='dictionary file name', default='data/dictionary.txt')
    argparser.add_argument('--train_file', type=str, help='train file name', default='data/triples.tsv')
    argparser.add_argument('--doc_file', type=str, help='doc file name', default='data/documents.tsv')
    argparser.add_argument('--pre_trained_embedding_file', type=str, help='embedding file name', default='data/embedding.txt')
    argparser.add_argument('--model_file', type=str, help='trained model file', default='model/trained.model')

    ## conv channel
    argparser.add_argument('--conv1_channel', type=int, help='channel length', default=500)
    argparser.add_argument('--conv2_channel', type=int, help='channel length', default=300)
    argparser.add_argument('--conv3_channel', type=int, help='channel length', default=5000)

    ## query, document max len
    argparser.add_argument('--max_q_len', type=int, help='maximum query length', default=10)
    argparser.add_argument('--max_doc_len', type=int, help='maximum document length', default=1000)
    argparser.add_argument('--dict_min_freq', type=int, help='minimum collection frequency of terms', default=0)

    args = argparser.parse_args()

    if(args.mode == "train"):
        train()
    elif(args.mode == "build_index"):
        build_index()
   # elif(args.mode == "retrieve"):
    

