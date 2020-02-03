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
#from psnrm import SNRM
from path_snrm import SNRM
import sys
from inverted_index import InMemoryInvertedIndex

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
                print('epoch : ', epoch, ' step:', i, '\ttraing cost:', accs.item(), '\r', file=sys.stderr, end='')  

            if(i % 100 == 0):
                ## evaluation()                                 ## evaluation per 100-batch  
                torch.save(snrm.state_dict(), args.model_file)  ## save model per 100-batch

        torch.save(snrm.state_dict(), args.model_file)  ## save model per epoch

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
    doc_data = Triplet('doc', args, dictionary)
    
    #4. make index 
    db_loader = DataLoader(dataset=doc_data, batch_size=1, shuffle=False, num_workers=0)
   
    inverted_index = InMemoryInvertedIndex(args.conv3_channel)  ## last channel is output representation
    with torch.no_grad():
        for i, (doc_id, doc, doc_idx)  in enumerate(db_loader):
            doc_repr = snrm(doc.float())
            inverted_index.add(doc_id.numpy(), doc_repr.numpy())
            if(i % 10 == 0):
                print(i, ' document inferenced \r', file=sys.stderr, end='')  

    inverted_index.store(args.index_file)
    print('>save index: ', args.index_file, file=sys.stderr)  

def retrieve():
    import pickle as pkl
    print('retrieve...', file=sys.stderr)
    #1. read dictionary
    dictionary = Dictionary()
    dictionary.load_from_galago_dump(args.dict_file, args.dict_min_freq)

    #2. make snrm instance & load weight
    device = torch.device('cpu')
    snrm = SNRM(args).to(device)
    snrm.load_state_dict(torch.load(args.model_file))  ## load model
    snrm.eval()      ## set inference mode

    #3. read train data
    q_data = Triplet('query', args, dictionary)
 
    #4. read index 
    inverted_index = InMemoryInvertedIndex(args.conv3_channel)
    inverted_index.load(args.index_file)

    #5. read data
    db_loader = DataLoader(dataset=q_data, batch_size=1, shuffle=False, num_workers=0)

    #6. retrieve
    with torch.no_grad():
        result = dict()
        for k, (q_id, query, q_idx)  in enumerate(db_loader):
            query_repr = snrm(query.float())

            query_repr = query_repr.numpy()
            retrieval_scores = dict()
            for i in range(len(query_repr[0])):
                if query_repr[0][i] > 0.:
                    doc_rank = 0
                    for (did, weight) in inverted_index.index[i]:
                        #print('did=', did)
                        #print('weight=', weight)
                        docid = did[0]
                        if docid not in retrieval_scores:
                            retrieval_scores[docid] = 0.
                        retrieval_scores[docid] += query_repr[0][i] * weight
                        doc_rank += 1
    
            if(k % 10 == 0):
                print(k, ' query retrieved \r', file=sys.stderr, end='')  
                #break

            qid = q_id[0]
            result[qid] = sorted(retrieval_scores.items(), key=lambda x: x[1])
            print('qid=', qid)
            print('result=', result[qid])

    pkl.dump(result, open(args.retrieve_result_file, 'wb'))
    print('>save result: ', args.retrieve_result_file, file=sys.stderr)  

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()

    ## run mode
    argparser.add_argument('--mode', type=str, help='run mode', default='train')

    ## Hyper parameter
    argparser.add_argument('--epoch', type=int, help='epoch number', default=1000) ## 100000
    argparser.add_argument('--batch_size', type=int, help='batch size', default=64)  ## 512
    argparser.add_argument('--learning_rate', type=float, help='learning rate with ADAM', default=0.0001)
    argparser.add_argument('--dropout_parameter', type=float, help='dropout', default=0.0)
    argparser.add_argument('--regularization_term', type=float, help='regularization', default=0.0000001) ## 0.1^10-8 ( sparcity : 0.65)
    #argparser.add_argument('--regularization_term', type=float, help='regularization', default=0.0001)

    ## file name
    argparser.add_argument('--emb_dim', type=int, help='embedding dimension', default=300)
    argparser.add_argument('--dict_file', type=str, help='dictionary file name', default='data/dictionary.txt')
    argparser.add_argument('--train_file', type=str, help='train file name', default='data/triples.tsv')
    argparser.add_argument('--doc_file', type=str, help='doc file name', default='data/triples.tsv_doc_100')
    argparser.add_argument('--query_file', type=str, help='query file name', default='data/triples.tsv_q')
    argparser.add_argument('--pre_trained_embedding_file', type=str, help='embedding file name', default='data/embedding.txt')
    argparser.add_argument('--model_file', type=str, help='trained model file', default='model/trained.model')
    argparser.add_argument('--index_file', type=str, help='inverted index file', default='model/inverted-index.pkl')
    argparser.add_argument('--retrieve_result_file', type=str, help='retrieve_result_file', default='result/search_result.pkl')

    ## conv channel
    argparser.add_argument('--conv1_channel', type=int, help='channel length', default=500)
    argparser.add_argument('--conv2_channel', type=int, help='channel length', default=300)
    argparser.add_argument('--conv3_channel', type=int, help='channel length', default=10000)

    ## query, document max len
    argparser.add_argument('--max_q_len', type=int, help='maximum query length', default=10)
    argparser.add_argument('--max_doc_len', type=int, help='maximum document length', default=1000)
    argparser.add_argument('--dict_min_freq', type=int, help='minimum collection frequency of terms', default=0)

    args = argparser.parse_args()

    if(args.mode == "train"):
        train()
    elif(args.mode == "build_index"):
        build_index()
    elif(args.mode == "retrieve"):
        retrieve()
    

