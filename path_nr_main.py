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
from path_snrm import SNRM
import sys
from epath import epathfinder

def printResult(docid, x, doc_idx, nrs, outfile, ep):
    if(type(nrs) is torch.Tensor):
        saveResult(docid, x, doc_idx, nrs, outfile, ep)
    elif(type(nrs) is list):
        fn = 0
        for nr in nrs:
            tout = outfile + str(fn)
            saveResult(docid, x, doc_idx, nr, tout, ep)
            fn += 1

def saveResult(docid, x, doc_idx, nrs, outfile, ep):
    infofile = outfile + ".info"
    fp = open(infofile, 'w')

    ##1. term ids
    np_didx = (doc_idx.squeeze()).numpy()
    print(docid[0,0].item(), file=fp, end='')
    for tid in np_didx:
        print(' %d' % tid, file=fp, end='')
    print('\n', file=fp, end='')

    ##2. embedding value
    ep.set_mode(x, 0)
    y = ep.net(x)
    np_y = (y.squeeze()).numpy()
    print(docid[0,0].item(), file=fp, end='')
    for tid in np_y:
        print(' %f' % tid, file=fp, end='')
    print('\n', file=fp, end='')

    ##3. other info
    x_bar = torch.zeros(x.shape)         
    x_bar.flatten()[ nrs[0].long() ] = nrs[2]
    word_sum = torch.sum(x_bar, dim=(0,1,2)) ## last dimesion is for word
    print('word sum nonzero count = ', len(word_sum[word_sum>0]), file=fp)
    tv, tp = torch.sort(word_sum, descending=True)
    print('top 20 position=', tp[:20], file=fp)
    print('top 20 value=', tv[:20], file=fp)
    # count per word
    x_bar = torch.zeros(x.shape)         
    x_bar.flatten()[ nrs[0].long() ] = 1
    word_count = torch.sum(x_bar, dim=(0,1,2)) ## last dimesion is for word
    tv, tp = torch.sort(word_count, descending=True)
    print('top 20 count position=', tp[:20], file=fp)
    print('top 20 count =', tv[:20], file=fp)
 
    torch.save(nrs, outfile)

def path_main(mode, input_type):
    print('e-path find.', file=sys.stderr)
    #1. read dictionary
    dictionary = Dictionary()
    dictionary.load_from_galago_dump(args.dict_file, args.dict_min_freq)

    #2. make snrm instance & load weight
    device = torch.device('cpu')
    snrm = SNRM(args).to(device)
    snrm.load_state_dict(torch.load(args.model_file))  ## load model
    snrm.eval()      ## set inference mode

    #3. read train data
    #doc_data = Triplet('doc', args, dictionary)
    doc_data = Triplet(input_type, args, dictionary)
    
    #4. make index 
    db_loader = DataLoader(dataset=doc_data, batch_size=1, shuffle=False, num_workers=0)
   
    #5. effective path
    ep = epathfinder(snrm)      ## pathfinder initialize
    outfile = "./result/epath." + input_type + "." + mode
    #outfile = "./result/epath.all"
    #outfile = "./result/each/epath"

    with torch.no_grad():
        for i, (doc_id, doc, doc_idx)  in enumerate(db_loader):
            #print('doc_id', doc_id)
            #print('doc', doc.shape)
            if(mode == 'each'):
                nrs = ep.find_eachpath(doc.float(), Class=None, Theta=0.7, File=outfile)
            elif(mode == 'all'):
                nrs = ep.find_eallpath(doc.float(), Class=None, Theta=0.7, File=outfile)
            else:
                nrs = ep.find_epath(doc.float(), Class=None, Theta=0.7, File=outfile)
            printResult(doc_id, doc.float(), doc_idx, nrs, outfile, ep)
            break

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()

    ## run mode
    argparser.add_argument('--mode', type=str, help='run mode', default='one')
    argparser.add_argument('--input_type', type=str, help='input type', default='doc')  ## query

    ## Hyper parameter
    argparser.add_argument('--epoch', type=int, help='epoch number', default=1000) ## 100000
    argparser.add_argument('--batch_size', type=int, help='batch size', default=128)  ## 512
    argparser.add_argument('--learning_rate', type=float, help='learning rate with ADAM', default=0.0001)
    argparser.add_argument('--dropout_parameter', type=float, help='dropout', default=0.0)
    argparser.add_argument('--regularization_term', type=float, help='regularization', default=0.0001)

    ## file name
    argparser.add_argument('--emb_dim', type=int, help='embedding dimension', default=300)
    argparser.add_argument('--dict_file', type=str, help='dictionary file name', default='data/dictionary.txt')
    argparser.add_argument('--train_file', type=str, help='train file name', default='data/triples.tsv')
    argparser.add_argument('--doc_file', type=str, help='doc file name', default='data/triples.tsv_doc_100')
    argparser.add_argument('--query_file', type=str, help='query file name', default='data/triples.tsv_q_100')
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

    path_main(args.mode, args.input_type)    

