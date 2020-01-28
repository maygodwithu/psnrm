import sys
import csv
import gzip
import os

dic_file = "./data/dictionary.txt"
emb_file = "./data/embedding.txt"
##1. read & write term embedding
emb={}
#with open("../data/embedding/glove.6B.100d.txt", 'r') as ef, \
with open("../data/embedding/glove.6B.300d.txt", 'r') as ef, \
     open(emb_file, 'w') as emb_out:
    tid = 1
    for line in ef:
        psd = (line.rstrip()).split(' ')
        term = psd[0]
        emb[term] = tid   ## emb id 

        emb_out.write(str(tid)) ## emb write
        for v in psd[1:]:
            emb_out.write(' ' + v)
        emb_out.write('\n')

        tid += 1 
    ef.close()
    emb_out.close()

print(tid, ' term read & write from embedding file', file=sys.stderr)

##2. print dictionary file
## format query, freq, docfreq ( instead of ids)
out = open(dic_file, 'w')
for q in emb:
    out.write(q + "\t10\t" + str(emb[q]) + "\n")
out.close()

print('dictionary file write end', file=sys.stderr)
