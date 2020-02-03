import torch
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np

#mode='one'
mode='all'
input_type='query'
#input_type='doc'
if(input_type is 'query'):
    term_len = 10
else:
    term_len = 1000   ## for doc

##1. read dictionary
fp = open('../data/dictionary.txt', 'r')
dic = {}
for line in fp:
    psd = (line.rstrip()).split('\t')
    term = psd[0] 
    tid = psd[2]
    dic[tid] = term
dic['0'] = 'Unknown'
maxl = len(dic)
dic[str(maxl)] = 'Padding'

##2. read path
#data = torch.load("../result/epath.doc.all")
fname ="../result/epath."+input_type+"."+mode
data = torch.load(fname)

##3. read info
fp = open(fname + ".info", 'r')
for line in fp:
    dids = (line.rstrip()).split(' ')
    break


##3. make heatmap image
#gdata = torch.zeros(300, 1000)
gdata = torch.zeros(300, term_len)
gdata.flatten()[ data[0].long() ] = data[2]

mdata = torch.sum(gdata, 0)
tv, tp = torch.sort(mdata, descending=True)
#print(mdata.shape)

statfp = open(fname + ".stat", 'w')
for i in range(len(tp)):
    v = tv[i].item()
    p = tp[i].item()
    did = dids[p+1]
    print(p+1, '\t', dic[did], '\t', v, '\t', did, file=statfp)

#i=0
#for nv in mdata:
    #i += 1
    #did = dids[i]
    #print(i, '\t', dic[did], '\t', nv.item(), '\t', did)

np_data = gdata.numpy()
plt.figure(figsize=(20, 5))
ax = sns.heatmap(np_data)

#plt.savefig('allpath.png')
plt.savefig(mode +"."+input_type+".png")

