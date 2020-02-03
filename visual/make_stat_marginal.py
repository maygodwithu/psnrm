import torch
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

#mode='one'
mode='all'
#input_type='query'
input_type='doc'
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

#statfp = open(fname + ".stat", 'w')
qs=[]
vs=[]
for i in range(len(tp)):
    if(i>20): break
    v = tv[i].item()
    p = tp[i].item()
    did = dids[p+1]
    #print(p+1, '\t', dic[did], '\t', v, '\t', did, file=statfp)
    qs.append(dic[did])
    vs.append(v)

np_data = gdata.numpy()
if(input_type is "doc"):
    plt.figure(figsize=(20, 5))
elif(input_type is "query"):
    plt.figure(figsize=(10, 10))

df = pd.DataFrame(dict(dim=qs, value=vs))
ax = sns.lineplot(x='dim', y='value', data=df)

#plt.savefig('allpath.png')
plt.savefig(mode +"."+input_type+".term.png")

