import torch
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

def getRepr(fname):
    data = []
    fp = open(fname + ".info", 'r')
    for line in fp:
        data.append(line)

    rep = np.fromstring(data[1].rstrip(), dtype=float, sep=' ')
    
    return rep[1:]
    
mode="all"

doc_fname ="../result/epath.doc."+mode
q_fname ="../result/epath.query."+mode
#1. read repr doc, query
doc_repr = getRepr(doc_fname)
q_repr = getRepr(q_fname)

#2. compute score
dot = doc_repr * q_repr
score = np.sum(dot)

print('score = ', score)

sorted_ind = np.flip(np.argsort(dot))
k=0
ssum=0
for ind in sorted_ind:
    ssum += dot[ind]
    print(ind, '=', dot[ind], '\t', int(ssum / score * 10000)/100)
    k += 1
    if(k > 10): break


#3. draw graph
plt.figure(figsize=(40, 5))
df = pd.DataFrame(dict(dim=np.arange(10000), value=dot))
g = sns.lineplot(x='dim', y='value', data=df)

#plt.savefig("repr_all.png")
plt.savefig("score_al.png")
