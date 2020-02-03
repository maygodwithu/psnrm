import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import util
import sys
from mynn.my_emodule import my_ReLU, my_MaxPool2d, my_Conv2d, my_Linear, my_BatchNorm2d, my_AvgPool2d, my_Mean

class SNRM(nn.Module):
    """
    Stand alone neural ranking
    """
    def __init__(self, args):
        super(SNRM, self).__init__()
        self.update_lr = args.learning_rate
        self.dropout_r = args.dropout_parameter
        self.regularization = args.regularization_term
        self.emb_dim = args.emb_dim
        self.conv1_ch = args.conv1_channel
        self.conv2_ch = args.conv2_channel
        self.conv3_ch = args.conv3_channel

        ## make network
        self.features = self._make_layers()

        ## hinge loss
        self.loss = nn.HingeEmbeddingLoss()
#        self.loss = nn.MarginRankingLoss()

        ## optimizer
        self.optimizer = optim.Adam(self.parameters(), lr = args.learning_rate)

        ## mandatory for path
        self._layers = None 

    def _make_layers(self): 
        layers = []
        layers += [my_Conv2d(self.emb_dim, self.conv1_ch, kernel_size=(1,5),padding=(0,2))]
        layers += [my_ReLU(inplace=True)]
        layers += [nn.Dropout(p=self.dropout_r, inplace=True)]
        layers += [my_Conv2d(self.conv1_ch, self.conv2_ch, kernel_size=1)]
        layers += [my_ReLU(inplace=True)]
        layers += [nn.Dropout(p=self.dropout_r, inplace=True)]
        layers += [my_Conv2d(self.conv2_ch, self.conv3_ch, kernel_size=1)]
        layers += [my_ReLU(inplace=True)]
        layers += [nn.Dropout(p=self.dropout_r, inplace=True)]
        layers += [my_Mean(dim=(2,3))]
        return nn.Sequential(*layers)

    def model_valid(self, query, doc1, doc2, label):
        with torch.no_grad():
            q_repr = self.forward(query.float())
            d1_repr = self.forward(doc1.float())
            d2_repr = self.forward(doc2.float())
            logits_d1 = torch.sum(q_repr * d1_repr, 1, keepdim=True) 
            logits_d2 = torch.sum(q_repr * d2_repr, 1, keepdim=True) 
            logits = torch.cat([logits_d1, logits_d2], dim=1)
            loss = self.loss(logits, label)
            l1_regular = torch.norm(q_repr, p=1)+torch.norm(d1_repr, p=1)+torch.norm(d2_repr, p=1)
            cost = loss + self.regularization * l1_regular
        return cost, loss

    def model_train(self, query, doc1, doc2, label):
        q_repr = self.forward(query.float())
        d1_repr = self.forward(doc1.float())
        d2_repr = self.forward(doc2.float())
        #print(q_repr.shape)
        #print(d1_repr.shape)
        #print(d2_repr.shape)
        logits_d1 = torch.sum(q_repr * d1_repr, 1, keepdim=True) 
        logits_d2 = torch.sum(q_repr * d2_repr, 1, keepdim=True) 
        logits = torch.cat([logits_d1, logits_d2], dim=1)
        #print(logits_d1.shape)
        #print(logits_d2.shape)
        #print(logits)
        #print(label.shape)
        #print(logits.shape)
        loss = self.loss(logits, label)
        l1_regular = torch.norm(q_repr, p=1)+torch.norm(d1_repr, p=1)+torch.norm(d2_repr, p=1)
        cost = loss + self.regularization * l1_regular
        print('loss       =', loss)
        print('L1 regualr = ', l1_regular)
        print('cost       =', cost)
        print('d1 shape       =', d1_repr.shape)
        print('d1_repr(>0)    =', len(d1_repr[d1_repr>0]))
        print('d2_repr(>0)    =', len(d2_repr[d2_repr>0]))
        print('q_repr(>0)     =', len(q_repr[q_repr>0]))

        self.optimizer.zero_grad()
        cost.backward()
        self.optimizer.step()
  
        return cost, loss
       
    def forward(self, x):
        out = self.features(x)
        #out = torch.mean(out, (2,3)) 
        return out

    ###!! Mandatory functions
    # fill_layers()
    def fill_layers(self, x):
        self._layers = []

        for fe in reversed(self.features): ## backward by the reversed order
            name = fe._get_name()
            if('Dropout' in name): continue
            shape = fe.getOutShape()
            self._layers.append((name, shape, fe))
            print(name, shape)

        self._layers.append(('Input', x.shape, None))
 
