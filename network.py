import numpy as np
import scipy.stats as st

import torch
import torch.nn as nn
import torch.optim as optim
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from input import generate_input
from input import generate_input_batch
# from loss import MSELoss as criterion
# from loss import NLLLoss as criterion
from loss import *

from tqdm import tqdm

Npoints = 10
# Npoints = 3
model = nn.Sequential(nn.Linear(in_features=2*Npoints, out_features=128),
                      nn.ReLU(),
                      nn.Linear(in_features=128, out_features=128),
                      nn.ReLU(),
                      nn.Linear(in_features=128, out_features=128),
                      nn.ReLU(),
                      nn.Linear(in_features=128, out_features=2)).to(device)

# model = nn.Sequential(nn.Linear(in_features=2*Npoints, out_features=2, bias=False),
#                       ).to(device)

model.train()

# model[0].weight = nn.Parameter(torch.Tensor([[1, 1, 1, 0, 0, 0], 
#                                              [0, 0, 0, 1, 1, 1]]).to(device))
# print(model[0].weight.shape)

pointDist = st.uniform(50, 100)
noiseDist = st.uniform(0, 50)
# noiseDist = st.uniform(0, 20)

optimizer = optim.Adam(model.parameters(),
                       lr = 1.e-2,
                       # weight_decay = 1.,
                   )
# maxIter = 10000000
maxIter = 10000
# maxIter = 20000
# maxIter = 4000
# maxIter = 5
# bs = 4
# bs = 64
# bs = 1024
bs = 256

pbar = tqdm(range(maxIter))

load = False
save = True
verbose = False

# uq = True
uq = False

# weightFileName = 'NLLweights.ckpt'
# weightFileName = 'NLLweights_7layer.ckpt'

if uq:
    # weightFileName = 'MLP_singleLayer_uq.ckpt'
    # trainFileName = 'MLP_singleLayer_uq_train'

    weightFileName = 'MLP_4layer_uq.ckpt'
    trainFileName = 'MLP_4layer_uq_train'
else:
    # weightFileName = 'MLP_singleLayer_zeroblind.ckpt'
    # trainFileName = 'MLP_singleLayer_zeroblind_train'

    weightFileName = 'MLP_4layer_zeroblind.ckpt'
    trainFileName = 'MLP_4layer_zeroblind_train'
    
# weightFileName = 'weights.ckpt'

if load:
    with open(weightFileName, 'rb') as f:
        checkpoint = torch.load(f,
                                map_location = device)
        model.load_state_dict(checkpoint['model'], strict=False)

trainList = []

for iter in pbar:
    inpt, inptNoise, truth = generate_input_batch(bs = bs,
                                                  N = Npoints,
                                                  points = pointDist,
                                                  noise = noiseDist,
                                                  label = sum)

    optimizer.zero_grad()
    # print (inpt.shape)
    # print (inpt, inptNoise)
    # print (torch.cat((inpt, inptNoise), dim = 1))
    # print (torch.cat((inpt, inptNoise), dim = 1).shape)
    if iter%100 == 0:
        print(model[0].weight)

    if uq:
        output = model(torch.cat((inpt, inptNoise), dim = 1))
    else:
        output = model(torch.cat((inpt, torch.zeros_like(inpt)), dim = 1))

    m_e = torch.sum(inpt, axis = 1)
    # s_e = torch.sqrt(torch.sum(torch.pow(inptNoise, 2), axis = 1))
    s_e = torch.sum(inptNoise, axis = 1)
    # print (m_e.shape, s_e.shape)

    mean = output[:,0]
    # std = torch.abs(output[:,1])
    var = torch.abs(output[:,1])

    # if iter < maxIter/2:
    # criterion = MSELoss
    # loss = criterion(mean, truth.T)
    # refLoss = criterion(m_e, truth.T)
    # else:
    criterion = NLLLoss
    loss = criterion(mean, var, truth.T)
    refLoss = criterion(m_e, s_e, truth.T)
    pbar.set_description("loss: "+str(round(loss.item(), 2)) + " LLR: " + str(round(loss.item()/refLoss.item(), 2)))

    trainList.append(loss.item())
    

    # print (mean.shape, std.shape, truth.shape)
    # print (m_e.shape, s_e.shape, truth.shape)

    # print (mean, var, truth)
    # print (m_e, s_e, torch.sqrt(s_e), truth)

    # print (model.state_dict())

    # print (criterion(m_e, s_e, truth))

    if verbose:
        print (inpt.shape)
        print (truth.shape)
    
        print ("mean estimate", m_e)
        print ("std estimate", s_e)

        print ("mean inference", output[:,0])
        print ("std inference", torch.abs(output[:,1]))
        
        print (truth.flatten())
        print()
    
    loss.backward()
    optimizer.step()

print(model[0].weight)

if save:
    torch.save(dict(model = model.state_dict()), weightFileName)
    np.save(trainFileName, trainList)

