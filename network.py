import numpy as np
import scipy.stats as st

import torch
import torch.nn as nn
import torch.optim as optim
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from input import generate_input
from input import generate_input_batch
# from loss import MSELoss as criterion
from loss import NLLLoss as criterion

from tqdm import tqdm

Npoints = 10
model = nn.Sequential(nn.Linear(in_features=Npoints, out_features=128),
                      nn.Linear(in_features=128, out_features=128),
                      # nn.Linear(in_features=128, out_features=128),
                      # nn.Linear(in_features=128, out_features=128),
                      # nn.Linear(in_features=128, out_features=1))
                      nn.Linear(in_features=128, out_features=2))
# model = nn.Sequential(nn.Linear(in_features=Npoints, out_features=1),
#                       )
model.train()

pointDist = st.uniform(50, 100)
noiseDist = st.uniform(0, 50)

optimizer = optim.Adam(model.parameters(),
                       lr = 1.e-8)
# maxIter = 10000000
maxIter = 100000
# maxIter = 10
bs = 8

pbar = tqdm(range(maxIter))

load = True
save = True
verbose = False

# weightFileName = 'NLLweights.ckpt'
weightFileName = 'NLLweights_3layer.ckpt'
# weightFileName = 'weights.ckpt'

if load:
    with open(weightFileName, 'rb') as f:
        checkpoint = torch.load(f,
                                map_location = device)
        model.load_state_dict(checkpoint['model'], strict=False)

for i in pbar:
    inpt, inptNoise, truth = generate_input_batch(bs = bs,
                                                  N = Npoints,
                                                  points = pointDist,
                                                  noise = noiseDist,
                                                  label = sum)

    output = model(inpt)

    loss = criterion(output, truth)
    pbar.set_description("loss: "+str(round(loss.item(), 4)))

    if verbose:
        print (inpt.shape)
        print (truth.shape)
    
        print ("mean estimate", torch.sum(inpt, axis = 0))
        print ("std estimate", torch.sqrt(torch.sum(torch.pow(inptNoise, 2), axis = 0)))

        print ("mean inference", output[:,0])
        print ("std inference", torch.abs(output[:,1]))
        
        print (truth.flatten())
        print()
    
    loss.backward()
    optimizer.step()

if save:
    torch.save(dict(model = model.state_dict()), weightFileName)

