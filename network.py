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

import yaml
import os

def loadManifestDict(manifest):
    """
    Load the manifest dictionary from a dictionary or a yaml path
    """
    if type(manifest) == dict: # if arg is a dict, do nothing
        manifestDict = manifest
    elif type(manifest) == str: # if arg is a yaml path, load
        with open(manifest, 'r') as mf:
            manifestDict = yaml.load(mf, Loader = yaml.FullLoader)
            
    assert type(manifestDict) == dict

    return manifestDict


def main(args):
    # manifestPath = 'singleLayer_uq.yaml'
    # manifestPath = 'singleLayer_blind.yaml'
    manifestPath = args.manifest
    manifest = loadManifestDict(manifestPath)

    Npoints = manifest['Npoints']

    if manifest['model'] == 'single_layer':
        model = nn.Sequential(nn.Linear(in_features=2*Npoints, out_features=2, bias=False)).to(device)
    elif manifest['model'] == '4_layer':
        model = nn.Sequential(nn.Linear(in_features=2*Npoints, out_features=128),
                              nn.ReLU(),
                              nn.Linear(in_features=128, out_features=128),
                              nn.ReLU(),
                              nn.Linear(in_features=128, out_features=128),
                              nn.ReLU(),
                              nn.Linear(in_features=128, out_features=2)).to(device)

    model.train()

    if 'init_params' in manifest and manifest['init_params'] == True:
        model[0].weight = nn.Parameter(torch.stack((torch.cat((torch.ones(Npoints),
                                                               torch.zeros(Npoints))),
                                                    torch.cat((torch.zeros(Npoints),
                                                               torch.ones(Npoints))))).to(device))

    refModel = nn.Sequential(nn.Linear(in_features=2*Npoints, out_features=2, bias=False)).to(device)
    refModel.eval()
    refModel[0].weight = nn.Parameter(torch.stack((torch.cat((torch.ones(Npoints),
                                                              torch.zeros(Npoints))),
                                                   torch.cat((torch.zeros(Npoints),
                                                              torch.ones(Npoints))))).to(device))

    if 'point_params' in manifest:
        pointParams = manifest['point_params']
    else:
        pointParams = (50, 100)
    print ("point params", pointParams)
    pointDist = st.uniform(*pointParams)

    if 'noise_params' in manifest:
        noiseParams = manifest['noise_params']
    else:
        noiseParams = (0, 50)
    print ("noise params", noiseParams)
    noiseDist = st.uniform(*noiseParams)

    optimizer = optim.Adam(model.parameters(),
                           lr = 1.e-2)

    maxIter = manifest['maxIter']
    bs = manifest['batchSize']

    # load = False
    load = manifest['resume']
    # save = True
    save = manifest['save']

    if 'verbose' in manifest:
        verbose = manifest['verbose']
    else:
        verbose = False

    uq = manifest['input'] == 'uq'

    baseDir = manifest['baseDir']
    os.makedirs(baseDir, exist_ok = True)
    weightFileName = os.path.join(baseDir, manifest['weightFileName'])
    trainFileName = os.path.join(baseDir, manifest['trainFileName'])

    if load:
        with open(weightFileName, 'rb') as f:
            checkpoint = torch.load(f,
                                    map_location = device)
            model.load_state_dict(checkpoint['model'], strict=False)

    trainList = []

    if verbose:
        iterator = tqdm(range(maxIter))
    else:
        iterator = range(maxIter)
            
    for iter in iterator:
        inpt, inptNoise, truth = generate_input_batch(bs = bs,
                                                      N = Npoints,
                                                      points = pointDist,
                                                      noise = noiseDist,
                                                      label = sum)

        optimizer.zero_grad()

        if save and iter%manifest['iterPerCKPT'] == 0:
            torch.save(dict(model = model.state_dict()), weightFileName)
            np.save(trainFileName, trainList)
        
        if uq:
            output = model(torch.cat((inpt, inptNoise), dim = 1))
        else:
            output = model(torch.cat((inpt, torch.zeros_like(inpt)), dim = 1))

        refOutput = refModel(torch.cat((inpt, inptNoise), dim = 1))
        m_e = refOutput[:,0]
        v_e = torch.abs(refOutput[:,1])
        s_e = torch.sqrt(v_e)

        mean = output[:,0]
        var = torch.abs(output[:,1])
        std = torch.sqrt(var)
        
        criterion = NLLLoss
        # criterion = sampledMSE
        loss = criterion(mean, var, truth.T)
        refLoss = criterion(m_e, v_e, truth.T)

        trainList.append(loss.item())
        
        if verbose:
            iterator.set_description("loss: "+str(round(loss.item(), 2)) + " refLoss: "+str(round(refLoss.item(), 2)) + " LLR: " + str(round(loss.item()/refLoss.item(), 2)))

            if iter%100 == 0:
                print(model[0].weight)

            # print (inpt.shape)
            # print (truth.shape)
    
            # print ("mean estimate", m_e)
            # print ("std estimate", v_e)

            # print ("mean inference", mean)
            # print ("std inference", var)
        
            # print (truth.flatten())
            # print()
    
        loss.backward()
        optimizer.step()

    print("final weights:", model[0].weight)

    if save:
        torch.save(dict(model = model.state_dict()), weightFileName)
        np.save(trainFileName, trainList)

if __name__ == '__main__':
    
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--manifest', type = str,
                        default = "/sdf/home/d/dougl215/studies/NDLArSimReco/NDLArSimReco/manifests/testManifest.yaml",
                        help = "network manifest yaml file")
    
    args = parser.parse_args()

    main(args)
