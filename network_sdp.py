import numpy as np
import scipy.stats as st

import torch
import torch.nn as nn
import torch.optim as optim
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from input import generate_input
from input import generate_input_batch
from loss import *

from sdp import SDP

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

    model = nn.Sequential(nn.Linear(in_features=Npoints, out_features=1, bias=False)).to(device)
    # model[0].weight = nn.Parameter(torch.ones(Npoints).to(device))
    model = SDP(model, num_outputs = 1)
    # model.net[0].weight *= 10
    # model.net[0].weight = nn.Parameter(torch.ones(Npoints).to(device))
    
    model.train()

    # refModel = nn.Sequential(nn.Linear(in_features=Npoints, out_features=1, bias=False)).to(device)
    refModel = nn.Sequential(nn.Linear(in_features=2*Npoints, out_features=2, bias=False)).to(device)
    refModel[0].weight = nn.Parameter(torch.stack((torch.cat((torch.ones(Npoints),
                                                              torch.zeros(Npoints))),
                                                   torch.cat((torch.zeros(Npoints),
                                                              torch.ones(Npoints))))).to(device))    
    # refModel[0].weight = nn.Parameter(torch.ones(Npoints).to(device))
    refModel.eval()

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

        Nsamples = 1

        shiftDist = torch.distributions.normal.Normal(loc = torch.zeros_like(inpt),
                                                      scale = torch.ones_like(inptNoise))
        shiftDir = shiftDist.sample(torch.tensor([Nsamples]))
        # print (shiftDir)
        # shift = shiftDir*torch.sqrt(inptNoise)
        shift = shiftDir*torch.zeros_like(inptNoise)

        # print (shift.shape)
        shiftedInpt = (inpt + shift).flatten(end_dim = -2)
        repNoise = inptNoise.repeat(Nsamples, 1)

        repTruth = truth.repeat_interleave(Nsamples, 1).T.flatten()

        # print (inpt, shiftedInpt, torch.sqrt(inptNoise), torch.sqrt(repNoise))
        # print ("shapes", inpt.shape, shiftedInpt.shape, inptNoise.shape, repNoise.shape)

        # print (truth, repTruth)
        # print (truth.shape, repTruth.shape)

        # print (inpt.shape, shift.shape)

        # inptTensor = torch.cat((shiftedInpt, repNoise), dim = 1)
        inptTensor = shiftedInpt
        # print ("input tensor", inptTensor, inptTensor.shape)

        output, cov = model(inpt, torch.sqrt(inptNoise))
        # print (output.shape, cov[:,:,0].shape)
        # print ("output tuple", output)

        refOutput = refModel(torch.cat((inpt, inptNoise), dim = 1))

        # print(output.shape)
        
        # print (shiftedInpt)
        # print (refOutput)
        # print (repTruth)
        
        # print (shifted
        # m_e = refOutput[:]
        m_e = refOutput[:,0]
        v_e = refOutput[:,1]
        
        mean = output[:,0]
        var = cov[:,0,0]
        # mean = output[:,0]
        # var = torch.abs(output[:,1])

        # print (m_e.shape, v_e.shape, mean.shape, var.shape)
        
        # criterion = NLLLoss
        if manifest['loss'] == 'MSE':
            criterion = MSELoss
        elif manifest['loss'] == 'NLL':
            criterion = NLLLoss
        else:
            # default: NLL
            criterion = NLLLoss

        # print (inptTensor, mean, repTruth) 
        
        # criterion = MSELoss
        # loss = criterion(mean, repTruth)
        # refLoss = criterion(m_e, repTruth)
        loss = criterion(mean, var, repTruth)
        refLoss = criterion(m_e, v_e, repTruth)

        trainList.append(loss.item())
        
        if verbose:
            iterator.set_description("loss: "+str(round(loss.item(), 2)) + " refLoss: "+str(round(refLoss.item(), 2)) + " LLR: " + str(round(loss.item()/refLoss.item(), 2)))

            if iter%100 == 0:
                print(model.net[0].weight)

                # print (inpt, inptNoise)

                # print (refOutput)
                # print (mean, var)

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

    print("final weights:", model.net[0].weight)

    if save:
        torch.save(dict(model = model.net[0].state_dict()), weightFileName)
        np.save(trainFileName, trainList)

if __name__ == '__main__':
    
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--manifest', type = str,
                        default = "/sdf/home/d/dougl215/studies/NDLArSimReco/NDLArSimReco/manifests/testManifest.yaml",
                        help = "network manifest yaml file")
    
    args = parser.parse_args()

    main(args)
