import numpy as np
import scipy.stats as st
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_input(N = 10,
                   points = None,
                   noise = None,
                   labelFunc = sum,
                   **kwargs):
    trueVal = points.rvs(N)
    noiseScale = noise.rvs(N)
    noiseVar = np.power(noiseScale, 2)
    noisyVal = st.norm(loc = trueVal, scale = noiseScale).rvs()

    label = labelFunc(trueVal)
    return (torch.Tensor(noisyVal).to(device),
            # torch.Tensor(noiseScale).to(device),
            torch.Tensor(noiseVar).to(device),
            torch.Tensor([label]).to(device))

def generate_input_batch(bs = 16,
                         **inptKwargs):
    valList = []
    scaleList = []
    labelList = []
    for i in range(bs):
        val, scale, label = generate_input(**inptKwargs)

        valList.append(val)
        scaleList.append(scale)
        labelList.append(label)

    valBatchTensor = torch.stack(valList)
    scaleBatchTensor = torch.stack(scaleList)
    labelBatchTensor = torch.stack(labelList)
        
    return valBatchTensor, scaleBatchTensor, labelBatchTensor

if __name__ == '__main__':
    # pointDist = st.norm(loc = 50, scale = 10)
    pointDist = st.uniform(50, 100)
    noiseDist = st.uniform(0, 50)
    generate_input(50, pointDist, noiseDist, sum)
