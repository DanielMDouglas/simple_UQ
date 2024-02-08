import torch

def MSELoss(output, truth):
    
    return torch.mean(torch.pow(output - truth, 2))

def NLLLoss(output, truth):
    mean = output[:,0]
    std = torch.abs(output[:,1])

    logp = -0.5*torch.pow((mean - truth.T)/std, 2) - torch.log(std)
    
    # return torch.mean(torch.pow(mean - truth, 2))
    return -torch.sum(logp)
