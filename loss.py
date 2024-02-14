import torch

def MSELoss(output, truth):
    return torch.mean(torch.pow(output - truth, 2))

# def NLLLoss(mean, std, truth):
def NLLLoss(mean, var, truth):
    # logp = -0.5*torch.pow((mean - truth.T)/std, 2) - torch.log(std)
    logp = -0.5*torch.pow(mean - truth.T, 2)/var - 0.5*torch.log(var)
    
    # return torch.mean(torch.pow(mean - truth, 2))
    return -torch.sum(logp)
