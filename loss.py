import torch

# def MSELoss(output, truth):
#     return torch.mean(torch.pow(output - truth, 2))

def MSELoss(mean, var, truth):
    return torch.mean(torch.pow(mean - truth, 2))

def NLLLoss(mean, var, truth):
    # logp = -0.5*torch.pow(mean - truth.T, 2)/var - 0.5*torch.log(var)

    # NLL = -torch.sum(logp)

    # print ((torch.pow(mean- truth, 2)/var + torch.log(var)).shape)
    # NLL = torch.sum(torch.pow(mean - truth, 2)/var + torch.log(var))
    # NLL = torch.sum(torch.pow(mean - truth, 2))
    NLL = torch.sum(0.5*torch.pow(mean - truth, 2)/var + 0.5*torch.log(var) + 0.5*torch.log(torch.tensor(2*torch.pi)))
    
    return NLL

def sampledMSE(mean, var, truth):
    # logp = -0.5*torch.pow(mean - truth.T, 2)/var - 0.5*torch.log(var)

    # NLL = -torch.sum(logp)

    # print ((torch.pow(mean- truth, 2)/var + torch.log(var)).shape)
    # NLL = torch.sum(torch.pow(mean - truth, 2)/var + torch.log(var))
    # NLL = torch.sum(torch.pow(mean - truth, 2))
    NLL = torch.sum(0.5*torch.pow(mean - truth, 2)/var + 0.5*torch.log(var) + 0.5*torch.log(torch.tensor(2*torch.pi)))

    Nsamples = 20
    # dist = torch.distributions.normal.Normal(loc = mean, scale = torch.sqrt(var))
    dist = torch.distributions.normal.Normal(loc = torch.zeros_like(mean), scale = torch.ones_like(var))
    # dist = torch.distributions.normal.Normal(loc = mean, scale = torch.ones_like(var))
    # torch.mean(
    sampleMags = dist.sample(torch.tensor([Nsamples]))
    samples = mean + sampleMags*torch.sqrt(var)
    # print ("inputs", mean, var)
    # print ("samples", samples)
    # print ("diff", samples - truth)
    MSE = torch.mean(torch.pow(samples - truth, 2))
    # print ("MSE", MSE)
    
    # return NLL

    return MSE
