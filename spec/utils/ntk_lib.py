import torch
from functorch import make_functional, vmap, vjp, jvp, jacrev
def ntk(net, x1): 
    fnet, params = make_functional(net) 
    jac1 = vmap(jacrev(fnet), (None, 0))(params, x1)
    jac1 = [j.flatten(2) for j in jac1]

    result = torch.stack([torch.einsum('Naf,Mbf->NMab', j1, j1) for j1, j1 in zip(jac1, jac1)])
    result = result.sum(0)

    return result.squeeze().detach()
