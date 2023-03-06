import numpy as np
import torch
import torch.nn.functional as F
import ot

def normal_loss(model, x_1, t):
    x_0 = torch.randn_like(x_1)
    x_t = t[:, None, None, None] * x_1 + (1 - t[:, None, None, None]) * x_0
    v = model(x_t, t)
    loss = F.mse_loss(x_1 - x_0, v)
    return loss

def ot_loss(model, x_1, t):
    bs = x_1.size(0)
    x_0 = torch.randn_like(x_1)
    a = ot.unif(bs)
    b = ot.unif(bs)
    M = torch.cdist(x_0.reshape(bs, -1), x_1.reshape(bs, -1)) ** 2
    pi = ot.emd(a, b, M.detach().cpu().numpy())
    p = pi.flatten()
    p = p / p.sum()
    choices = np.random.choice(pi.shape[0] * pi.shape[1], p=p, size=bs)
    i, j = np.divmod(choices, pi.shape[1])
    x_0 = x_0[i]
    x_1 = x_1[j]

    x_t = t[:, None, None, None] * x_1 + (1 - t[:, None, None, None]) * x_0
    v = model(x_t, t)
    loss = F.mse_loss(x_1 - x_0, v)
    return loss

_loss_dict = {
    'normal': normal_loss,
    'ot': ot_loss
}

def get_loss_fn(loss_name):
    return _loss_dict[loss_name]