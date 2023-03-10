import numpy as np
import torch
from scipy.integrate import solve_ivp


@torch.no_grad()
def sample_ode(model, noise, method='RK45'):
    shape = noise.shape
    device = next(model.parameters()).device
    x = noise

    b = shape[0]
    def ode_func(t, x):
        x = torch.tensor(x, device=device, dtype=torch.float).reshape(shape)
        t = torch.full(size=(b,), fill_value=t, device=device, dtype=torch.float).reshape((b,))
        v = model(x, t)
        return v.cpu().numpy().reshape((-1,)).astype(np.float64)
    
    res = solve_ivp(ode_func, (0., 1.), x.reshape((-1,)).cpu().numpy(), method=method)
    x = torch.tensor(res.y[:, -1], device=device).reshape(shape)
    return x
