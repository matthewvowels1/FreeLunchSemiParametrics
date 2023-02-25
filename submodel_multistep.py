import numpy as np
import torch
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt

def IF(x, y, Q0, Q1, G10):
    D0 = ((1 - x) * (y - Q0)) / (1 - G10) + Q0 - torch.mean(Q0)
    D1 = (x * (y - Q1) / G10) + Q1 - torch.mean(Q1)
    return D0, D1


class Submodel_Optim(nn.Module):
    def __init__(self):
        super().__init__()
        self.gamma0 = torch.nn.Parameter(torch.tensor([0.0]), requires_grad=True)
        self.gamma1 = torch.nn.Parameter(torch.tensor([0.0]), requires_grad=True)

    def forward(self, x, y, Q0, Q1, G10, use_y=False):
        H1 = (x / (G10))
        H0 = ((1 - x) / (1 - G10))
        Q0_next = Q0 + self.gamma0 * H0
        Q1_next = Q1 + self.gamma1 * H1

        D0, D1 = IF(x=x, y=y, Q0=Q0_next, Q1=Q1_next, G10=G10)

        return Q0_next, Q1_next, D0, D1

class Nonlinear_Submodel_Optim(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 10)
        self.act = torch.nn.LeakyReLU(0.1)
        self.linear2 = torch.nn.Linear(10, 1)

    def forward(self, x, y, Q0, Q1, G10, use_y=1):
        H1 = (x / (G10))
        H0 = ((1 - x) / (1 - G10))
        H1, H0, Q1, Q0 = H1[:, None], H0[:, None], Q1[:, None], Q0[:, None]

        if use_y == 1:
            feed_Q1 = Q1
            feed_Q0 = Q0
        elif use_y == 0:
            feed_Q1 = torch.zeros_like(Q1)
            feed_Q0 = torch.zeros_like(Q0)

        Q1_H1 = torch.cat([feed_Q1, H1], 1)
        Q0_H0 = torch.cat([feed_Q0, H0], 1)

        Q0_next = Q0 + self.linear2(self.act(self.linear1(Q0_H0)))
        Q1_next = Q1 + self.linear2(self.act(self.linear1(Q1_H1)))

        D0, D1 = IF(x=x, y=y, Q0=Q0_next, Q1=Q1_next, G10=G10)

        return Q0_next, Q1_next, D0, D1

def submodel_optim_train(x, y, Q0, Q1, G10, model, optim, meanvar, use_y=1):
    device = 'cpu'
    x = torch.tensor(x, dtype=torch.float32, device=device)
    y = torch.tensor(y, dtype=torch.float32, device=device)
    Q0 = torch.tensor(Q0, dtype=torch.float32, device=device)
    Q1 = torch.tensor(Q1, dtype=torch.float32, device=device)
    G10 = torch.tensor(G10, dtype=torch.float32, device=device)
    if meanvar:
        mv = 1.0
    else:
        mv = 0.0
    batch_size = 500
    indexes = np.arange(len(x))
    losses = []
    mean_D1s = []
    mean_D0s = []
    var_D1s = []
    var_D0s = []
    for i in range(4000):
        if len(x) > 1000:
            inds = np.random.choice(indexes, batch_size)
            batch_x, batch_y, batch_Q0, batch_Q1, batch_G10 = x[inds], y[inds], Q0[inds], Q1[inds], G10[inds]
        else:
            batch_x, batch_y, batch_Q0, batch_Q1, batch_G10 = x, y, Q0, Q1, G10

        Q0_next, Q1_next, D0, D1 = model(batch_x, batch_y, batch_Q0, batch_Q1, batch_G10, use_y=use_y)

        D1_mean, D0_mean = D1.mean(), D0.mean()
        D1_var, D0_var = D1.var(), D0.var()
        mean_D1s.append(D1_mean.item())
        mean_D0s.append(D0_mean.item())
        var_D1s.append(D1_var.item())
        var_D0s.append(D0_var.item())

        loss0, loss1 = mv * torch.abs(D1_mean) + D1_var, mv * torch.abs(D0_mean) + D0_var
        loss0.backward(retain_graph=True)
        loss1.backward(retain_graph=False)
        optim.step()
        optim.zero_grad()
        losses.append((loss1 + loss1).item())

    Q0_next, Q1_next, D0, D1 = model(x, y, Q0, Q1, G10, use_y=use_y)

    return losses, Q0_next, Q1_next, mean_D1s, mean_D0s, var_D1s, var_D0s


def get_multistep_update(x, y, Q0, Q1, G10, meanvar, type='linear', use_y=1):
    device = 'cpu'
    if type == 'linear':
        sub_opt = Submodel_Optim().to(device)
        optimizer = optim.Adam(sub_opt.parameters(), lr=0.0005)
    elif type == 'nonlinear':
        sub_opt = Nonlinear_Submodel_Optim().to(device)
        optimizer = optim.Adam(sub_opt.parameters(), lr=0.0005)

    losses, Q0_, Q1_, mean_D1s, mean_D0s, var_D1s, var_D0s = submodel_optim_train(x=x, y=y, Q0=Q0,
                                                                                                  Q1=Q1, G10=G10,
                                                                                                  model=sub_opt,
                                                                                                  optim=optimizer,
                                                                                                  meanvar=meanvar,
                                                                                                  use_y=use_y)

    # plt.plot(losses)
    # plt.show()
    return Q0_.detach().numpy(), Q1_.detach().numpy()




