
import numpy as np
import os

def sigm(x):
    return 1/(1 + np.exp(-x))

def inv_sigm(x):
    return np.log(x/(1-x))


def generate_data(N, seed, dataset):
    np.random.seed(seed=seed)
    if dataset != 'general':
        z1 = np.random.binomial(1, 0.5, (N, 1))
        z2 = np.random.binomial(1, 0.65, (N, 1))
        z3 = np.round(np.random.uniform(0, 4, (N, 1)), 0)
        z4 = np.round(np.random.uniform(0, 5, (N, 1)), 0)
        if dataset == 'synth1' or dataset == 'synth2':
            xp = sigm(-5 + 0.05 * z2 + 0.25 * z3 + 0.6 * z4 + 0.4 * z2 * z4)
        elif dataset == 'synth3':
            xp = sigm(np.exp(-5 + 0.05 * z2 + 0.25 * z3 + 0.6 * z4 + 0.4 * z2 * z4) - 2)
        if dataset == 'synth1':
            Y1 = np.random.binomial(1, sigm(-1 + 1 - 0.1 * z1 + 0.35 * z2 + 0.25 * z3 + 0.2 * z4 + 0.15 * z2 * z4), (N, 1))
            Y0 = np.random.binomial(1, sigm(-1 + 0 - 0.1 * z1 + 0.35 * z2 + 0.25 * z3 + 0.2 * z4 + 0.15 * z2 * z4), (N, 1))
        elif dataset == 'synth2':
            Y1 = np.random.binomial(1, sigm(np.exp(-1 - 0.1 * z1 + 0.35 * z2 + 0.25 * z3 + 0.2 * z4 + 0.15 * z2 * z4)),
                                    (N, 1))
            Y0 = np.random.binomial(1, sigm(-1 + 0 - 0.1 * z1 + 0.35 * z2 + 0.25 * z3 + 0.2 * z4 + 0.15 * z2 * z4), (N, 1))
        elif dataset == 'synth3':
            Y1 = np.random.binomial(1, sigm(np.exp(-1 + 1 - 0.1 * z1 + 0.35 * z2 + 0.25 * z3 + 0.2 * z4 + 0.15 * z2 * z4)),
                                    (N, 1))
            Y0 = np.random.binomial(1, sigm(-1 + 0 - 0.1 * z1 + 0.35 * z2 + 0.25 * z3 + 0.2 * z4 + 0.15 * z2 * z4), (N, 1))

        X = np.random.binomial(1, xp, (N, 1))
        Y = Y1 * X + Y0 * (1 - X)
        Z = np.concatenate([z1, z2, z3, z4], 1)
    elif dataset == 'general':
        z1 = np.random.binomial(1, 0.5, (N, 1))
        z2 = np.random.binomial(1, 0.65, (N, 1))
        z3 = np.round(np.random.uniform(0, 4, (N, 1)), 0)
        z4 = np.round(np.random.uniform(0, 5, (N, 1)), 0)
        uz5 = np.random.randn(N, 1)
        z5 = 0.2 * z1 + uz5
        # risk vars:
        r1 = np.random.randn(N, 1)
        r2 = np.random.randn(N, 1)

        # instrumental vars:
        i1 = np.random.randn(N, 1)
        i2 = np.random.randn(N, 1)
        # treatment:
        ux = np.random.randn(N, 1)
        xp = sigm(-5 + 0.05 * z2 + 0.25 * z3 + 0.6 * z4 + 0.4 * z2 * z4 + 0.15 * z5 + 0.1 * i1 + 0.15 * i2 + 0.1 * ux)
        X = np.random.binomial(1, xp, (N, 1))
        # mediator:
        Um = np.random.randn(N, 1)
        m1 = 0.8 + 0.15 * Um
        m0 = 0.15 * Um
        # outcomes:
        Y1 = np.random.binomial(1, sigm(np.exp(-1 + m1 - 0.1 * z1 + 0.35 * z2 +
                                               0.25 * z3 + 0.2 * z4 + 0.15 * z2 * z4 + r1 + r2)),
                                (N, 1))
        Y0 = np.random.binomial(1,
                                sigm(-1 + m0 - 0.1 * z1 + 0.35 * z2 + 0.25 * z3 + 0.2 * z4 + 0.15 * z2 * z4 + r1 + r2),
                                (N, 1))
        Y = Y1 * X + Y0 * (1 - X)
        # colliders:
        C = 0.6 * Y + 0.4 * X + 0.4 * np.random.randn(N, 1)
        Z = np.concatenate([z1, z2, z3, z4, z5], 1)
    return Z, X, Y, Y1, Y0




def IHDP(path="IHDP/", seed=1):

    path_data = path
    replications = seed
    # which features are binary
    binfeats = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    # which features are continuous
    contfeats = [i for i in range(25) if i not in binfeats]

    print(os.path.join(path_data, 'ihdp_npci_train_' + str(replications) + '.csv'))
    data = np.loadtxt(os.path.join(path_data, 'ihdp_npci_train_' + str(replications) + '.csv'), delimiter=',', skiprows=1)
    x_tr, y_tr = data[:, 0], data[:, 1][:, np.newaxis]
    mu_0_tr, mu_1_tr, z_tr = data[:, 3][:, np.newaxis], data[:, 4][:, np.newaxis], data[:, 5:]
    z_tr[:, 13] -= 1
    # perm = binfeats + contfeats
    # x = x[:, perm]

    data_test = np.loadtxt(path_data + '/ihdp_npci_test_' + str(replications) + '.csv', delimiter=',', skiprows=1)
    x_test, y_test = data_test[:, 0][:, np.newaxis], data_test[:, 1][:, np.newaxis]
    mu_0_test, mu_1_test, z_test = data_test[:, 3][:, np.newaxis], data_test[:, 4][:, np.newaxis], data_test[:, 5:]
    z_test[:, 13] -= 1
    # x_test = x_test[:, perm]

    z = np.concatenate([z_tr, z_test])
    x = np.concatenate([x_tr.reshape(-1,1), x_test])
    y = np.concatenate([y_tr, y_test])
    y1 = np.concatenate([mu_1_tr, mu_1_test])
    y0 = np.concatenate([mu_0_tr, mu_0_test])
    return z, x, y, y1, y0

