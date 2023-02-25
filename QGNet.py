import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import traceback
import time

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.01)


class QNet(nn.Module):
    def __init__(self, input_size, num_layers, layers_size, output_size, output_type,
                 dropout, use_t, device='cpu'):
        super(QNet, self).__init__()
        self.device = device
        self.use_t = use_t  # whether to use treatment as a predictor
        self.epsilon = nn.Parameter(torch.tensor([0.0]), requires_grad=True).to(self.device)
        self.output_type = output_type
        if not self.use_t:
            input_size -= 1
        layers = []
        layers.extend([nn.Linear(input_size, layers_size), nn.ReLU()])
        for i in range(num_layers - 1):
            layers.extend([nn.Linear(layers_size, layers_size), nn.ReLU(), nn.Dropout(p=dropout)])
        self.net = nn.Sequential(*layers)

        pos_arm = []
        pos_arm.extend([nn.Linear(layers_size, layers_size), nn.ReLU()])
        pos_arm.extend([nn.Linear(layers_size, output_size)])

        neg_arm = []
        neg_arm.extend([nn.Linear(layers_size, layers_size), nn.ReLU()])
        neg_arm.extend([nn.Linear(layers_size, output_size)])

        self.pos_arm = nn.Sequential(*pos_arm)
        self.neg_arm = nn.Sequential(*neg_arm)

        self.net.apply(init_weights)
        self.neg_arm.apply(init_weights)
        self.pos_arm.apply(init_weights)

    def forward(self, X, Z):
        out = self.net(torch.cat([X, Z], 1)) if self.use_t else self.net(Z)
        logit_0 = self.neg_arm(out)
        logit_1 = self.pos_arm(out)

        if self.output_type == 'categorical':
            out0 = torch.sigmoid(logit_0)
            out1 = torch.sigmoid(logit_1)
        elif self.output_type == 'continuous':
            out0 = logit_0
            out1 = logit_1

        cond = X.bool()
        return torch.where(cond, out1, out0), torch.where(cond, logit_0, logit_1)


class GNet(nn.Module):
    def __init__(self, input_size, num_layers, layers_size, output_size, output_type, dropout, device='cpu'):
        super(GNet, self).__init__()
        self.output_type = output_type
        self.device = device
        layers = []
        layers.extend([nn.Linear(input_size, layers_size), nn.ReLU()])
        for i in range(num_layers - 1):
            layers.extend([nn.Linear(layers_size, layers_size), nn.ReLU(), nn.Dropout(p=dropout)])
        layers.extend([nn.Linear(layers_size, output_size)])

        self.net = nn.Sequential(*layers)
        self.net.apply(init_weights)

    def forward(self, Z):
        logit = self.net(Z)
        if self.output_type == 'categorical':
            out = (0.01 + torch.sigmoid(logit)) / 1.02
        elif self.output_type == 'continuous':
            out = self.net(Z)
        return out, logit


def logit_(p):
    return torch.log(p / (1 - p))


def T_scaling(logits, temperature):
    return torch.div(logits, temperature)


class Trainer(object):
    def __init__(self, net, net_type='Q', beta=1.0, test_loss_plot=False, outcome_type='categorical', iterations=1000,
                 batch_size=30, test_iter=10, lr=0.001, weight_reg=1e-5, calibration=0, split=True, device='cpu'):
        self.net_type = net_type
        self.device = device
        self.net = net
        self.beta = beta
        self.test_loss_plot = test_loss_plot
        self.iterations = iterations
        self.batch_size = batch_size
        self.test_iter = test_iter
        self.outcome_type = outcome_type
        self.calibration = calibration
        self.weight_reg = weight_reg
        self.split = split  # whether or not to use train/test splits (can train and test on same data with causal stuff)
        self.window_length = 50  # number of measures of loss over which to determine early stopping

        self.optimizer = optim.SGD(self.net.parameters(), lr=lr, momentum=0.9, nesterov=True,
                                   weight_decay=self.weight_reg)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min',
                                                              patience=self.window_length // 2)

        self.bce_loss_func = nn.BCELoss(reduction='none')
        self.mse_loss_func = nn.MSELoss()


    def mse_loss(self, a, b):
        assert (torch.isnan(a)).sum() == 0, print('predictions contain nan value(s)')
        assert (torch.isnan(b)).sum() == 0, print('targets contain nan value(s)')
        return self.mse_loss_func(a, b)

    def bce_loss(self, a, b):
        assert(a < 0).sum() == 0 and (a > 1).sum() == 0, print('predictions out of bounds')
        assert (b < 0).sum() == 0 and (b > 1).sum() == 0, print('targets out of bounds')
        assert (torch.isnan(a)).sum() == 0, print('predictions contain nan value(s)')
        assert (torch.isnan(b)).sum() == 0, print('targets contain nan value(s)')
        return self.bce_loss_func(a, b)

    def train(self, x, y, z, x_pred=None):
        x_pred_train, x_pred_val = None, None
        # create a small validation set
        if self.split:
            print('Splitting the data for training...')
            indices = np.arange(len(x))
            np.random.shuffle(indices)
            val_inds = indices[:len(x) // 7]
            train_inds = indices[len(x) // 7:]
            x_val, y_val, z_val = x[val_inds], y[val_inds], z[val_inds]
            x_train, y_train, z_train = x[train_inds], y[train_inds], z[train_inds]

            if self.net_type == 'Q' and x_pred != None:
                x_pred_train, x_pred_val = x_pred[train_inds], x_pred[val_inds]

        else:
            print('Using all data for training...')
            x_val = x_train = x
            y_val = y_train = y
            z_val = z_train = z
            if self.net_type == 'Q' and x_pred != None:
                x_pred_train = x_pred_val = x_pred

        indices = np.arange(len(x_train))

        train_losses = []
        test_losses = []
        epsilons = []
        best_model_test_loss = 1e10
        # best_early_stop_test_loss = 1e10
        # test_loss_window = []
        # best_model_iter = 0
        stopping_iteration = self.iterations  # initialise early stopping iter as the total iters
        start = time.time()
        for it in range(self.iterations):
            if it % 500 == 0:
                print('training iteration:', it)
            inds = np.random.choice(indices, self.batch_size)
            x_batch, y_batch, z_batch = x_train[inds], y_train[inds], z_train[inds]

            if self.net_type == 'Q':
                x_pred_batch = x_pred_train[inds] if x_pred != None else None
                pred, _ = self.net(x_batch, z_batch)

                if self.outcome_type == 'categorical':
                    loss = self.bce_loss(pred, y_batch).mean()

                else:
                    loss = self.mse_loss(pred, y_batch)


            elif self.net_type == 'G':
                pred, _ = self.net(z_batch)
                loss = self.bce_loss(pred, x_batch).mean()


            loss.backward(retain_graph=True)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step(loss)

            if (it > 0) and ((it % self.test_iter == 0) or (it == (self.iterations - 1))):
                self.net.eval()

                if self.net_type == 'Q':
                    pred, _ = self.net(x_train[:len(x_val)], z_train[:len(x_val)])

                    if self.outcome_type == 'categorical':
                        loss = self.bce_loss(pred, y_train[:len(x_val)]).mean()
                    else:
                        loss = self.mse_loss(pred, y_train[:len(x_val)])


                elif self.net_type == 'G':
                    pred, _ = self.net(z_train[:len(x_val)])
                    loss = self.bce_loss(pred, x_train[:len(x_val)]).mean()


            if (it > 0) and (it % self.test_iter == 0):
                train_losses.append(loss.item())

                loss_test, _, _ = self.test(self.net, x_val, y_val, z_val, x_pred_val)

                loss_test = loss_test.detach().cpu().numpy()
                test_losses.append(loss_test.item())

                self.net.train()

        end = time.time()
        print('Training Took: ', end - start, ' seconds')
        best_model = self.net
        last_model = self.net
        temp_bm = temp_lm = None
        if self.outcome_type == 'categorical' and self.calibration:
            temp_bm = self.calibration_func(best_model, x_val, y_val, z_val)
            temp_lm = self.calibration_func(last_model, x_val, y_val, z_val)
        #         print('Best model saved at iteration:', best_model_iter )
        #         print('Last model saved at iteration:', it )

        return train_losses, test_losses, stopping_iteration, best_model, best_model_test_loss, epsilons, last_model, temp_bm, temp_lm

    def calibration_func(self, model, x, y, z):
        model.eval()
        print('Calibrating Model with Temperature Scaling...')
        # adapted from https://towardsdatascience.com/neural-network-calibration-using-pytorch-c44b7221a61

        with torch.no_grad():
            if self.net_type == 'Q':
                _, logits = model(x, z)
                target = y

            elif self.net_type == 'G':
                _, logits = model(z)
                target = x

        temp = nn.Parameter(torch.ones(1), requires_grad=True).to(self.device)
        criterion = nn.BCEWithLogitsLoss()
        calib_optim = optim.LBFGS([temp], lr=0.001, max_iter=10000, line_search_fn='strong_wolfe')

        def calib_eval():
            calib_loss = criterion(T_scaling(logits.detach(), temp), target)
            calib_loss.backward()
            return calib_loss

        calib_optim.step(calib_eval)
        return temp

    def test(self, model, x, y, z, x_pred=None):
        model.eval()

        if self.net_type == 'Q':
            pred, logits = model(x, z)
            if self.outcome_type == 'categorical':
                loss = self.bce_loss(pred, y).mean()
            else:
                loss = self.mse_loss(pred, y)

        elif self.net_type == 'G':
            pred, logits = model(z)
            loss = self.bce_loss(pred, x).mean()


        return loss, pred, logits


class Tuner(object):
    def __init__(self, x, y, z, trials, x_pred=None, test_loss_plot=False, output_type='categorical',
                 test_iter=5, net_type='Q', best_params=None, use_beta=True, use_t=True, calibration=0, device='cpu'):
        self.net_type = net_type
        self.best_params = best_params
        self.output_type = output_type
        self.x = x
        self.y = y
        self.z = z
        self.x_pred = x_pred
        self.trials = trials
        self.test_iter = test_iter
        self.best_params = best_params
        self.net = None
        self.best_model = None
        self.use_beta = use_beta
        self.use_t = use_t
        self.test_loss_plot = test_loss_plot
        self.calibration = calibration
        self.device = device

    def tune(self, run):

        output_size = 1
        if self.net_type == 'Q':
            input_size = self.z.shape[-1] + 1  # we will concatenate the treatment var inside the qnet class

        elif self.net_type == 'G':
            input_size = self.z.shape[-1]

        train_loss = []
        val_loss = []
        bs_ = []
        iters_ = []
        lr_ = []
        stop_it_ = []  # list for early stopping iteration
        layers_ = []
        dropout_ = []
        beta_ = []
        layer_size_ = []
        best_loss = 1e10
        best_losses = []
        weight_regs = []
        epsilons_ = None

        j = 0
        while j < self.trials:
            print('Tuning for RUN: {}'.format(run))
            try:
                # sample hyper params and store the history
                weight_reg = np.random.uniform(0.00001, 0.001) if self.best_params == None else self.best_params[
                    'weight_reg']
                weight_regs.append(weight_reg)
                bs = np.random.randint(10, 64) if self.best_params == None else self.best_params['batch_size']
                bs_.append(bs)
                iters = np.random.randint(1000, 5000) if self.best_params == None else self.best_params['iters']
                iters_.append(iters)
                lr = np.random.uniform(0.0001, 0.005) if self.best_params == None else self.best_params['lr']
                if self.use_beta:
                    beta = 1.0 if self.best_params == None else self.best_params['beta']
                    lr = 0.0001
                else:
                    beta = 0.0
                beta_.append(beta)
                lr_.append(lr)
                layers = np.random.randint(2, 20) if self.best_params == None else self.best_params['layers']
                layers_.append(layers)
                dropout = np.random.uniform(0.1, 0.5) if self.best_params == None else self.best_params['dropout']
                dropout_.append(dropout)
                layer_size = np.random.randint(5, 200) if self.best_params == None else self.best_params['layer_size']
                layer_size_.append(layer_size)
                print('======== Trial {} of {} ========='.format(j, self.trials - 1))
                print('Batch size', bs, ' Iters', iters, ' Lr', lr, ' Layers', layers,
                      ' Dropout', dropout, ' Layer Size', layer_size, 'beta', beta,
                      'weight reg', weight_reg)

                if self.net_type == 'Q':
                    self.net = QNet(input_size=input_size, num_layers=layers,
                                    layers_size=layer_size, output_size=output_size,
                                    output_type=self.output_type, dropout=dropout, use_t=self.use_t).to(self.device)
                elif self.net_type == 'G':
                    self.net = GNet(input_size=input_size, num_layers=layers,
                                    layers_size=layer_size, output_size=output_size,
                                    output_type=self.output_type, dropout=dropout).to(self.device)

                trainer = Trainer(net=self.net, net_type=self.net_type, beta=beta, outcome_type=self.output_type,
                                  test_loss_plot=self.test_loss_plot, device=self.device,
                                  iterations=iters, batch_size=bs, test_iter=self.test_iter, weight_reg=weight_reg,
                                  lr=lr, calibration=self.calibration, split=True)
                print('training....')
                train_loss_, val_loss_, stop_it, best_model, best_model_test_loss_, epsilons_, _, _, _ = trainer.train(self.x,
                                                                                                                 self.y,
                                                                                                                 self.z)

                print('Best number of iterations: ', stop_it, 'compared with total:', iters)
                stop_it_.append(stop_it)
                train_loss.append(train_loss_[-1])
                val_loss.append(val_loss_[-1])
                best_losses.append(best_model_test_loss_)

                total_val_loss = val_loss_[-1]

                if val_loss_[-1] < best_loss:
                    print('old loss:', best_loss)
                    print('new loss:', total_val_loss)
                    print('best model updated')
                    best_loss = best_model_test_loss_
                    self.best_model = best_model
                j += 1
            except Exception as exc:
                print(traceback.format_exc())
                print(exc)
                print('Error at trial {}:', j)

        tuning_dict = {'batch_size': bs_, 'layers': layers_, 'dropout': dropout_, 'beta': beta_,
                       'layer_size': layer_size_, 'lr': lr_, 'iters': iters_, 'stop_it': stop_it_,
                       'train_loss': train_loss, 'val_loss': val_loss, 'best_model_test_loss': best_losses,
                       'weight_reg': weight_regs}

        if self.net_type == 'G':
            _, best_model_preds, _ = trainer.test(self.best_model, self.x, self.y, self.z)
        else:
            best_model_preds = None
        return tuning_dict, self.best_model, best_model_preds, epsilons_

