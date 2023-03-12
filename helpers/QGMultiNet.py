import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.optimize import minimize
from scipy.optimize import nnls
import traceback
import gc
import time


def fn(x, A, b):
	return np.linalg.norm(A.dot(x) - b)


def combiner_solve(x, y):
	# adapted from https://stackoverflow.com/questions/33385898/how-to-include-constraint-to-scipy-nnls-function-solution-so-that-it-sums-to-1/33388181
	beta_0, rnorm = nnls(x, y)
	cons = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
	bounds = [[0.0, None]] * x.shape[1]
	minout = minimize(fn, beta_0, args=(x, y), method='SLSQP', bounds=bounds, constraints=cons)
	beta = minout.x
	return beta


def init_weights(m):
	if isinstance(m, nn.Linear):
		torch.nn.init.xavier_normal_(m.weight)
		m.bias.data.fill_(0.01)


def logit_(p):
	return torch.log(p / (1 - p))


class QMultiNet(nn.Module):
	def __init__(self, input_size, num_layers=4, layers_size=56, output_size=1, output_type='categorical',
	             dropout=0.2, use_t=True, layerwise_optim=0, device='cpu'):
		super(QMultiNet, self).__init__()
		self.device = device
		if self.device != 'cpu':
			torch.set_default_tensor_type(torch.cuda.FloatTensor)
		self.num_layers = num_layers
		self.use_t = use_t
		self.epsilon = nn.Parameter(torch.tensor([0.0]), requires_grad=True).to(self.device)
		self.output_type = output_type
		self.layerwise_optim = layerwise_optim

		self.output_layers_pos = []
		self.output_layers_pos.extend([nn.Sequential(nn.Linear(input_size, output_size))])

		self.output_layers_neg = []
		self.output_layers_neg.extend([nn.Sequential(nn.Linear(input_size, output_size))])

		self.layers = []
		self.layers.extend([nn.Sequential(nn.Linear(input_size, layers_size), nn.ReLU())])

		for i in range(num_layers - 1):
			self.layers.extend([nn.Sequential(nn.Linear(layers_size, layers_size), nn.ReLU(), nn.Dropout(p=dropout))])
			self.output_layers_pos.extend([nn.Sequential(nn.Linear(layers_size, output_size))])
			self.output_layers_neg.extend([nn.Sequential(nn.Linear(layers_size, output_size))])

		for seq in self.output_layers_pos:
			seq.apply(init_weights)

		for seq in self.output_layers_neg:
			seq.apply(init_weights)

		for seq in self.layers:
			seq.apply(init_weights)

	def forward(self, X, Z):
		bs = X.shape[0]
		lwo = self.layerwise_optim
		inp = torch.cat([X, Z], 1) if self.use_t else Z

		outs_pos = []
		outs_neg = []
		for i in range(self.num_layers - 1):
			outs_pos.append(self.output_layers_pos[i](inp.detach() if lwo else inp))
			outs_neg.append(self.output_layers_neg[i](inp.detach() if lwo else inp))
			inp = self.layers[i](inp.detach() if lwo else inp)

		outs_pos.append(self.output_layers_pos[-1](inp.detach() if lwo else inp))
		outs_neg.append(self.output_layers_neg[-1](inp.detach() if lwo else inp))

		outs_pos = torch.cat(outs_pos, 0)
		outs_pos = outs_pos.reshape(self.num_layers, bs, 1)

		outs_neg = torch.cat(outs_neg, 0)
		outs_neg = outs_neg.reshape(self.num_layers, bs, 1)

		cond = X.bool().repeat(self.num_layers, 1).reshape(self.num_layers, bs, 1)
		logits_out = torch.where(cond, outs_pos, outs_neg)

		if self.output_type == 'categorical':
			sig_out = torch.sigmoid(logits_out)
		else:
			sig_out = logits_out
		# each output is (num layers, batch_dim, output_size)
		return sig_out, logits_out


class GMultiNet(nn.Module):
	def __init__(self, input_size, num_layers=4, layers_size=56, output_size=1, output_type='categorical',
	             dropout=0.2, layerwise_optim=0, device='cpu'):
		super(GMultiNet, self).__init__()
		self.device = device
		if self.device != 'cpu':
			print(self.device)
			torch.set_default_tensor_type(torch.cuda.FloatTensor)
		self.num_layers = num_layers
		self.output_type = output_type
		self.layerwise_optim = layerwise_optim
		self.output_layers = []
		self.output_layers.extend([nn.Sequential(nn.Linear(input_size, output_size))])

		self.layers = []
		self.layers.extend([nn.Sequential(nn.Linear(input_size, layers_size), nn.ReLU())])

		for i in range(num_layers - 1):
			self.layers.extend([nn.Sequential(nn.Linear(layers_size, layers_size), nn.ReLU(), nn.Dropout(p=dropout))])
			self.output_layers.extend([nn.Sequential(nn.Linear(layers_size, output_size))])

		for seq in self.output_layers:
			seq.apply(init_weights)

		for seq in self.layers:
			seq.apply(init_weights)

	def forward(self, Z):
		bs = Z.shape[0]
		inp = Z.to(self.device)
		lwo = self.layerwise_optim
		outs = []

		for i in range(self.num_layers - 1):
			outs.append(self.output_layers[i](inp.detach() if lwo else inp))
			inp = self.layers[i](inp.detach() if lwo else inp)

		outs.append(self.output_layers[-1](inp.detach() if lwo else inp))

		outs = torch.cat(outs, 0)

		logits_out = outs.reshape(self.num_layers, bs, 1)

		# each output is (num layers, batch_dim, output_size)
		return torch.sigmoid(logits_out), logits_out


class MultiNetTrainer(object):
	def __init__(self, net, net_type='Q', beta=1.0, test_loss_plot=False, outcome_type='categorical', iterations=1000,
	             batch_size=30, test_iter=10, lr=0.001, weight_reg=1e-5, split=True, data_masking=0, layerwise_optim=0,
	             device='cpu'):
		self.net_type = net_type
		self.net = net.to(device)
		self.device = device
		if self.device != 'cpu':
			torch.set_default_tensor_type(torch.cuda.FloatTensor)
		self.data_masking = data_masking
		self.beta = beta
		self.test_loss_plot = test_loss_plot
		self.iterations = iterations
		self.batch_size = batch_size
		self.test_iter = test_iter
		self.outcome_type = outcome_type
		self.weight_reg = weight_reg
		self.split = split  # whether or not to use train/test splits (can train and test on same data with causal stuff)
		self.window_length = 50  # number of measures of loss over which to determine early stopping
		self.beta_bm = None
		self.beta_lm = None
		self.layerwise_optim = layerwise_optim

		if self.layerwise_optim:
			self.optims = {}
			# self.schedulers = {}
			num_layers = self.net.num_layers
			for l in range(num_layers):
				layer_params = self.net.layers[l].parameters()
				if net_type == 'Q':
					out_layer_pos_params = self.net.output_layers_pos[l].parameters()
					out_layer_neg_params = self.net.output_layers_neg[l].parameters()
					optim_ = optim.SGD(
						[{"params": list(layer_params) + list(out_layer_pos_params) + list(out_layer_neg_params),
						  "lr": lr}])
				elif net_type == 'G':
					out_layer_params = self.net.output_layers[l].parameters()
					optim_ = optim.SGD([{"params": list(layer_params) + list(out_layer_params),
					                     "lr": lr}])

				self.optims['layer_{}'.format(l)] = optim_
				# self.schedulers['layer_{}'.format(l)] = optim.lr_scheduler.ReduceLROnPlateau(optim_, 'min',
				#                                               patience=self.window_length // 2)
		else:
			if net_type == 'Q':
				output_pos_params = []
				for seq in self.net.output_layers_pos:
					output_pos_params += list(seq.parameters())

				output_neg_params = []
				for seq in self.net.output_layers_neg:
					output_neg_params += list(seq.parameters())

				output_params = output_pos_params + output_neg_params

			elif net_type == 'G':

				output_params = []
				for seq in self.net.output_layers:
					output_params += list(seq.parameters())

			layer_params = []
			for seq in self.net.layers:
				layer_params += list(seq.parameters())

			all_params = [{"params": layer_params + output_params, "lr": lr}]
			self.optimizer = optim.SGD(all_params, momentum=0.9, nesterov=True,
			                           weight_decay=self.weight_reg)
			# self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min',
			#                                                       patience=self.window_length // 2)

		self.bce_loss_func = nn.BCELoss(reduction='none')
		self.mse_loss_func = nn.MSELoss()

	def mse_loss(self, a, b):
		assert (torch.isnan(a)).sum() == 0, print('predictions contain nan value(s)', a)
		assert (torch.isnan(b)).sum() == 0, print('targets contain nan value(s)', b)
		return self.mse_loss_func(a, b)

	def bce_loss(self, a, b):
		assert (a < 0).sum() == 0 and (a > 1).sum() == 0, print('predictions out of bounds')
		assert (b < 0).sum() == 0 and (b > 1).sum() == 0, print('targets out of bounds')
		assert (torch.isnan(a)).sum() == 0, print('predictions contain nan value(s)')
		assert (torch.isnan(b)).sum() == 0, print('targets contain nan value(s)')
		return self.bce_loss_func(a, b)

	def train(self, x, y, z, x_pred=None):
		torch.autograd.set_detect_anomaly(False)
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

		# prepare indices for masked data training
		ntrain = len(x_train)
		n_layers = self.net.num_layers
		chunk_inds = np.arange(0, ntrain + n_layers, ntrain // n_layers)
		chunks = []
		for l in range(len(chunk_inds - 1)):
			chunks.append(indices[chunk_inds[l]:chunk_inds[l] + (ntrain // n_layers)])
		chunks = chunks[:n_layers]

		train_losses = []
		test_losses = []
		epsilons = []
		best_model_test_loss = 1e10
		stopping_iteration = self.iterations  # initialise early stopping iter as the total iters
		gc.collect()
		start = time.time()
		for it in range(self.iterations):

			if it % 500 == 0:
				print('training iteration:', it)
			inds = np.random.choice(indices, self.batch_size)
			x_batch, y_batch, z_batch = x_train[inds], y_train[inds], z_train[inds]
			loss = {} if self.layerwise_optim else 0.0
			if self.net_type == 'Q':
				x_pred_batch = x_pred_train[inds] if x_pred != None else None
				pred, _ = self.net(x_batch, z_batch)

				for nl in range(self.net.num_layers):
					if self.data_masking:
						chunk_mask = chunks[nl]
						mask = torch.tensor(np.asarray([b in chunk_mask for b in inds]).astype('float'),
						                    dtype=torch.float32).to(self.device)
					else:
						mask = torch.tensor([1.0]).to(self.device)

					if self.outcome_type == 'categorical':
						ll = (mask * self.bce_loss(pred[nl, :, :], y_batch)).mean()
					else:
						ll = (mask * self.mse_loss(pred[nl, :, :], y_batch)).mean()

					if self.layerwise_optim:
						loss['layer_{}'.format(nl)] = ll
					else:
						loss += ll

			elif self.net_type == 'G':
				pred, _ = self.net(z_batch)
				for nl in range(self.net.num_layers):
					if self.data_masking:
						chunk_mask = chunks[nl]
						mask = torch.tensor(np.asarray([b in chunk_mask for b in inds]).astype('float'),
						                    dtype=torch.float32).to(self.device)
					else:
						mask = torch.tensor([1.0]).to(self.device)

					ll = (mask * self.bce_loss(pred[nl, :, :], x_batch)).mean()

					if self.layerwise_optim:
						loss['layer_{}'.format(nl)] = ll
					else:
						loss += ll

			if self.layerwise_optim:
				for nl in range(self.net.num_layers):
					l = loss['layer_{}'.format(nl)]
					opt = self.optims['layer_{}'.format(nl)]
					# sched = self.schedulers['layer_{}'.format(nl)]
					l.backward(retain_graph=True) if nl < (self.net.num_layers - 1) else l.backward(retain_graph=False)
					opt.step()
					opt.zero_grad()
					# sched.step(l)

			else:
				loss.backward(retain_graph=False)
				self.optimizer.step()
				self.optimizer.zero_grad()
				# self.scheduler.step(loss)

			if (it > 0) and ((it % self.test_iter == 0) or (it == (self.iterations - 1))):
				gc.collect()
				self.net.eval()
				loss = {} if self.layerwise_optim else 0.0
				if self.net_type == 'Q':
					pred, _ = self.net(x_train[:len(x_val)], z_train[:len(x_val)])
					for nl in range(self.net.num_layers):

						if self.data_masking:
							chunk_mask = chunks[nl]
							mask = torch.tensor(np.asarray([b in chunk_mask for b in inds]).astype('float'),
							                    dtype=torch.float32).to(self.device)
						else:
							mask = torch.tensor([1.0]).to(self.device)

						if self.outcome_type == 'categorical':
							ll = (mask * self.bce_loss(pred[nl, :, :], y_train[:len(x_val)])).mean()
						else:
							ll = (mask * self.mse_loss(pred[nl, :, :], y_train[:len(x_val)])).mean()

						if self.layerwise_optim:
							total_loss = sum(loss.values())
						else:
							loss += ll
							total_loss = loss

					epsilons.append(self.net.epsilon.detach().cpu().numpy()[0])

				elif self.net_type == 'G':
					pred, _ = self.net(z_train[:len(x_val)])
					for nl in range(self.net.num_layers):
						if self.data_masking:
							chunk_mask = chunks[nl]
							mask = torch.tensor(np.asarray([b in chunk_mask for b in inds]).astype('float'),
							                    dtype=torch.float32).to(self.device)
						else:
							mask = torch.tensor([1.0]).to(self.device)

						ll = (mask * self.bce_loss(pred[nl, :, :], x_train[:len(x_val)])).mean()

						if self.layerwise_optim:
							loss['layer_{}'.format(nl)] = ll
							total_loss = sum(loss.values())
						else:
							loss += ll
							total_loss = loss

			if (it > 0) and (it % self.test_iter == 0):
				train_losses.append(total_loss)

				loss_test, _, _ = self.test(self.net, x_val, y_val, z_val, x_pred_val)

				loss_test = loss_test.detach().cpu().numpy()
				test_losses.append(loss_test.item())

				self.net.train()

		end = time.time()
		print('Training Took: ', end - start, ' seconds')
		best_model = self.net
		last_model = self.net

		self.beta_bm = self.get_betas(best_model, x, y, z, x_pred)
		self.beta_lm = self.get_betas(last_model, x, y, z, x_pred)

		return train_losses, test_losses, stopping_iteration, best_model, best_model_test_loss, epsilons, last_model, self.beta_bm, self.beta_lm

	def get_betas(self, model, x, y, z, x_pred=None):
		model.eval()

		if self.net_type == 'Q':
			gts = y
			pred, logits = model(x, z)

		elif self.net_type == 'G':
			gts = x
			pred, logits = model(z)

		# need to get preds into correct shape from (num layers, batch_dim, output_size) to ->>> [bs, categories (1), predictors]
		all_preds = torch.permute(pred, (1, 2, 0))
		all_preds = all_preds.detach().cpu().numpy()
		gts = gts.detach().cpu().numpy()
		beta = combiner_solve(all_preds[:, 0, :], gts[:, 0])
		return beta

	def test(self, model, x, y, z, x_pred=None):
		gc.collect()
		model.eval()

		if self.net_type == 'Q':
			pred, logits = model(x, z)
			pred.detach()
			logits.detach()
			loss = 0
			for nl in range(model.num_layers):
				gts = y
				if self.outcome_type == 'categorical':
					loss += self.bce_loss(pred[nl, :, :], gts).mean()
				else:
					loss += self.mse_loss(pred[nl, :, :], gts)

		elif self.net_type == 'G':
			gts = x
			pred, logits = model(z)
			loss = 0
			for nl in range(model.num_layers):
				loss += self.bce_loss(pred[nl, :, :], gts).mean()

		pred = torch.permute(pred, (1, 2, 0))  # output is [bs, num cats, num layers]
		return loss, pred, logits




class MultiNetTuner(object):
	def __init__(self, x, y, z, trials, x_pred=None, test_loss_plot=False, data_masking=0, output_type='categorical',
	             test_iter=5, net_type='Q', best_params=None, use_beta=True, use_t=True, layerwise_optim=0,
	             device='cpu'):
		self.net_type = net_type
		self.device = device
		self.output_type = output_type
		if self.device != 'cpu':
			torch.set_default_tensor_type(torch.cuda.FloatTensor)
		self.best_params = best_params
		self.data_masking = data_masking
		self.layerwise_optim = layerwise_optim
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
				iters = np.random.randint(2000, 10000) if self.best_params == None else self.best_params['iters']
				iters_.append(iters)
				lr = np.random.uniform(0.0001, 0.005) if self.best_params == None else self.best_params['lr']
				if self.use_beta:
					beta = 1.0 if self.best_params == None else self.best_params['beta']
					lr = 0.0001
				else:
					beta = 0.0
				beta_.append(beta)
				lr_.append(lr)
				layers = np.random.randint(2, 14) if self.best_params == None else self.best_params['layers']
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
					self.net = QMultiNet(input_size=input_size, num_layers=layers,
					                     layers_size=layer_size, output_size=output_size, device=self.device,
					                     output_type=self.output_type, dropout=dropout, use_t=self.use_t,
					                     layerwise_optim=self.layerwise_optim).to(self.device)
				elif self.net_type == 'G':
					self.net = GMultiNet(input_size=input_size, num_layers=layers,
					                     layers_size=layer_size, output_size=output_size, device=self.device,
					                     output_type=self.output_type, dropout=dropout,
					                     layerwise_optim=self.layerwise_optim).to(self.device)

				trainer = MultiNetTrainer(net=self.net, net_type=self.net_type, beta=beta,
				                          outcome_type=self.output_type,
				                          test_loss_plot=self.test_loss_plot, data_masking=self.data_masking,
				                          device=self.device,
				                          iterations=iters, batch_size=bs, test_iter=self.test_iter,
				                          weight_reg=weight_reg,
				                          lr=lr, split=True, layerwise_optim=self.layerwise_optim)
				print('training....')
				train_loss_, val_loss_, stop_it, best_model, best_model_test_loss_, epsilons_, _, _, _ = trainer.train(
					self.x,
					self.y,
					self.z,
					x_pred=self.x_pred)

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
				exc
				print('Error at trial {}:', j)

		tuning_dict = {'batch_size': bs_, 'layers': layers_, 'dropout': dropout_, 'beta': beta_,
		               'layer_size': layer_size_, 'lr': lr_, 'iters': iters_, 'stop_it': stop_it_,
		               'train_loss': train_loss, 'val_loss': val_loss, 'best_model_test_loss': best_losses,
		               'weight_reg': weight_regs}

		if self.net_type == 'G':
			_, best_model_preds, _ = trainer.test(self.best_model, self.x, self.y, self.z)

			beta_bm = trainer.beta_bm
			all_preds = best_model_preds.detach().cpu().numpy()
			x_pred = np.dot(all_preds, beta_bm)
			best_model_preds = torch.tensor(x_pred, dtype=torch.float32).to(self.device)

		else:
			best_model_preds = None
		return tuning_dict, self.best_model, best_model_preds, epsilons_
