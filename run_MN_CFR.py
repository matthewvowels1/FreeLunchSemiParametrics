import numpy as np
import pandas as pd
from QGNet import Tuner, QNet, GNet, Trainer, T_scaling
from QGMultiNet import MultiNetTuner, GMultiNet, QMultiNet, MultiNetTrainer
from data_gen import generate_data, sigm, inv_sigm, IHDP
import torch


def run_MN_CFR_G(x, z, y, run_string, network_type='cfr'):
	# network_type = {'cfr', 'mn'}

	# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	device = "cpu"
	use_last_model = True  # whether to use the model with truly the best test loss (=False), or the one which the training loop 'breaked' on (=True)
	data_masking = 1
	layerwise_optim = 0

	x = torch.tensor(x.reshape(-1, 1)).type(torch.float32).to(device)
	z = torch.tensor(z).type(torch.float32).to(device)
	y = torch.tensor(y).type(torch.float32).to(device)

	if network_type == 'mn':
		gtuner = MultiNetTuner(x=x, y=y, z=z, net_type='G', test_iter=100,
		                       output_type='categorical',
		                       trials=15, use_beta=False, use_t=None,
		                       data_masking=data_masking, test_loss_plot=False,
		                       layerwise_optim=layerwise_optim,
		                       device=device)
	else:
		gtuner = Tuner(x=x, y=y, z=z, net_type='G', test_iter=100, calibration=0,
		               device=device, trials=15, use_beta=False, use_t=None,
		               test_loss_plot=False, output_type='categorical')

	print('Tuning MultiNet...')
	gtuning_history, best_g, x_pred, _ = gtuner.tune(run=run_string)

	gtotal_losses = np.asarray(gtuning_history['best_model_test_loss'])
	gbest_index = np.argmin(gtotal_losses)

	gbest_params = {}
	for key in gtuning_history.keys():
		gbest_params[key] = gtuning_history[key][gbest_index]

	print('Best G params:', gbest_params)
	output_type_G = 'categorical'
	output_size_G = 1
	input_size_G = z.shape[-1]
	glayers = gbest_params['layers']
	gdropout = gbest_params['dropout']
	glayer_size = gbest_params['layer_size']
	giters = gbest_params['iters']  # override the early stopping iter (will still use early stopping)
	glr = gbest_params['lr']
	gbatch_size = gbest_params['batch_size']
	gweight_reg = gbest_params['weight_reg']

	print('==============ESTIMATION==============')

	if network_type == 'mn':
		gnet = GMultiNet(input_size=input_size_G, num_layers=glayers, device=device,
		                 layers_size=glayer_size, output_size=output_size_G,
		                 output_type=output_type_G, dropout=gdropout,
		                 layerwise_optim=layerwise_optim).to(device)

		print('Training G....')
		gtrainer = MultiNetTrainer(net=gnet, net_type='G', beta=0.0, iterations=giters,
		                           device=device, outcome_type=output_type_G, data_masking=data_masking,
		                           batch_size=gbatch_size, test_iter=1000, lr=glr,
		                           weight_reg=gweight_reg, test_loss_plot=False, split=False,
		                           layerwise_optim=layerwise_optim)

		train_loss_g_, val_loss_g_, stop_it_g, best_model_g, best_model_test_loss_g, eps, last_modelg, gbetas_bm, gbetas_lm = gtrainer.train(
			x, y, z)

	else:
		gnet = GNet(input_size=input_size_G, num_layers=glayers, device=device,
		            layers_size=glayer_size, output_size=output_size_G,
		            output_type=output_type_G, dropout=gdropout).to(device)
		print('Training G....')
		# def G trainer
		gtrainer = Trainer(net=gnet, net_type='G', beta=0.0, iterations=giters,
		                   outcome_type=output_type_G,
		                   batch_size=gbatch_size, test_iter=1000, lr=glr, weight_reg=gweight_reg,
		                   device=device, test_loss_plot=False, calibration=False, split=False)

		train_loss_g_, val_loss_g_, stop_it_g, best_model_g, best_model_test_loss_g, eps, last_modelg, gtemp_bm, gtemp_lm = gtrainer.train(
			x, y, z)

	if use_last_model:
		best_model_g = last_modelg
		gbetas_bm = gbetas_lm if network_type == 'mn' else None

	# Get x_preds from G
	_, x_pred, x_logits = gtrainer.test(best_model_g, x, y, z)

	x_pred = x_pred.detach().cpu().numpy()
	x_pred = np.dot(x_pred, gbetas_bm) if network_type == 'mn' else x_pred
	x_pred = np.clip(x_pred, a_min=0.025, a_max=0.975)

	return x_pred


def run_MN_CFR_Q(x, z, y, run_string, output_type='continuous', network_type='cfr'):
	# network_type = {'cfr', 'mn'}

	# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	device = "cpu"
	use_last_model = True  # whether to use the model with truly the best test loss (=False), or the one which the training loop 'breaked' on (=True)
	data_masking = 1
	layerwise_optim = 0

	x = torch.tensor(x.reshape(-1, 1)).type(torch.float32).to(device)
	z = torch.tensor(z).type(torch.float32).to(device)
	y = torch.tensor(y.reshape(-1, 1)).type(torch.float32).to(device)
	x_int1 = torch.ones_like(x).to(device)  # this is the 'intervention data'
	x_int0 = torch.zeros_like(x).to(device)

	trials = 15

	if network_type == 'mn':
		qtuner = MultiNetTuner(x=x, y=y, z=z, net_type='Q', test_iter=100,
		                        data_masking=data_masking, device=device,
		                        trials=trials, use_beta=False, use_t=True,
		                        test_loss_plot=False, layerwise_optim=layerwise_optim,
		                        output_type=output_type)
	else:
		qtuner = Tuner(x=x, y=y, z=z, net_type='Q', test_iter=100,
		               calibration=0, device=device,
		               trials=trials, use_beta=False, use_t=True,
		               test_loss_plot=False, output_type=output_type)

	print('Tuning MultiNet...')
	qtuning_history, best_q, _, eps = qtuner.tune(run=run_string)

	qtotal_losses = np.asarray(qtuning_history['best_model_test_loss'])
	qbest_index = np.argmin(qtotal_losses)

	qbest_params = {}
	for key in qtuning_history.keys():
		qbest_params[key] = qtuning_history[key][qbest_index]

	print('Best Q params:', qbest_params)
	output_size_Q = 1
	input_size_Q = z.shape[-1] + 1  # we will concatenate the treatment var inside the qnet class
	qlayers = qbest_params['layers']
	qdropout = qbest_params['dropout']
	qlayer_size = qbest_params['layer_size']
	qiters = qbest_params['iters']  # override the early stopping iter (will still use early stopping)
	qlr = qbest_params['lr']
	qbatch_size = qbest_params['batch_size']
	qweight_reg = qbest_params['weight_reg']

	print('==============ESTIMATION==============')
	num_trys = 20
	k = 0
	while k <= num_trys:
		try:
			if network_type == 'mn':

				qnet = QMultiNet(input_size=input_size_Q, num_layers=qlayers, device=device,
				                 layers_size=qlayer_size, output_size=output_size_Q,
				                 output_type=output_type, dropout=qdropout, use_t=True,
				                 layerwise_optim=layerwise_optim).to(device)
				print('Training Q....')
				qtrainer = MultiNetTrainer(net=qnet, net_type='Q', beta=0.0, iterations=qiters,
				                           outcome_type=output_type, device=device,
				                           batch_size=qbatch_size, test_iter=1000, lr=qlr,
				                           weight_reg=qweight_reg, data_masking=data_masking,
				                           test_loss_plot=False, split=False,
				                           layerwise_optim=layerwise_optim)
				train_loss_q_, val_loss_q_, stop_it_q, best_model_q, best_model_test_loss_q, eps, last_modelq, qbetas_bm, qbetas_lm = qtrainer.train(
					x, y, z)
			else:
				qnet = QNet(input_size=input_size_Q, num_layers=qlayers, device=device,
				            layers_size=qlayer_size, output_size=output_size_Q,
				            output_type=output_type, dropout=qdropout, use_t=True).to(device)

				print('Training Q....')
				qtrainer = Trainer(net=qnet, net_type='Q', beta=0.0, iterations=qiters,
				                   outcome_type=output_type, batch_size=qbatch_size, test_iter=1000, lr=qlr,
				                   weight_reg=qweight_reg, device=device, test_loss_plot=False, calibration=False, split=False)

				train_loss_q_, val_loss_q_, stop_it_q, best_model_q, best_model_test_loss_q, eps, last_modelq, qtemp_bm, qtemp_lm = qtrainer.train(
					x, y, z)

			if use_last_model:
				best_model_q = last_modelq
				qbetas_bm = qbetas_lm if network_type == 'mn' else None

			_, Q1, _ = qtrainer.test(best_model_q, x_int1, y, z, None)
			_, Q0, _ = qtrainer.test(best_model_q, x_int0, y, z, None)
			Q1 = Q1.detach().cpu().numpy()
			Q0 = Q0.detach().cpu().numpy()
			Q1 = np.dot(Q1, qbetas_bm) if network_type == 'mn' else Q1
			Q0 = np.dot(Q0, qbetas_bm) if network_type == 'mn' else Q0
			break
		except:
			k += 1
			print('Experienced an error at attempt {}, trying to train the model again...'.format(k))

	return Q0, Q1
