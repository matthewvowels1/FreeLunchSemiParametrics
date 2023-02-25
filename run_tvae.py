

import torch
import argparse
import pyro
from TVAE.TVAE_wrapper import TVAE

def run_tvae(x, z, y, dataset, run_string):
	# network_type = {'cfr', 'mn'}

	# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	device = "cpu"

	x = torch.tensor(x).type(torch.float32).to(device)  # treatment
	z = torch.tensor(z).type(torch.float32).to(device)  # covariates
	y = torch.tensor(y).type(torch.float32).to(device)   # outcome


	# main_TVAEsynth.py - -tl_weight	0.1 - -latent_dim_o	1 - -latent_dim_c	2 - -latent_dim_t	2 - -latent_dim_y	2 - -hidden_dim	20 - -num_layers	2 - -num_epochs	40 - -batch_size	200 - -learning_rate	0.0005 - -reps	100

	latent_dim_o = 5
	latent_dim_c = 5
	latent_dim_t = 5
	latent_dim_y = 5
	hidden_dim = 20
	num_layers = 2
	num_epochs = 100
	batch_size = 200
	learning_rate = 0.0005
	learning_rate_decay = 0.01
	weight_decay = 1e-4
	tl_weight = 0.1


	if dataset == 'IHDP':
		outcome_dist = 'normal'
		binfeats = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
		contfeats = [i for i in range(25) if i not in binfeats]
		ym, ys = y.mean(), y.std()
		y = (y - ym) / ys
	elif dataset == 'synth1' or dataset == 'synth2':
		outcome_dist = 'bernoulli'
		binfeats = [0, 1]
		contfeats = [2, 3]
		ym, ys = 0.0, 1.0
	elif dataset == 'general':
		outcome_dist = 'bernoulli'
		binfeats = [0, 1]
		contfeats = [2, 3, 4]
		ym, ys = 0.0, 1.0


	# Train
	pyro.clear_param_store()
	tvae = TVAE(feature_dim=z.shape[1], continuous_dim=contfeats, binary_dim=binfeats,
	            outcome_dist=outcome_dist, device=device,
	            latent_dim_o=latent_dim_o, latent_dim_c=latent_dim_c, latent_dim_t=latent_dim_t,
	            latent_dim_y=latent_dim_y,
	            hidden_dim=hidden_dim,
	            num_layers=num_layers,
	            num_samples=100)
	tvae.fit(z, x, y,
	         num_epochs=num_epochs,
	         batch_size=batch_size,
	         learning_rate=learning_rate,
	         learning_rate_decay=learning_rate_decay,
	         weight_decay=weight_decay,
	         treg_weight=tl_weight)

	# Evaluate.
	yhat_0, yhat_1 = tvae.preds(z, ym, ys)
	return yhat_0.reshape(-1,1), yhat_1.reshape(-1,1)