import argparse
import pandas as pd
import numpy as np
from run_MN_CFR import run_MN_CFR_Q
from run_tvae import run_tvae
''' This script pulls in the data and generates predictions for the outcomes as well as two propensity score predictions'''


def main(args):
	# important args
	dataset = args.dataset
	run = args.run+str(args.N)+'_'+str(args.iteration)

	# data
	fn = args.fn
	df_z = np.asarray(pd.read_csv(fn + str(run) + '_data_z_{}.csv'.format(dataset)).astype(float).values)  # covariates
	df_x = np.asarray(pd.read_csv(fn + str(run) + '_data_x_{}.csv'.format(dataset)).astype(float).values)[:, 0]  # treatment
	df_y = np.asarray(pd.read_csv(fn + str(run) + '_data_y_{}.csv'.format(dataset)).astype(float).values)[:, 0]  # outcome

	if dataset == 'IHDP':
		output_type = 'continuous'
	elif dataset == 'synth1' or dataset == 'synth2' or dataset == 'general':
		output_type = 'categorical'

	if args.train_mnlearner:
		print('Training MultiNet outcome model...')
		mn_0, mn_1 = run_MN_CFR_Q(x=df_x, z=df_z, y=df_y, run_string=str(run),
		                          output_type=output_type, network_type='mn')
		df_results_mn = pd.DataFrame(np.concatenate([mn_0, mn_1], 1), columns=['t=0', 't=1'])
		df_results_mn.to_csv('model_output/' + str(run) + '_mn_q_{}.csv'.format(dataset), index=False)

	if args.train_cfrlearner:
		print('Training CFR outcome model...')
		cfr_0, cfr_1 = run_MN_CFR_Q(x=df_x, z=df_z, y=df_y, run_string=str(run),
		                          output_type=output_type, network_type='cfr')
		df_results_cfr = pd.DataFrame(np.concatenate([cfr_0, cfr_1], 1), columns=['t=0', 't=1'])
		df_results_cfr.to_csv('model_output/' + str(run) + '_cfr_q_{}.csv'.format(dataset), index=False)

	if args.train_tvae:
		print('Training TVAE outcome model...')
		tvae_0, tvae_1 = run_tvae(x=df_x, z=df_z, y=df_y, dataset=dataset, run_string=str(run))
		df_results_tvae = pd.DataFrame(np.concatenate([tvae_0, tvae_1], 1), columns=['t=0', 't=1'])
		df_results_tvae.to_csv('model_output/' + str(run) + '_tvae_q_{}.csv'.format(dataset), index=False)



if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="semiparametric_testbed")
	parser.add_argument("--run", default='RUN2', type=str)  # run name (for filenames)
	parser.add_argument("--train_mnlearner", default=1, type=int)
	parser.add_argument("--train_cfrlearner", default=1, type=int)
	parser.add_argument("--train_tvae", default=1, type=int)
	parser.add_argument("--N", default=5000, type=int)  # dataset size (ignored if dataset==IHDP)
	parser.add_argument("--dataset", default='synth1', type=str)  # 'IHDP', or 'synth1', or 'synth2' or 'general'
	parser.add_argument("--iteration", default='1', type=int)  # iteration number (for filenames and dataset seeds)
	parser.add_argument("--fn", default='data/', type=str)
	args = parser.parse_args()
	main(args)