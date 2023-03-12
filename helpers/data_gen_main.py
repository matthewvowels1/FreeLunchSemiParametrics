from data_gen import generate_data, IHDP
import argparse
import os
import pandas as pd

''' This script creates temporary data csvs for use by the different models.'''

def main(args):

	run = args.run + str(args.N) + '_' + str(args.iteration)
	iter = args.iteration
	dataset = args.dataset
	N = args.N
	fn = os.path.join(args.fn, run + '_data_')

	os.makedirs('../data', exist_ok=True)
	os.makedirs('../model_output', exist_ok=True)
	os.makedirs('../results', exist_ok=True)

	if dataset == 'IHDP':
		z, x, y, y1, y0 = IHDP(seed=iter)
	else:
		z, x, y, y1, y0 = generate_data(N=N, seed=iter, dataset=dataset)

	df_z = pd.DataFrame(z)
	df_z.to_csv(fn+'z_{}.csv'.format(dataset), index=False)

	df_x = pd.DataFrame(x)
	df_x.to_csv(fn + 'x_{}.csv'.format(dataset), index=False)

	df_y = pd.DataFrame(y)
	df_y.to_csv(fn + 'y_{}.csv'.format(dataset), index=False)

	df_y1 = pd.DataFrame(y1)
	df_y1.to_csv(fn + 'y1_{}.csv'.format(dataset), index=False)

	df_y0 = pd.DataFrame(y0)
	df_y0.to_csv(fn + 'y0_{}.csv'.format(dataset), index=False)



if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="semiparametric_testbed")
	parser.add_argument("--run", default='RUN2', type=str)  # run name (for filenames)
	parser.add_argument("--iteration", default='1', type=int)  # iteration number (for filenames and dataset seeds)
	parser.add_argument("--N", default=5000, type=int)  # dataset size (ignored if dataset==IHDP)
	parser.add_argument("--dataset", default='synth1', type=str)  # 'IHDP', or 'synth1', or 'synth2' or  'general'
	parser.add_argument("--fn", default='data/', type=str)
	args = parser.parse_args()
	main(args)