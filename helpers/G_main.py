import argparse
import pandas as pd
import numpy as np
from run_MN_CFR import run_MN_CFR_G
from super_learner import SuperLearner
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, RandomForestRegressor, AdaBoostRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR

''' This script pulls in the data and generates predictions propensity scores'''

def init_super_dict(output_type):
	if output_type == 'categorical':
		est_dict = {'LR': LogisticRegression(max_iter=1000), 'SVC': SVC(probability=True),
		            'RF': RandomForestClassifier(), 'KNN': KNeighborsClassifier(),
		            'AB': AdaBoostClassifier(), 'poly': 'poly'}
	else:
		est_dict = {'LR': LinearRegression(), 'SVR': SVR(),
		            'RF': RandomForestRegressor(), 'KNN': KNeighborsRegressor(),
		            'AB': AdaBoostRegressor(), 'poly': 'poly'}
	return est_dict


def main(args):
	# important args
	dataset = args.dataset
	run = args.run+str(args.N)+'_'+str(args.iteration)

	# data
	fn = args.fn
	df_z = np.asarray(pd.read_csv(fn + str(run) + '_data_z_{}.csv'.format(dataset)).astype(float).values)  # covariates
	df_x = np.asarray(pd.read_csv(fn + str(run) + '_data_x_{}.csv'.format(dataset)).astype(float).values)[:, 0]  # treatment
	df_y = np.asarray(pd.read_csv(fn + str(run) + '_data_y_{}.csv'.format(dataset)).astype(float).values)[:, 0]  # outcome

	if args.train_lr:
		print('Training LR treatment model...')
		GLR = LogisticRegression(max_iter=1000).fit(df_z, df_x)
		lr_p = np.clip(GLR.predict_proba(df_z), a_min=0.025, a_max=0.975)[:, 1:]
		df_results_lr_p = pd.DataFrame(lr_p)
		df_results_lr_p.to_csv('model_output/' + str(run) + '_lr_g_{}.csv'.format(dataset), index=False)

	if args.train_sl:
		print('Training SL treatment model...')
		Gest_dict = init_super_dict('categorical')
		GSL = SuperLearner(output='cls', est_dict=Gest_dict, k=10)
		GSL.train_combiner(df_z, df_x)
		GSL.train_superlearner(df_z, df_x)
		sl_p = np.clip(GSL.estimation(df_z, df_x), a_min=0.025, a_max=0.975)
		df_results_sl_p = pd.DataFrame(sl_p)
		df_results_sl_p.to_csv('model_output/' + str(run) + '_sl_g_{}.csv'.format(dataset), index=False)

	if args.train_mn:
		print('Training MultiNet treatment model...')
		mn_p = run_MN_CFR_G(x=df_x,  z=df_z, y=df_y, run_string=str(run), network_type='mn')
		df_results_mn_p = pd.DataFrame(mn_p)
		df_results_mn_p.to_csv('model_output/' + str(run) + '_mn_g_{}.csv'.format(dataset), index=False)

	if args.train_cfr:
		print('Training MultiNet treatment model...')
		cfr_p = run_MN_CFR_G(x=df_x, z=df_z, y=df_y, run_string=str(run), network_type='cfr')
		df_results_cfr_p = pd.DataFrame(cfr_p)
		df_results_cfr_p.to_csv('model_output/' + str(run) + '_cfr_g_{}.csv'.format(dataset), index=False)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="semiparametric_testbed")
	parser.add_argument("--run", default='RUN2', type=str)  # run name (for filenames)
	parser.add_argument("--dataset", default='synth1', type=str)  # 'IHDP', or 'synth1', or 'synth2'
	parser.add_argument("--train_mn", default=1, type=int)
	parser.add_argument("--train_lr", default=1, type=int)
	parser.add_argument("--train_sl", default=1, type=int)
	parser.add_argument("--train_cfr", default=1, type=int)
	parser.add_argument("--N", default=5000, type=int)  # dataset size (ignored if dataset==IHDP)
	parser.add_argument("--iteration", default='1', type=int)  # iteration number (for filenames and dataset seeds)
	parser.add_argument("--fn", default='data/', type=str)
	args = parser.parse_args()
	main(args)