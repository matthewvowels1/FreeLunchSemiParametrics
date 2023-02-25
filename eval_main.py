import argparse
import pandas as pd
import numpy as np
import statsmodels.api as sm
from data_gen import sigm, inv_sigm
from submodel_multistep import get_multistep_update

''' This script pulls in the data and outcome and propensity predictions  in order to evaluate the models '''

def one_step(x, y, Q0, Q1, G10):
	D0 = ((1 - x) * (y - Q0)) / (1 - G10) + Q0 - Q0.mean()
	D1 = (x * (y - Q1) / G10) + Q1 - Q1.mean()
	Q1_star = Q1 + D1
	Q0_star = Q0 + D0
	return Q0_star, Q1_star


def submodel(x, y, Q1, Q0, G10, outcome_type):
	Q10 = x*Q1 + (1-x)*Q0

	H1 = (x / (G10)).reshape(-1,1)
	H0 = ((1 - x) / (1 - G10)).reshape(-1,1)

	if outcome_type == 'categorical':
		eps0, eps1 = sm.GLM(y, np.concatenate([H0, H1], 1), offset=inv_sigm(Q10),
		                    family=sm.families.Binomial()).fit(maxiter=1000).params
		Q0_star_solve = sigm(inv_sigm(Q0) + eps0 / (1 - G10))
		Q1_star_solve = sigm(inv_sigm(Q1) + eps1 / G10)
	else:
		eps0, eps1 = sm.GLM(y, np.concatenate([H0, H1], 1), offset=Q10,
		                    family=sm.families.Gaussian()).fit(maxiter=1000).params
		Q0_star_solve = Q0 + (eps0 / (1 - G10))
		Q1_star_solve = Q1 + (eps1 / G10)

	return Q0_star_solve, Q1_star_solve


def evaluator(tau, yhat0, yhat1, tau_hat=None):
	if tau_hat is None:
		tau_hat = yhat1 - yhat0
		eATE = np.mean(np.abs(tau - tau_hat))
		ePEHE = np.sqrt(np.mean((tau - tau_hat)**2))
		ate = np.mean(tau_hat)
		aeATE = np.abs(np.mean(tau) - np.mean(tau_hat))

	else:
		eATE = np.nan
		ePEHE = np.nan
		ate = tau_hat
		aeATE = np.abs(np.mean(tau) - tau_hat)
	return eATE, ePEHE, ate, aeATE

def main(args):
	# important args
	dataset = args.dataset
	run = args.run+str(args.N)+'_'+str(args.iteration)



	outcome_type = 'continuous' if dataset == 'IHDP' else 'categorical'
	# data
	fn = args.fn
	mo_folder = args.fn_mo
	df_z = np.asarray(
		pd.read_csv(fn + str(run) + '_data_z_{}.csv'.format(dataset)).astype(float).values)  # covariates
	df_x = np.asarray(pd.read_csv(fn + str(run) + '_data_x_{}.csv'.format(dataset)).astype(float).values)[:,
	       0]  # treatment
	df_y = np.asarray(pd.read_csv(fn + str(run) + '_data_y_{}.csv'.format(dataset)).astype(float).values)[:,
	       0]  # outcome
	df_y0 = np.asarray(pd.read_csv(fn + str(run) + '_data_y0_{}.csv'.format(dataset)).astype(float).values)[:, 0]  # cf outcome
	df_y1 = np.asarray(pd.read_csv(fn + str(run) + '_data_y1_{}.csv'.format(dataset)).astype(float).values)[:, 0]  # cf outcome

	tau = df_y1 - df_y0


	q_preds = {}
	g_preds = {}
	# load outcome model predictions
	if args.QDML:
		df_qdml = pd.read_csv(mo_folder + str(run) + '_dml_q_{}.csv'.format(dataset))
		dml_coef = df_qdml.coef.values[0]
		dml_se = df_qdml['std err'].values[0]
	if args.QTVAE:
		df_qtvae = pd.read_csv(mo_folder + str(run) + '_tvae_q_{}.csv'.format(dataset))
		q_preds['tvae'] = df_qtvae
	if args.QS:
		df_qs = pd.read_csv(mo_folder + str(run) + '_s_q_{}.csv'.format(dataset))
		q_preds['s_learner'] = df_qs
	if args.QT:
		df_qt = pd.read_csv(mo_folder + str(run) + '_t_q_{}.csv'.format(dataset))
		q_preds['t_learner'] = df_qt
	if args.QD:
		df_qd = pd.read_csv(mo_folder + str(run) + '_d_q_{}.csv'.format(dataset))
		q_preds['d_learner'] = df_qd
		df_qdnotreg = pd.read_csv(mo_folder + str(run) + '_dnotreg_q_{}.csv'.format(dataset))
		q_preds['dnotreg_learner'] = df_qdnotreg
	if args.QLR:
		df_qlr = pd.read_csv(mo_folder + str(run) + '_lr_q_{}.csv'.format(dataset))
		q_preds['lr_learner'] = df_qlr
	if args.QSL:
		df_qsl = pd.read_csv(mo_folder + str(run) + '_sl_q_{}.csv'.format(dataset))
		q_preds['sl_learner'] = df_qsl
	if args.QCFR:
		df_qcfr = pd.read_csv(mo_folder + str(run) + '_cfr_q_{}.csv'.format(dataset))
		q_preds['cfr_learner'] = df_qcfr
	if args.QMN:
		df_qmn = pd.read_csv(mo_folder + str(run) + '_mn_q_{}.csv'.format(dataset))
		q_preds['mn_learner'] = df_qmn

	# load propensity model predictions
	if args.GP:
		df_gp = pd.read_csv(mo_folder + str(run) + '_p_g_{}.csv'.format(dataset))
		g_preds['p_learner'] = df_gp
	if args.GLR:
		df_glr = pd.read_csv(mo_folder + str(run) + '_lr_g_{}.csv'.format(dataset))
		g_preds['lr_learner'] = df_glr
	if args.GCFR:
		df_gcfr = pd.read_csv(mo_folder + str(run) + '_cfr_g_{}.csv'.format(dataset))
		g_preds['cfr_learner'] = df_gcfr
	if args.GMN:
		df_gmn = pd.read_csv(mo_folder + str(run) + '_mn_g_{}.csv'.format(dataset))
		g_preds['mn_learner'] = df_gmn
	if args.GDP:
		df_gdp = pd.read_csv(mo_folder + str(run) + '_d_g_{}.csv'.format(dataset))
		g_preds['dp_learner'] = df_gdp
		df_gdpnotreg = pd.read_csv(mo_folder + str(run) + '_dnotreg_g_{}.csv'.format(dataset))
		g_preds['dpnotreg_learner'] = df_gdpnotreg
	if args.GSL:
		df_gsl = pd.read_csv(mo_folder+ str(run) + '_sl_g_{}.csv'.format(dataset))
		g_preds['sl_learner'] = df_gsl


	# measure intitial estimator performance and save results
	initial_results = {}
	initial_results['true_ate'] = tau.mean()
	for qkey in q_preds.keys():
		eATE, ePEHE, ate, aeATE = evaluator(tau=tau, yhat0=q_preds[qkey]['t=0'], yhat1=q_preds[qkey]['t=1'], tau_hat=None)
		initial_results[qkey] = np.array([eATE, ePEHE, ate, aeATE])


	# integrate updates and save results
	updated_results = {}
	updated_results['true_ate'] = tau.mean()
	print('Updating ', len(q_preds.keys()), ' outcome models and ', len(g_preds.keys()), ' propensity models.')
	for qkey in q_preds.keys():
		for gkey in g_preds.keys():
			try:
				print('Updating outcome model:',qkey, ' with propensity model:', gkey)
				q_pred0, q_pred1 = np.asarray(q_preds[qkey]['t=0'].values), np.asarray(q_preds[qkey]['t=1'].values)
				g_pred = np.asarray(g_preds[gkey].values)[:, 0]
				print('Running submodel update...')
				q_pred0_sub, q_pred1_sub = submodel(x=df_x, y=df_y, Q1=q_pred1, Q0=q_pred0, G10=g_pred,
				                                    outcome_type=outcome_type)
				print('Running one-step update...')
				q_pred0_os, q_pred1_os = one_step(x=df_x, y=df_y, Q1=q_pred1, Q0=q_pred0, G10=g_pred)
				print('Running linear multinet updates...')
				# for these next two, use_y arg is ignored (it only applies to the nonlinear multistep)
				q_pred0_multi_var, q_pred1_multi_var = get_multistep_update(x=df_x, y=df_y, Q1=q_pred1, Q0=q_pred0,
				                                                    G10=g_pred, meanvar=0, type='linear', use_y=1)

				q_pred0_multi_meanvar, q_pred1_multi_meanvar = get_multistep_update(x=df_x, y=df_y, Q1=q_pred1, Q0=q_pred0,
				                                                                    G10=g_pred, meanvar=1, type='linear',
				                                                                    use_y=1)
				print('Running nonlinear multinet update using f(Q,H) and without a mean=0 objective...')
				# for the next four, the use_y arg is important because it determines whether the function takes Q1 and Q0
				q_pred0_multi_nonlin_y_var, q_pred1_multi_nonlin_y_var = get_multistep_update(x=df_x, y=df_y, Q1=q_pred1, Q0=q_pred0,
				                                                    G10=g_pred, meanvar=0, type='nonlinear', use_y=True)
				print('Running nonlinear multinet update using f(H) and without a mean=0 objective...')
				q_pred0_multi_nonlin_noy_var, q_pred1_multi_nonlin_noy_var = get_multistep_update(x=df_x, y=df_y, Q1=q_pred1,
				                                                                     Q0=q_pred0,
				                                                                     G10=g_pred, meanvar=0,
				                                                                     type='nonlinear', use_y=False)
				print('Running nonlinear multinet update using f(Q,H) and with a mean=0 objective...')
				q_pred0_multi_nonlin_y_meanvar, q_pred1_multi_nonlin_y_meanvar = get_multistep_update(x=df_x, y=df_y, Q1=q_pred1,
				                                                                              Q0=q_pred0,
				                                                                              G10=g_pred, meanvar=1,
				                                                                              type='nonlinear', use_y=True)
				print('Running nonlinear multinet update using f(H) and with a mean=0 objective...')
				q_pred0_multi_nonlin_noy_meanvar, q_pred1_multi_nonlin_noy_meanvar = get_multistep_update(x=df_x, y=df_y,
				                                                                                  Q1=q_pred1,
				                                                                                  Q0=q_pred0,
				                                                                                  G10=g_pred, meanvar=1,
				                                                                                  type='nonlinear',
				                                                                                  use_y=False)


				eATE_multi_nonlin_y_var, ePEHE_multi_nonlin_y_var, ate_multi_nonlin_y_var, aeate_multi_nonlin_y_var = evaluator(tau=tau, yhat0=q_pred0_multi_nonlin_y_var,
				                                                           yhat1=q_pred1_multi_nonlin_y_var, tau_hat=None)

				eATE_multi_nonlin_noy_var, ePEHE_multi_nonlin_noy_var, ate_multi_nonlin_noy_var, aeate_multi_nonlin_noy_var = evaluator(tau=tau,
				                                                                                      yhat0=q_pred0_multi_nonlin_noy_var,
				                                                                                      yhat1=q_pred1_multi_nonlin_noy_var, tau_hat=None)

				eATE_multi_nonlin_y_meanvar, ePEHE_multi_nonlin_y_meanvar, ate_multi_nonlin_y_meanvar, aeate_multi_nonlin_y_meanvar = evaluator(tau=tau,
				                                                                                      yhat0=q_pred0_multi_nonlin_y_meanvar,
				                                                                                      yhat1=q_pred1_multi_nonlin_y_meanvar, tau_hat=None)

				eATE_multi_nonlin_noy_meanvar, ePEHE_multi_nonlin_noy_meanvar, ate_multi_nonlin_noy_meanvar, aeate_multi_nonlin_noy_meanvar = evaluator(tau=tau,
				                                                                                            yhat0=q_pred0_multi_nonlin_noy_meanvar,
				                                                                                            yhat1=q_pred1_multi_nonlin_noy_meanvar, tau_hat=None)

				eATE_multi_var, ePEHE_multi_var, ate_multi_var, aeate_multi_var = evaluator(tau=tau, yhat0=q_pred0_multi_var,
				                                                                      yhat1=q_pred1_multi_var, tau_hat=None)
				eATE_multi_meanvar, ePEHE_multi_meanvar, ate_multi_meanvar, aeate_multi_meanvar = evaluator(tau=tau, yhat0=q_pred0_multi_meanvar, yhat1=q_pred1_multi_meanvar, tau_hat=None)
				eATE_sub, ePEHE_sub, ate_sub, aeate_sub = evaluator(tau=tau, yhat0=q_pred0_sub, yhat1=q_pred1_sub, tau_hat=None)
				eATE_os, ePEHE_os, ate_os, aeate_os = evaluator(tau=tau, yhat0=q_pred0_os, yhat1=q_pred1_os, tau_hat=None)

				updated_results[qkey + '_' + gkey + '_multi_nonlin_y_var'] = np.array(
					[eATE_multi_nonlin_y_var, ePEHE_multi_nonlin_y_var, ate_multi_nonlin_y_var, aeate_multi_nonlin_y_var])
				updated_results[qkey + '_' + gkey + '_multi_nonlin_noy_var'] = np.array(
					[eATE_multi_nonlin_noy_var, ePEHE_multi_nonlin_noy_var, ate_multi_nonlin_noy_var, aeate_multi_nonlin_noy_var])
				updated_results[qkey + '_' + gkey + '_multi_nonlin_y_meanvar'] = np.array(
					[eATE_multi_nonlin_y_meanvar, ePEHE_multi_nonlin_y_meanvar, ate_multi_nonlin_y_meanvar, aeate_multi_nonlin_y_meanvar])
				updated_results[qkey + '_' + gkey + '_multi_nonlin_noy_meanvar'] = np.array(
					[eATE_multi_nonlin_noy_meanvar, ePEHE_multi_nonlin_noy_meanvar, ate_multi_nonlin_noy_meanvar, aeate_multi_nonlin_noy_meanvar])

				updated_results[qkey + '_' + gkey + '_multi_linear_var'] = np.array([eATE_multi_var, ePEHE_multi_var, ate_multi_var, aeate_multi_var])
				updated_results[qkey + '_' + gkey + '_multi_linear_meanvar'] = np.array([eATE_multi_meanvar, ePEHE_multi_meanvar, ate_multi_meanvar, aeate_multi_meanvar])
				updated_results[qkey+'_'+gkey+'_submodel'] = np.array([eATE_sub, ePEHE_sub, ate_sub, aeate_sub])
				updated_results[qkey+'_'+gkey+'_onestep'] = np.array([eATE_os, ePEHE_os, ate_os, aeate_os])
			except:
				print('Experienced a problem such as NaNs, skipping method combination and saving as NaNs...')
				eATE_multi_var = ePEHE_multi_var = ate_multi_var = eATE_sub = aeate_multi_nonlin_y_meanvar = ePEHE_sub = aeate_sub = ate_sub = np.nan
				eATE_multi_meanvar = ePEHE_multi_meanvar = ate_multi_meanvar = eATE_os = ePEHE_os = aeate_os= ate_os = np.nan
				eATE_multi_nonlin_noy_meanvar = ePEHE_multi_nonlin_noy_meanvar = ate_multi_nonlin_noy_meanvar = np.nan
				eATE_multi_nonlin_y_meanvar = ePEHE_multi_nonlin_y_meanvar = aeate_multi_var =ate_multi_nonlin_y_meanvar = np.nan
				eATE_multi_nonlin_noy_var = ePEHE_multi_nonlin_noy_var = ate_multi_nonlin_noy_var = aeate_multi_nonlin_noy_meanvar = np.nan
				eATE_multi_nonlin_y_var = ePEHE_multi_nonlin_y_var = aeate_multi_nonlin_y_var = aeate_multi_nonlin_noy_var =ate_multi_nonlin_y_var = aeate_multi_meanvar = np.nan

				updated_results[qkey + '_' + gkey + '_multi_nonlin_y_var'] = np.array(
					[eATE_multi_nonlin_y_var, ePEHE_multi_nonlin_y_var, ate_multi_nonlin_y_var, aeate_multi_nonlin_y_var])
				updated_results[qkey + '_' + gkey + '_multi_nonlin_noy_var'] = np.array(
					[eATE_multi_nonlin_noy_var, ePEHE_multi_nonlin_noy_var, ate_multi_nonlin_noy_var, aeate_multi_nonlin_noy_var])
				updated_results[qkey + '_' + gkey + '_multi_nonlin_y_meanvar'] = np.array(
					[eATE_multi_nonlin_y_meanvar, ePEHE_multi_nonlin_y_meanvar, ate_multi_nonlin_y_meanvar, aeate_multi_nonlin_y_meanvar])
				updated_results[qkey + '_' + gkey + '_multi_nonlin_noy_meanvar'] = np.array(
					[eATE_multi_nonlin_noy_meanvar, ePEHE_multi_nonlin_noy_meanvar, ate_multi_nonlin_noy_meanvar, aeate_multi_nonlin_noy_meanvar])
				updated_results[qkey + '_' + gkey + '_multi_linear_var'] = np.array(
					[eATE_multi_var, ePEHE_multi_var, ate_multi_var, aeate_multi_var])
				updated_results[qkey + '_' + gkey + '_multi_linear_meanvar'] = np.array(
					[eATE_multi_meanvar, ePEHE_multi_meanvar, ate_multi_meanvar, aeate_multi_meanvar])
				updated_results[qkey + '_' + gkey + '_submodel'] = np.array([eATE_sub, ePEHE_sub, ate_sub, aeate_sub])
				updated_results[qkey + '_' + gkey + '_onestep'] = np.array([eATE_os, ePEHE_os, ate_os, aeate_os])

	if args.QDML:
		eATE, ePEHE, ate, aeATE = evaluator(tau=tau, yhat0=None, yhat1=None, tau_hat=dml_coef)
		updated_results['dml_learner'] = np.array([eATE, ePEHE, ate, aeATE])

	print('Saving Results...')
	df_init = pd.DataFrame(initial_results)
	df_init['measurement'] = ['eATE', 'ePEHE', 'ate', 'aeATE']
	df_init.set_index(['measurement'])

	df_updated = pd.DataFrame(updated_results)
	df_updated['measurement'] = ['eATE', 'ePEHE', 'ate', 'aeATE']
	df_updated.set_index(['measurement'])

	df_init.to_csv('results/' + str(run) + '_initial_{}.csv'.format(dataset), index=False)
	df_updated.to_csv('results/' + str(run) + '_updated_{}.csv'.format(dataset), index=False)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="semiparametric_testbed")
	parser.add_argument("--run", default='RUN2', type=str)  # run name (for filenames)
	parser.add_argument("--dataset", default='synth1', type=str)  # 'IHDP', or 'synth1', or 'synth2' or 'general'
	parser.add_argument("--N", default=5000, type=int)  # dataset size (ignored if dataset==IHDP)
	parser.add_argument("--QLR", default=1, type=int)  # Logistic/Linear regression outcome predictions
	parser.add_argument("--QSL", default=1, type=int)  # Super Learner outcome predictions
	parser.add_argument("--QMN", default=1, type=int)  # MultiNet outcome predictions
	parser.add_argument("--QCFR", default=1, type=int)  # CFR outcome predictions
	parser.add_argument("--QS", default=1, type=int)  # S-learner outcome predictions
	parser.add_argument("--QT", default=1, type=int)  # T-learner outcome predictions
	parser.add_argument("--QTVAE", default=1, type=int)  # TVAE outcome predictions
	parser.add_argument("--QDML", default=1, type=int)  # DML results
	parser.add_argument("--QD", default=1, type=int)  # DragonNet outcome predictions
	parser.add_argument("--GMN", default=1, type=int)  # MultiNet propensity scores
	parser.add_argument("--GLR", default=1, type=int)  # Logistic Regression propensity scores
	parser.add_argument("--GSL", default=1, type=int)  # SuperLearner propensity scores
	parser.add_argument("--GCFR", default=1, type=int)  # CFR net propensity scores (basically just a NN)
	parser.add_argument("--GP", default=1, type=int)  # p-model propensity scores
	parser.add_argument("--GDP", default=1, type=int)  # DragonNet propensity scores
	parser.add_argument("--iteration", default='1', type=int)  # iteration number (for filenames and dataset seeds)
	parser.add_argument("--train_sl", default=1, type=int)
	parser.add_argument("--fn", default='data/', type=str)  # filename where the data are stored
	parser.add_argument("--fn_mo", default='model_output/', type=str)  # filename where the model outputs are stored
	args = parser.parse_args()
	main(args)