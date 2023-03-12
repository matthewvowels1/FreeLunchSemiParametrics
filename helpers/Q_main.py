import argparse
import pandas as pd
from data_gen import sigm
import numpy as np
from lightgbm import LGBMRegressor, LGBMClassifier
from causalml.inference.tf.utils import dragonnet_loss_binarycross_cat, dragonnet_loss_binarycross
from causalml.inference.meta import BaseXRegressor, BaseSRegressor, BaseTRegressor, BaseXClassifier, BaseSClassifier, BaseTClassifier
from causalml.inference.tf import DragonNet
from causalml.propensity import ElasticNetPropensityModel
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, RandomForestRegressor, AdaBoostRegressor, StackingClassifier, StackingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from super_learner import SuperLearner
import doubleml as dml
''' This script pulls in the data and generates predictions for the outcomes as well as two propensity score predictions
(DragonNet and p-learner)'''


def init_super_dict(output_type):
    if output_type == 'categorical':
        est_dict = {'LR': LogisticRegression(max_iter=1000), 'SVC': SVC(probability=True),
                    'RF': RandomForestClassifier(), 'KNN': KNeighborsClassifier(),
                    'AB': AdaBoostClassifier(), 'poly': 'poly'}


    else:
        est_dict = {'LR': LinearRegression(), 'SVR': SVR(),
                    'RF': RandomForestRegressor(), 'KNN': KNeighborsRegressor(),
                    'AB': AdaBoostRegressor(), 'poly': 'poly'}

    dml_dict_cls = {'LR': LogisticRegression(max_iter=1000), 'SVC': SVC(probability=True),
                    'RF': RandomForestClassifier(), 'KNN': KNeighborsClassifier(),
                    'AB': AdaBoostClassifier()}
    dml_dict_reg = {'LR': LinearRegression(), 'SVR': SVR(),
                    'RF': RandomForestRegressor(), 'KNN': KNeighborsRegressor(),
                    'AB': AdaBoostRegressor()}

    return est_dict, dml_dict_cls, dml_dict_reg


def main(args):
    # important args
    dataset = args.dataset
    run = args.run+str(args.N)+'_'+str(args.iteration)

    # data
    fn = args.fn
    df_z = np.asarray(pd.read_csv(fn + str(run) + '_data_z_{}.csv'.format(dataset)).astype(float).values)  # covariates
    df_x = np.asarray(pd.read_csv(fn + str(run) + '_data_x_{}.csv'.format(dataset)).astype(float).values)[:, 0]  # treatment
    df_y = np.asarray(pd.read_csv(fn + str(run) + '_data_y_{}.csv'.format(dataset)).astype(float).values)[:, 0]  # outcome

    print(df_x.shape, df_z.shape, df_x.dtype, df_z.dtype)
    print('Fitting P treatment model...')
    p_model = ElasticNetPropensityModel(max_iter=1000)
    p = np.clip(p_model.fit_predict(X=df_z, y=df_x), a_min=0.025, a_max=0.975)
    df_results_p = pd.DataFrame(p)
    df_results_p.to_csv('model_output/' + str(run) + '_p_g_{}.csv'.format(dataset), index=False)

    # init models (dragonnet initalized below)
    if dataset == 'IHDP':
        sl_dict, dml_dict_cls, dml_dict_reg = init_super_dict('continuous')
        sl_learner = SuperLearner('reg', est_dict=sl_dict, k=10)
        lr_learner = LinearRegression()
        d_loss_func = dragonnet_loss_binarycross
        s_learner = BaseSRegressor(LGBMRegressor())
        t_learner = BaseTRegressor(LGBMRegressor())
        x_learner = BaseXRegressor(LGBMRegressor())

    elif dataset == 'synth1' or dataset == 'synth2' or dataset == 'general':
        sl_dict, dml_dict_cls, dml_dict_reg = init_super_dict('categorical')
        sl_learner = SuperLearner('cls', est_dict=sl_dict, k=10)
        lr_learner = LogisticRegression(max_iter=1000)
        d_loss_func = dragonnet_loss_binarycross_cat
        s_learner = BaseSClassifier(LGBMClassifier())
        t_learner = BaseTClassifier(LGBMClassifier())
        x_learner = BaseXClassifier(outcome_learner=LGBMClassifier(), effect_learner=LGBMRegressor())

    dml_estimators_cls = [(b, dml_dict_cls[b]) for a, b in enumerate(dml_dict_cls)]
    dml_estimators_reg = [(b, dml_dict_reg[b]) for a, b in enumerate(dml_dict_reg)]
    dml_cls = StackingClassifier(estimators=dml_estimators_cls, final_estimator=LogisticRegression(max_iter=1000), stack_method='predict_proba')
    dml_reg = StackingRegressor(estimators=dml_estimators_reg, final_estimator=LinearRegression())


    # train models and collect and store outputs
    if args.train_sllearner:
        print('Fitting SL outcome model...')
        x_int1 = np.ones_like(df_x)
        x_int0 = np.zeros_like(df_x)
        sl_learner.train_combiner(np.concatenate([df_x.reshape(-1, 1), df_z], 1), df_y)
        sl_learner.train_superlearner(np.concatenate([df_x.reshape(-1, 1), df_z], 1), df_y)
        sl_0 = sl_learner.estimation(np.concatenate([x_int0.reshape(-1, 1), df_z], 1), df_y)
        sl_1 = sl_learner.estimation(np.concatenate([x_int1.reshape(-1, 1), df_z], 1), df_y)
        df_results_sl = pd.DataFrame(np.concatenate([sl_0, sl_1], 1), columns=['t=0', 't=1'])
        df_results_sl.to_csv('model_output/' + str(run) + '_sl_q_{}.csv'.format(dataset), index=False)

    if args.train_dmllearner:
        print('Fitting DML complete model')
        pdxy = pd.DataFrame([df_x, df_y]).T
        pdz = pd.DataFrame(df_z)
        cols = ['x', 'y']
        z_cols = ['z_{}'.format(i) for i in range(df_z.shape[1])]
        cols += z_cols
        model_data = pd.concat([pdxy.reset_index(drop=True), pdz.reset_index(drop=True)], axis=1)
        model_data.columns = cols
        data_dml = dml.DoubleMLData(model_data, y_col='y', d_cols='x')

        if dataset == 'IHDP':
            dml_learner = dml.DoubleMLIRM(data_dml,
                                         ml_g=dml_reg,  # models E[Outcome|Treat,Covs]
                                         ml_m=dml_cls,   # models E[Treat|Covs]
                                         # note that these inputs use different terminologies (M for nuisance and G for outcome)
                                         trimming_threshold=0.025,
                                         n_folds=5, score='ATE')

        elif dataset == 'synth1' or dataset == 'synth2' or dataset == 'general':
            dml_learner = dml.DoubleMLIRM(data_dml,
                                         ml_g=dml_cls,  # models E[Outcome|Treat,Covs]
                                         ml_m=dml_cls,   # models E[Treat|Covs]
                                         # note that these inputs use different terminologies (M for nuisance and G for outcome)
                                         trimming_threshold=0.025,
                                         n_folds=5, score='ATE')

        dml_learner.fit(store_predictions=True)
        df_results_dml = pd.DataFrame(dml_learner.summary)
        df_results_dml.to_csv('model_output/' + str(run) + '_dml_q_{}.csv'.format(dataset), index=False)


    if args.train_lrlearner:
        print('Fitting LR outcome model...')
        x_int1 = np.ones_like(df_x)
        x_int0 = np.zeros_like(df_x)
        lr_learner.fit(np.concatenate([df_x.reshape(-1, 1), df_z], 1), df_y)
        lr_0 = lr_learner.predict(np.concatenate([x_int0.reshape(-1, 1), df_z], 1)).reshape(-1, 1) if dataset == 'IHDP' else lr_learner.predict_proba(np.concatenate([x_int0.reshape(-1, 1), df_z], 1))[:, 1:]
        lr_1 = lr_learner.predict(np.concatenate([x_int1.reshape(-1, 1), df_z], 1)).reshape(-1, 1) if dataset == 'IHDP' else lr_learner.predict_proba(np.concatenate([x_int1.reshape(-1, 1), df_z], 1))[:, 1:]
        df_results_lr = pd.DataFrame(np.concatenate([lr_0, lr_1], 1), columns=['t=0', 't=1'])
        df_results_lr.to_csv('model_output/' + str(run) + '_lr_q_{}.csv'.format(dataset), index=False)

    if args.train_slearner:
        print('Fitting S-learner outcome model...')
        s_ate = s_learner.estimate_ate(df_z, df_x, df_y)[0]
        s_components = s_learner.predict(df_z, df_x, df_y, return_components=True)
        s_0 = list(s_components[1].values())[0].reshape(-1, 1)
        s_1 = list(s_components[2].values())[0].reshape(-1, 1)
        df_results_s = pd.DataFrame(np.concatenate([s_0, s_1], 1), columns=['t=0', 't=1'])
        df_results_s.to_csv('model_output/' + str(run) + '_s_q_{}.csv'.format(dataset), index=False)

    if args.train_tlearner:
        print('Fitting T-learner outcome model...')
        t_ate = t_learner.estimate_ate(df_z, df_x, df_y)[0][0]
        t_components = t_learner.predict(df_z, df_x, df_y, return_components=True)
        t_0 = list(t_components[1].values())[0].reshape(-1, 1)
        t_1 = list(t_components[2].values())[0].reshape(-1, 1)
        df_results_t = pd.DataFrame(np.concatenate([t_0, t_1], 1), columns=['t=0', 't=1'])
        df_results_t.to_csv('model_output/' + str(run) + '_t_q_{}.csv'.format(dataset), index=False)


    if args.train_dragon:
        print('Fitting Dragon-Learner outcome model with treg...')
        dragon = DragonNet(neurons_per_layer=200, targeted_reg=True, loss_func=d_loss_func, ratio=1., val_split=0.2,
                           batch_size=64, epochs=100, learning_rate=1e-4, reg_l2=0.01,)
        dragon.fit(df_z, df_x, df_y)
        d = dragon.predict(df_z)
        d_0 = d[:, 0].reshape(-1, 1) if dataset == 'IHDP' else sigm(d[:, 0].reshape(-1, 1))
        d_1 = d[:, 1].reshape(-1, 1) if dataset == 'IHDP' else sigm(d[:, 1].reshape(-1, 1))
        dp = np.clip(d[:, 2].reshape(-1, 1), a_min=0.025, a_max=0.975)
        df_results_d = pd.DataFrame(np.concatenate([d_0, d_1], 1), columns=['t=0', 't=1'])
        df_results_d.to_csv('model_output/' + str(run) + '_d_q_{}.csv'.format(dataset), index=False)
        df_results_dp = pd.DataFrame(dp)
        df_results_dp.to_csv('model_output/' + str(run) + '_d_g_{}.csv'.format(dataset), index=False)
        print(dp)
        print('Fitting Dragon-Learner outcome model without treg...')
        dragon = DragonNet(neurons_per_layer=200, targeted_reg=False, loss_func=d_loss_func, ratio=1.,
                           val_split=0.2, batch_size=64, epochs=50, learning_rate=1e-5, reg_l2=0.01, )
        dragon.fit(df_z, df_x, df_y)
        d = dragon.predict(df_z)
        d_0 = d[:, 0].reshape(-1, 1) if dataset == 'IHDP' else sigm(d[:, 0].reshape(-1, 1))
        d_1 = d[:, 1].reshape(-1, 1) if dataset == 'IHDP' else sigm(d[:, 1].reshape(-1, 1))
        dp = np.clip(d[:, 2].reshape(-1, 1), a_min=0.025, a_max=0.975)
        df_results_d = pd.DataFrame(np.concatenate([d_0, d_1], 1), columns=['t=0', 't=1'])
        df_results_d.to_csv('model_output/' + str(run) + '_dnotreg_q_{}.csv'.format(dataset), index=False)
        df_results_dp = pd.DataFrame(dp)
        df_results_dp.to_csv('model_output/' + str(run) + '_dnotreg_g_{}.csv'.format(dataset), index=False)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="semiparametric_testbed")
    parser.add_argument("--run", default='RUN2', type=str)  # run name (for filenames)
    parser.add_argument("--train_dragon", default=1, type=int)
    parser.add_argument("--train_lrlearner", default=1, type=int)
    parser.add_argument("--train_slearner", default=1, type=int)
    parser.add_argument("--train_tlearner", default=1, type=int)
    parser.add_argument("--train_dmllearner", default=1, type=int)
    parser.add_argument("--train_sllearner", default=1, type=int)
    parser.add_argument("--N", default=5000, type=int)  # dataset size (ignored if dataset==IHDP)
    parser.add_argument("--dataset", default='synth1', type=str)  # 'IHDP', or 'synth1', or 'synth2'
    parser.add_argument("--iteration", default='1', type=int)  # iteration number (for filenames and dataset seeds)
    parser.add_argument("--fn", default='data/', type=str)
    args = parser.parse_args()
    main(args)