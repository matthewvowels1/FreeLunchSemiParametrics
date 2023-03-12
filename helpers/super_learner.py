import numpy as np
from sklearn.model_selection import KFold
from scipy.optimize import minimize
import scipy
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures


def fn(x, A, b):
	return np.linalg.norm(A.dot(x) - b)


''' An example dictionary of estimators can be specified as follows:
Gest_dict = {'LR': LogisticRegression(), 'SVC': SVC(probability=True),
                                 'RF': RandomForestClassifier(), 'KNN': KNeighborsClassifier(),
                                 'AB': AdaBoostClassifier(), 'poly': 'poly'}

Note that 'poly' is specified as a string because there are no default polynomial feature regressors in sklearn.
This one defaults to 2nd order features (e.g. x1*x2, x1*x3 etc...)'''


def combiner_solve(x, y):
	# adapted from https://stackoverflow.com/questions/33385898/how-to-include-constraint-to-scipy-nnls-function-solution-so-that-it-sums-to-1/33388181
	beta_0, rnorm = scipy.optimize.nnls(x, y, maxiter=None)
	cons = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
	bounds = [[0.0, None]] * x.shape[1]
	minout = minimize(fn, beta_0, args=(x, y), method='SLSQP', bounds=bounds, constraints=cons)
	beta = minout.x
	return beta


class SuperLearner(object):
	def __init__(self, output, est_dict, k):
		self.output = output
		self.k = k  # number of cross validation folds
		self.beta = None
		self.trained_superlearner = None
		self.output = output
		self.est_dict = est_dict  # dictionary of learners/algos
		self.x_min = None
		self.x_max = None
		self.x_max_sq = None

	def train_combiner(self, x, y):

		self.x_min = x.min()
		self.x_max = x.max()
		self.x_max_sq = x.max() ** 2

		if (self.output == 'cls') or (self.output == 'proba'):
			num_classes = np.unique(y)
			if len(num_classes) == 2:
				num_classes = 1
			elif len(num_classes) > 2:
				num_classes = len(num_classes)
		else:
			num_classes = 1

		all_preds = np.zeros((len(y), num_classes, len(self.est_dict)))

		i = 0
		for key in self.est_dict.keys():

			kf = KFold(n_splits=self.k, shuffle=True, random_state=0)

			if self.output == 'proba' or (self.output == 'cls'):
				probs = []
			preds = []
			gts = []

			for train_index, test_index in kf.split(x):
				x_train = x[train_index]
				x_test = x[test_index]
				y_train = y[train_index]
				y_test = y[test_index]

				est = self.est_dict[key]
				if key == 'poly':
					est = LogisticRegression(C=1e2, max_iter=350) if (self.output == 'cls') or (
								self.output == 'proba') else LinearRegression()
					poly = PolynomialFeatures(2)
					x_train = poly.fit_transform(x_train)
					x_test = poly.fit_transform(x_test)

				est.fit(x_train, y_train)

				if self.output == 'proba' or (self.output == 'cls'):
					p = est.predict(x_test)
					p_robs = est.predict_proba(x_test)
					preds.append(p)
					probs.append(p_robs)
				elif (self.output == 'reg'):
					p = est.predict(x_test)
					preds.append(p)
				gts.append(y_test)

			preds = np.concatenate(preds)
			if self.output == 'proba' or (self.output == 'cls'):
				probs = np.concatenate(probs)
			if num_classes == 1:
				if self.output == 'proba' or (self.output == 'cls'):
					probs = probs[:, 1].reshape(-1, 1)
				preds = preds.reshape(-1, 1)

			gts = np.concatenate(gts)

			if (self.output == 'cls') or (self.output == 'proba'):
				all_preds[:, :, i] = probs
			elif self.output == 'reg':
				all_preds[:, :, i] = preds

			i += 1

		beta = combiner_solve(all_preds[:, 0, :], gts)  # all_preds is of shape [batch, categories, predictors]
		self.beta = beta
		return beta

	def train_superlearner(self, x, y):
		assert self.beta is not None, 'Train combiner first using SuperLearner.train_combiner(x,y)'
		# now we have the coefficients we can retrain the networks on all the data and apply this weighting
		trained_superlearner = {}

		for key in self.est_dict.keys():
			est = self.est_dict[key]
			if key == 'poly':
				est = LogisticRegression(C=1e2, max_iter=350) if (self.output == 'cls') or (
							self.output == 'proba') else LinearRegression()
				poly = PolynomialFeatures(2)
				x_scaled = poly.fit_transform(x)
				# x_scaled = (2 * (x_scaled - self.x_min) / (self.x_max_sq - self.x_min)) - 1

				est.fit(x_scaled, y)
			else:
				est.fit(x, y)
			trained_superlearner[key] = est
		self.trained_superlearner = trained_superlearner
		return trained_superlearner

	def estimation(self, x, y):

		all_preds = np.zeros((len(y), len(self.trained_superlearner)))
		i = 0
		for key in self.trained_superlearner.keys():
			est = self.trained_superlearner[key]
			if key == 'poly':
				poly = PolynomialFeatures(2)
				x_scaled = poly.fit_transform(x)
			# x_scaled = (2 * (x_scaled - self.x_min) / (self.x_max_sq - self.x_min)) - 1

			if (self.output == 'cls') or self.output == 'proba':
				preds = est.predict_proba(x)[:, 1] if key != 'poly' else est.predict_proba(x_scaled)[:, 1]
			else:
				preds = est.predict(x) if key != 'poly' else est.predict(x_scaled)
			all_preds[:, i] = preds

			i += 1
		weighted_preds = np.dot(all_preds, self.beta)
		weighted_preds = weighted_preds.reshape(-1, 1)
		return weighted_preds

