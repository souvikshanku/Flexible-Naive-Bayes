import numpy as np
from scipy.stats import gaussian_kde as sp_kde
from statsmodels.nonparametric.kernel_density import KDEMultivariate as sm_kde


class kde:
    __bw_methods__ = ['scott', 'silverman', 'normal_reference', 'cv_ml', 'cv_ls']

    def __init__(self, X, bw_sel_method, var_type):
        if X.ndim == 1:
            X = X.reshape(1, -1)

        self.n_samples, self.dimension = X.shape
        self.kde_estimator = None
        if bw_sel_method not in kde.__bw_methods__:
            raise ValueError("Invalid bandwidth selection method")

        self.bw_sel_method = bw_sel_method

        if bw_sel_method in ['scott', 'silverman']:
            self.kde_estimator = sp_kde(X.T, bw_sel_method)

        elif bw_sel_method in ['normal_reference', 'cv_ml', 'cv_ls']:
            self.kde_estimator = sm_kde(X, var_type, bw_sel_method)
        

    def __call__(self, X_new):
        if self.bw_sel_method in ['scott', 'silverman']:
            return self.kde_estimator(X_new.T)

        elif self.bw_sel_method in ['normal_reference', 'cv_ml', 'cv_ls']:
            return self.kde_estimator.pdf(X_new)
        

class FlexibleNB:
    def __init__(self, bw_sel, var_type):
        self.init_params()
        if bw_sel not in kde.__bw_methods__:
            raise ValueError("Invalid bandwidth selection method")
        self.bw_sel = bw_sel
        self.var_type = var_type

    def init_params(self):
        self.n = 0
        self.p = 0
        self.counts = None
        self.n_classes = 0
        self.kdes = None

    def fit(self, X, y):
        self.n, self.p = X.shape
        y = y.flatten()

        _, self.counts = np.unique(y, return_counts=True)
        self.n_classes = len(self.counts)

        self.kdes = [
            kde(X[y == k, :], self.bw_sel, self.var_type)
            for k in range(self.n_classes)
        ]
        
    def predict(self, X):
        y_prob = self.predict_prob(X)
        y_pred = y_prob.argmax(axis=1)
        return y_pred

    def predict_prob(self, X):
        X = X.reshape(-1, self.p)
        m, _ = X.shape

        y_prob = np.zeros((m, self.n_classes))

        for j in range(self.n_classes):
            y_prob[:, j] = self.kdes[j](X)

        y_prob = y_prob * self.counts.reshape(1, -1)
        return y_prob

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
