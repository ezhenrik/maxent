import numpy as np

from .plots import plot_terminal, results_terminal, plot_pdf

class Model:

    def __init__(self, X, freq, Y=False, **kwargs):
        X = np.array(X)
        Y = np.array(Y) if Y else np.zeros(len(freq), dtype=int)
        freq = np.array(freq)

        self.d = {
            'X': X,
            'Y': Y,
            'freq': freq,
            'f_names': kwargs.get('f_names', [f'f_{n+1}' for n in range(len(X[0]))]),
            'x_names': kwargs.get('x_names', [f'x_{n+1}' for n in range(len(X.T[0]))]),
            'y_freq': np.zeros(len(np.unique(Y))),
            'freq_pred': np.zeros(len(freq)),
            'freq_max': np.amax(Y),
            'f': np.zeros(len(X[0])),
            'k': len(X[0]),
            'obs': np.sum(freq),
            'm': len(X),
            'x_nonzero': np.count_nonzero(freq),
            'p_hat':  np.zeros(len(freq)),
            'p_hat_pred':  np.zeros(len(freq)),
            'grads': np.zeros(len(X[0])),
            'loglikelihood': -9999999,
            'E_f': np.sum(X.T*freq, axis=1),
            'E_f_pred': np.zeros(len(X[0])),
            's2': kwargs.get('s2', 1000),
            'mu': kwargs.get('mu', 0),
            'alpha': kwargs.get('alpha', 0.01),
            'clip': kwargs.get('clip', 0),
            'convergence': kwargs.get('convergence', 0.000000000001),
            'convergent': 0,
            'r2_adj': 0,
            'aicc': None,
            'limit': kwargs.get('limit', 10000),
            'report_step': kwargs.get('report_step', 1000),
            'report_callback': kwargs.get('report_callback', lambda: print(self.d['r2_adj'], end='\r')),
            'i': 0,
            'verbose': kwargs.get('verbose', 0)
        }

        # Calculate y frequencies                                                                                                  
        for i,f in zip(self.d['Y'],freq): self.d['y_freq'][i] += f

        # Calculate p hat
        for i, x in enumerate(self.d['freq']): self.d['p_hat'][i] = (x / (self.d['y_freq'][self.d['Y'][i]] or 1))

        self.plot_terminal = lambda: plot_terminal(self.d)
        self.results_terminal = lambda: results_terminal(self.d)
        self.plot_pdf = lambda x: plot_pdf(self.d, x)
    
        np.set_printoptions(suppress=kwargs.get('suppress_e', True))

        if self.d['verbose']:
            print(self.d)

    def predict(self):
        self.d['freq_pred'] = self.d['p_hat_pred']*self.d['y_freq'][self.d['Y']]

    def r2_adj(self):
        numerator = ((self.d['freq'] - self.d['freq_pred']) ** 2).sum(axis=0, dtype=np.float64)
        denominator = ((self.d['freq'] - np.average(self.d['freq'], axis=0)) ** 2).sum(axis=0, dtype=np.float64)
        r2 = 1 - (numerator / denominator)
        r2_adj = 1 - (1 - r2) * ((self.d['obs']-1) / (self.d['obs'] - self.d['k'] - 1))
        self.d['r2_adj'] = r2_adj

    def aicc(self):
        self.d['aicc'] = -2 * self.d['loglikelihood'] + (2.0 * self.d['k']) * (self.d['m'] / (self.d['m']-self.d['k']-1.0))
    
    def fit(self):
        while self.d['i'] < self.d['limit'] and not self.d['convergent']:

            # Get harmonies
            hs = np.exp(-np.dot(self.d['X'], self.d['f']))

            # Get Zs (for each y)
            Zs = np.bincount(self.d['Y'],  weights=hs)

            # Partition using Zs
            self.d['p_hat_pred'] = hs/Zs[self.d['Y']]

            # Get log likelihood
            self.d['loglikelihood'] = np.dot(self.d['freq'], np.log(self.d['p_hat_pred']))

            # Get predicted feature values
            self.d['E_f_pred'] = np.sum(self.d['X'].T * self.d['p_hat_pred'] * self.d['y_freq'][self.d['Y']], axis=1)

            # Gradient descent
            step = self.d['alpha'] * (self.d['E_f'] - self.d['E_f_pred'] + (self.d['f']-self.d['mu'])/self.d['s2'])
            self.d['f'] -= step
            
            # Clip, if set
            if self.d['clip']: self.d['f'] = np.clip(self.d['f'], 0, None)

            # Check convergence
            self.d['convergent'] = np.sum(abs(step) / self.d['k']) < self.d['convergence']

            # Report
            if self.d['report_step'] and (self.d['i'] % self.d['report_step']) == 0:
                self.predict()
                self.r2_adj()
                self.d['report_callback']()

            self.d['i'] += 1

        self.predict()
        self.r2_adj()
        self.aicc()