import numpy as np

from .plots import plot_terminal, plot_pdf

class Model:

    def __init__(self, X, freq, Y=False, **kwargs):
        X = np.array(X)
        Y = np.array(Y) if Y else np.zeros(len(freq), dtype=int)
        freq = np.array(freq)

        self.d = {
            'X': X,
            'Y': Y,
            'y_freq': np.zeros(len(np.unique(Y))),
            'freq': freq,
            'freq_pred': np.zeros(len(freq)),
            'freq_max': np.amax(Y),
            'f': np.ones(len(X[0])),
            'obs': np.sum(freq),
            'm': len(X),
            'p_hat':  np.zeros(len(freq)),
            'p_hat_pred':  np.zeros(len(freq)),
            'last_loglikelihood': '',
            'grads': np.zeros(len(X[0])),
            'loglikelihood': -9999999,
            'E_f': np.sum(X.T*freq, axis=1),
            'E_f_pred': np.zeros(len(X[0])),
            'sigma': kwargs.get('sigma', 1000000),
            'mu': kwargs.get('mu', 0),
            'f_names': kwargs.get('f_names', [f'f_{i+1}' for i, x in enumerate(X[0])]),
            'x_names': kwargs.get('x_names', [f'x_{i+1}' for i, x in enumerate(X.T[0])]),
            'alpha': kwargs.get('alpha', 0.001),
            'clip': kwargs.get('clip', 1),
            'convergence': kwargs.get('convergence', 0.00000000001),
            'convergent': 0,
            'r2_adj': 0,
            'limit': kwargs.get('limit', 500000),
            'report_step': kwargs.get('report_step', 1000),
            'report_callback': kwargs.get('report_callback', lambda: print(self.d['r2_adj'], end='\r')),
            'i': 0,
        }

        # Calculate y frequencies                                                                                                  
        for i,f in zip(self.d['Y'],freq): self.d['y_freq'][i] += f

        # Calculate p hat
        for i, x in enumerate(self.d['freq']): self.d['p_hat'][i] = (x / (self.d['y_freq'][self.d['Y'][i]] or 1))

        self.plot_terminal = lambda: print(plot_terminal(self.d))
        self.plot_pdf = lambda x: plot_pdf(self.d, x)

        np.set_printoptions(suppress=kwargs.get('suppress_e', True))

    def predict(self):
        for i, x in enumerate(self.d['p_hat_pred']):
            self.d['freq_pred'][i] = x*self.d['y_freq'][self.d['Y'][i]]

    def r2_adj(self):
        numerator = ((self.d['freq'] - self.d['freq_pred']) ** 2).sum(axis=0, dtype=np.float64)
        denominator = ((self.d['freq'] - np.average(self.d['freq'], axis=0)) ** 2).sum(axis=0, dtype=np.float64)
        r2 = 1 - (numerator / denominator)
        r2_adj = 1 - (1 - r2) * ((self.d['obs']-1) / (self.d['obs'] - len(self.d['X'][0]) - 1))
        self.d['r2_adj'] = r2_adj

    def fit(self):
        while self.d['i'] < self.d['limit'] and not self.d['convergent']:

            # Get harmonies
            hs = np.exp(-np.dot(self.d['X'], self.d['f']))

            # Get Zs (for each y)
            Zs = np.zeros(len(np.unique(self.d['Y'])))                                                                                                    
            for i, h in zip(self.d['Y'],hs): 
                Zs[i] += h 

            # Partition using Zs
            for i, y in enumerate(self.d['Y']):
                self.d['p_hat_pred'][i] = hs[i]/Zs[y]

            # Get log likelihood (first saving previous value)
            self.d['last_loglikelihood'] = self.d['loglikelihood']
            self.d['loglikelihood'] = np.dot(self.d['freq'], np.log(self.d['p_hat_pred']))

            # Get predicted feature values
            for i, x in enumerate(self.d['X'].T):
                f_pred = 0
                for j, y in enumerate(x):
                    f_pred  += y*self.d['p_hat_pred'][j]*self.d['y_freq'][self.d['Y'][j]]
                self.d['E_f_pred'][i] = f_pred

            # Get gradients
            self.d['grads'] =  self.d['E_f'] - self.d['E_f_pred']

            # Gradient descent
            self.d['f'] -= self.d['alpha'] * self.d['grads']

            # Priors
            self.d['f'] -= (self.d['f']-self.d['mu'])/self.d['sigma']

            # Clip, if set
            if self.d['clip']:
                self.d['f'] = np.clip(self.d['f'], 0, None)

            # Check convergence
            self.d['convergent'] = (self.d['loglikelihood'] - self.d['last_loglikelihood']) < self.d['convergence']

            # Report
            if self.d['i'] % self.d['report_step'] == 0:
                self.predict()
                self.r2_adj()
                self.d['report_callback']()

            self.d['i'] += 1

        self.predict()
        self.r2_adj()