import numpy as np

def learn(X, freq, Y=False, **kwargs):
    X = np.array(X)
    freq = np.array(freq)

    d = {
        'X': X,
        'Y': np.array(Y) if Y else np.zeros(len(X.T[0]), dtype=int),
        'freq': freq,
        'f': np.ones(len(X[0])),
        'obs': np.sum(freq),
        'm': len(X),
        'p_hat_pred':  np.zeros(len(X.T[0])),
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
        'limit': kwargs.get('limit', 500000),
        'report_step': kwargs.get('report_step', 1000),
        'report_callback': kwargs.get('report_callback'),
        'i': 0,
    }

    np.set_printoptions(suppress=kwargs.get('suppress_e', True))

    # Calculate y frequencies
    d['y_freq'] = np.zeros(len(np.unique(d['Y'])))                                                                                                    
    for i,f in zip(d['Y'],freq): 
        d['y_freq'][i] += f

    # Calculate p hat
    d['p_hat'] = np.zeros(len(d['freq']))
    for i, x in enumerate(d['freq']):
        d['p_hat'][i] = (x / (d['y_freq'][d['Y'][i]] or 1))

    while d['i'] < d['limit'] and not d['convergent']:

        # Get harmonies
        hs = np.exp(-np.dot(d['X'], d['f']))

        # Get Zs (for each y)
        Zs = np.zeros(len(np.unique(d['Y'])))                                                                                                    
        for i, h in zip(d['Y'],hs): 
            Zs[i] += h 

        # Partition using Zs
        for i, y in enumerate(d['Y']):
            d['p_hat_pred'][i] = hs[i]/Zs[y]

        # Get log likelihood (first saving previous value)
        d['last_loglikelihood'] = d['loglikelihood']
        d['loglikelihood'] = np.dot(d['freq'], np.log(d['p_hat_pred']))

        # Get predicted feature values
        for i, x in enumerate(d['X'].T):
            f_pred = 0
            for j, y in enumerate(x):
                f_pred  += y*d['p_hat_pred'][j]*d['y_freq'][d['Y'][j]]
            d['E_f_pred'][i] = f_pred

        # Get gradients
        d['grads'] =  d['E_f'] - d['E_f_pred']

        # Gradient descent
        d['f'] -= d['alpha'] * d['grads']

        # Priors
        d['f'] -= (d['f']-d['mu'])/d['sigma']

        # Clip, if set
        if d['clip']:
            d['f'] = np.clip(d['f'], 0, None)

        # Check convergence
        d['convergent'] = (d['loglikelihood'] - d['last_loglikelihood']) < d['convergence']

        # Report
        if d['i'] % d['report_step'] == 0 and d['report_callback']:
            d['report_callback'](d)

        d['i'] += 1

    return d