from terminaltables import AsciiTable
import termplotlib as tpl
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use('pgf')

matplotlib.rcParams.update({
    'pgf.texsystem': 'pdflatex',
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

matplotlib.rc('axes',edgecolor='dimgrey')

def best_fit(X, Y):

    xbar = sum(X)/len(X)
    ybar = sum(Y)/len(Y)
    n = len(X) # or len(Y)

    numer = sum(xi*yi for xi,yi in zip(X, Y)) - n * xbar * ybar
    denum = sum(xi**2 for xi in X) - n * xbar**2

    b = numer / denum
    a = ybar - b * xbar

    return a, b

def plot_terminal(d):
    fig = tpl.figure()
    plot_config = {
        'width': 70,
        'height': 30,
        'xlim': [-1, d['freq_max']+1],
        'ylim': [-1, d['freq_max']+1],
        'extra_gnuplot_arguments': [
            'set label "%1.3g" at %s,1 right' % (d['r2_adj'], d['freq_max'])
        ],
        'plot_command': 'plot "-" pt "*"'
    }
    fig.plot(d['freq_pred'], d['freq'], **plot_config)
    return fig.get_string()
    
def plot_pdf(d, path):
    fontsize = 10
    figdim = 1 / 72.27 * 455.25 * 0.60
    fig, ax = plt.subplots(figsize=(figdim * 1.12, figdim))
    ax.scatter(d['freq'], d['freq_pred'], color="black", s=10,rasterized=True)
    plt.xlabel("Predicted", fontsize=fontsize)
    plt.ylabel("Observed", fontsize=fontsize)
    ax.tick_params(axis = 'both', which = 'major', labelsize = fontsize)
    ax.tick_params(axis = 'both', which = 'minor', labelsize = fontsize)
    font = {'size': fontsize}
    plt.rc('font', **font)

    a, b = best_fit(d['freq'], d['freq_pred'])

    yfit = [a + b * xi for xi in d['freq']]
    ax.plot(d['freq'], yfit, linestyle='-', color="black", linewidth=1,alpha=0.33)
    plt.xlim([-1, d['freq_max']+1])
    plt.ylim([-1, d['freq_max']+1])

    oper = '-' if a < 0 else '+'

    label = '$y = %1.3gx %s %1.3g$\n$R^2_{adj} = %1.3g$' % (b, oper, np.abs(a), d['r2_adj'])
    extra_label = '%s candidates\n$n$ = %s' % (d['m'], d['obs'])

    ax.text(0.95, 0.05, label, horizontalalignment='right',
        verticalalignment='bottom',
        transform=ax.transAxes)

    ax.text(0.05, 0.95, extra_label, horizontalalignment='left',
        verticalalignment='top',
        transform=ax.transAxes)

    plt.savefig(path, format='pdf',dpi=600)
    plt.clf()


