import maxent

# Frequencies 
freq = [1,1,2,1,0,0]

# Labels
Y = [0,0,1,1,2,2]

# Feature matrix
X = [
    [1,0],
    [0,1],
    [0,0],
    [0,1],
    [2,0],
    [0,1],
]

if __name__ == '__main__':
    model = maxent.Model(X, freq, Y)
    model.fit()
    model.plot_terminal()
    model.results_terminal()
    model.plot_pdf('plot.pdf')