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
    def report(val):
        print(val['r2'], end='\r')

    model = maxent.learn(X, freq, Y, report_callback=report)
    print(model)