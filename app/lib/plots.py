import numpy as np
import matplotlib.pyplot as plt

def showScatter(Z,title='scatter plot', lx='x', ly='y'):
    N = Z.size
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_title(title)
    ax.set_xlabel(lx)
    ax.set_ylabel(ly)
    ax.scatter(np.arange(N)+1,Z)
    fig.show()
