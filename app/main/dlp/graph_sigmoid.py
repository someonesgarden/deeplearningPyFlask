#!/usr/bin/env python3
# _*_ coding:utf-8 _*_

import sys
sys.path.append('/Users/user/PycharmProjects/deeplearningFlask')

import matplotlib.pyplot as plt
import numpy as np
from app.lib.util import sigmoid

def draw_sigmoid():
    z = np.arange(-7,7,0.1)
    phi_z = sigmoid(z)
    plt.plot(z, phi_z)
    plt.axvline(x=0.0, color='k')
    plt.axhline(y=0.0, xmin=0.0, xmax=1.0, color='k',alpha=1.0, ls='dotted')
    plt.axhline(y=1.0, xmin=0.0, xmax=1.0, color='k', alpha=1.0, ls='dotted')
    #plt.yticks(np.linspace(0,1,10))
    plt.yticks(np.arange(0,1,0.1))
    plt.ylim(-0.1, 1.1)
    plt.xlabel('z')
    plt.ylabel('$\phi (z)$')
    plt.show()


if __name__ == '__main__':
    draw_sigmoid()