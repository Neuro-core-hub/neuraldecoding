import numpy as np
import sys
import os
from matplotlib import pyplot as plt

def generateData(time,channel,method,mu=0.,std=1.,val = 1.):
    if(method == 'sequence'):
        data = np.arange(1, time *channel + 1).reshape(channel, time).T
    elif(method == 'gaussian'):
        data = np.random.normal(mu,std,size = (time,channel))
    elif(method == 'same'):
        data = np.full(shape = (time,channel),fill_value = val)
    elif(method == 'same_zero'):
        data = np.random.randint(2, size = (time,channel))
    else:
        raise Exception("no such method")
    return data

def plotData(x,y,xhat,yhat,title,same_y_axis = True):
    fig, ax = plt.subplots(2, 3, figsize=(10, 5))

    fig.suptitle(title)

    ax[0, 0].set_ylabel("before")
    ax[1, 0].set_ylabel("after")

    ax[0,0].set_xlabel("both")
    ax[0,0].xaxis.set_label_position('top')

    ax[0,1].set_xlabel("x")
    ax[0,1].xaxis.set_label_position('top')

    ax[0,2].set_xlabel("y")
    ax[0,2].xaxis.set_label_position('top')

    ax[0][0].plot(x)
    ax[0][0].plot(y,color='black')
    ax[0][1].plot(x)
    if(same_y_axis):
        ax[0][1].set_ylim(ax[0][0].get_ylim())
    ax[0][2].plot(y,color='black')
    if(same_y_axis):
        ax[0][2].set_ylim(ax[0][0].get_ylim())

    ax[1][0].plot(xhat)
    ax[1][0].plot(yhat,color='black')
    ax[1][1].plot(xhat)
    if(same_y_axis):
        ax[1][1].set_ylim(ax[1][0].get_ylim())
    ax[1][2].plot(yhat,color='black')
    if(same_y_axis):
        ax[1][2].set_ylim(ax[1][0].get_ylim())
    plt.show()
