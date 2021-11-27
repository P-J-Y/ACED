import h5py
import matplotlib.pyplot as plt
import numpy as np

def plotEvent(idx,xdata,ydata,eventSteps):
    ylabels = ["Vp", "Np", "Tp", "theta_B", "phi_B", "|B|", "Bx", "By", "Bz"]
    for i in range(9):
        plt.subplot(10, 1, i+1)
        plt.plot(xdata[i, :eventSteps[0,idx], idx]*stds[i,0] + means[i,0])
        plt.ylabel(ylabels[i])
        plt.xticks([])  # 去掉x轴

    plt.subplot(10, 1, 10)
    plt.plot(ydata[:eventSteps[0,idx],idx]*0.5)
    plt.ylim(0, 1)
    plt.yticks([])  #去掉y轴
    plt.show()

def creat_train_data(xdata,ydata):
    xtrain = np.array(xdata)
    xtrain = xtrain.transpose(2,1,0)
    ytrain = np.array(ydata)
    ytrain = ytrain.transpose(1,0)
    yshape = ytrain.shape
    ytrain = ytrain.reshape(yshape[0],yshape[1],1)
    return xtrain,ytrain

if __name__ == '__main__':
    fileName = 'data/train_v1.mat'
    file = h5py.File(fileName) # "eventSteps","eventTimes","xdata","ydata","means","stds"
    xdata = file['xdata']
    ydata = file['ydata']
    means = file['means']
    stds = file['stds']
    eventSteps = file['eventSteps']

    idx = 1
    #plotEvent(idx, xdata, ydata, eventSteps)
    xtrain,ytrain = creat_train_data(xdata,ydata)

    print("ok?")