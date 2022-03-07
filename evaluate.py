# 彭镜宇 2022-03-06
# 对不同的算法进行评价

def checkIcme(icme,args):
    '''
    把间隔很近的分撒的icme放在一起，如果实在是比较分散，再识别为多个icme; 输出每个icme事件的起始和结束时间
    icme: boolean array
    args: dict
    '''

    if sum(icme) == 0:
        print('No icme detected!')
        return
    else:
        print('Icme detected!')
        # diff of icme is 1
        icmeStart = np.where(np.diff(icme.astype(int)) == 1)[0]+1
        icmeEnd = np.where(np.diff(icme.astype(int)) == -1)[0]
        if icme[0] == 1:
            icmeStart = np.insert(icmeStart,0,0)
        if icme[-1] == 1:
            icmeEnd = np.append(icmeEnd,len(icme)-1)

        icmes = []
        theStart = icmeStart[0]
        theEnd = icmeEnd[0]

        if len(icmeStart) > 1:
            assert len(icmeEnd) == len(icmeStart)
            for i in range(1,len(icmeStart)):
                # if the start and end are too close, merge them
                if (args['time'][icmeStart[i]] - args['time'][theEnd]) < datetime.timedelta(hours=6) or (args['time'][icmeStart[i]] - args['time'][theStart]) < datetime.timedelta(hours=18):
                    theEnd = icmeEnd[i]
                else:
                    icmes.append((args['time'][theStart],args['time'][theEnd]))
                    theStart = icmeStart[i]
                    theEnd = icmeEnd[i]
        icmes.append((args['time'][theStart],args['time'][theEnd]))
        return icmes

def evaluateIcme(icmes,ys,args):
    # icmes: list of tuple (start,end), icme detected
    # ys: list of tuple (start,end), icme list given
    # args: dict
    # find the overlap between icme detected and icme list given
    # return: dict
    # {'icme_detected':icme_detected, 'icme_list':icme_list, 'overlap':overlap, 'precision':precision, 'recall':recall}
    icme_detected = []
    icme_list = []
    overlap = []
    precision = []
    recall = []
    for i in range(len(icmes)):
        icme_detected.append((icmes[i][0],icmes[i][1]))
    for i in range(len(ys)):
        icme_list.append((ys[i][0],ys[i][1]))
    for i in range(len(icme_detected)):
        for j in range(len(icme_list)):
            if icme_detected[i][0] <= icme_list[j][0] and icme_detected[i][1] >= icme_list[j][1]:
                overlap.append((icme_list[j][0],icme_list[j][1]))
            elif icme_detected[i][0] >= icme_list[j][0] and icme_detected[i][1] <= icme_list[j][1]:
                overlap.append((icme_detected[i][0],icme_detected[i][1]))
            elif icme_detected[i][0] >= icme_list[j][0] and icme_detected[i][0] <= icme_list[j][1]:
                overlap.append((icme_detected[i][0],icme_list[j][1]))
            elif icme_detected[i][1] >= icme_list[j][0] and icme_detected[i][1] <= icme_list[j][1]:
                overlap.append((icme_list[j][0],icme_detected[i][1]))
    for i in range(len(overlap)):
        precision.append(overlap[i][1]-overlap[i][0])
        recall.append(overlap[i][1]-overlap[i][0])
    if precision == []:
        precision = 0
        recall = 0
        return {'icme_detected':icme_detected, 'icme_list':icme_list, 'overlap':overlap, 'precision':precision, 'recall':recall}
    elif precision[0] < datetime.timedelta(seconds=1): # if overlap is too small, ignore it
        precision = 0
        recall = 0
        return {'icme_detected':icme_detected, 'icme_list':icme_list, 'overlap':overlap, 'precision':precision, 'recall':recall}
    precision = np.sum(precision)/np.sum([icme_detected[i][1]-icme_detected[i][0] for i in range(len(icme_detected))])
    recall = np.sum(recall)/np.sum([icme_list[i][1]-icme_list[i][0] for i in range(len(icme_list))])
    return {'icme_detected':icme_detected, 'icme_list':icme_list, 'overlap':overlap, 'precision':precision, 'recall':recall}


#################### SWICS ####################
import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from aced_utils import *
import datetime
import os



def loaddata_swics():
    fileName = 'data/eval/SWICS/data.mat'
    file = h5py.File(fileName)  # "eventSteps","eventEpochs","xdata","ydata"
    xdata = file['xdata']
    ydata = file['ydata']
    eventTimes = file['eventEpochs']
    eventSteps = file['eventSteps']
    return xdata,ydata,eventTimes,eventSteps


def SWICS(args):
    icme = args['O76'] >= 6.008*np.exp(-0.00578*args['Vp'])
    return icme


def plot_swics(args, icmes, ys=None, eval=None):
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(311)
    ax1.plot(args['time'], args['O76'])
    # set ylabel
    ax1.set_ylabel('O76', fontsize=16)
    # close x ticks
    ax1.set_xticklabels([])
    # set title
    ax1.set_title('SWICS', fontsize=16)
    # set xlim
    ax1.set_xlim(args['time'][0], args['time'][-1])
    ax2 = fig.add_subplot(312)
    ax2.plot(args['time'], args['Vp'], label='V$_{p}$')
    # set ylabel
    ax2.set_ylabel('V$_{p}$ [Km/s]', fontsize=16)
    # close x ticks
    ax2.set_xticklabels([])
    # set xlim
    ax2.set_xlim(args['time'][0], args['time'][-1])
    ax3 = fig.add_subplot(313)
    idx = 0
    for icme in icmes:
        if idx == 0:
            # set label
            line1, = ax3.plot([icme[0], icme[1]], [3 / 4, 3 / 4], 'r-', linewidth=3)
            idx += 1
        else:
            ax3.plot([icme[0], icme[1]], [3 / 4, 3 / 4], 'r-', linewidth=3)

    idx = 0
    if ys is not None:  # if ys is not None, plot ys
        for y in ys:
            if idx == 0:
                # set label
                line2, = ax3.plot([y[0], y[1]], [2 / 4, 2 / 4], 'b-', linewidth=3)
                idx += 1
            else:
                ax3.plot([y[0], y[1]], [2 / 4, 2 / 4], 'b-', linewidth=3)
        if eval is None:    # if eval is None, 标注就只有两个
            ax3.legend(handles=[line1, line2], labels=['SWICS', 'R&C'])
        else:
            overlapLen = len(eval['overlap'])
            ax1.set_title('SWICS P={:.2f} R={:.2f}'.format(eval['precision'], eval['recall']), fontsize=16)
            # plot overlap
            idx = 0
            for i in range(overlapLen):
                if idx == 0:
                    line3, = ax3.plot([eval['overlap'][i][0], eval['overlap'][i][1]], [1 / 4, 1 / 4], 'g-',
                                      linewidth=3)
                    idx += 1
                else:
                    ax3.plot([eval['overlap'][i][0], eval['overlap'][i][1]], [1 / 4, 1 / 4], 'g-', linewidth=3)
            ax3.legend(handles=[line1, line2, line3], labels=['SWICS', 'R&C', 'overlap'])




    # set ylabel
    ax3.set_ylabel('ICME',fontsize=16)
    # set xlabel
    ax3.set_xlabel('Time',fontsize=16)
    # set xlim
    ax3.set_xlim(args['time'][0],args['time'][-1])
    # set ylim
    ax3.set_ylim([0,1])
    # close y ticks
    ax3.set_yticklabels([])

    # save figure
    if not os.path.exists('image/eval/SWICS'):
        os.makedirs('image/eval/SWICS')
    plt.savefig('image/eval/SWICS/'+args['time'][0].strftime('%Y%m%d%H%M')+'_'+args['time'][-1].strftime('%Y%m%d%H%M')+'.png')

def eventTest_swics(eventIdx):
    eventTime = eventTimes[:eventSteps[0, eventIdx], eventIdx]
    # convert to datetime
    eventTime = (eventTime - 719529.0) * 86400.0 - 8.0 * 3600.0
    eventTime = [datetime.datetime.fromtimestamp(t) for t in eventTime]
    eventO76 = xdata[0, :eventSteps[0, eventIdx], eventIdx]
    eventVp = xdata[1, :eventSteps[0, eventIdx], eventIdx]
    args = {'time': eventTime, 'O76': eventO76, 'Vp': eventVp, 'y': ydata[:eventSteps[0, eventIdx], eventIdx]}
    icme = SWICS(args)
    icmes = checkIcme(icme, args)
    ys = checkIcme(args['y'], args)
    if icmes is not None:
        eval_swics = evaluateIcme(icmes, ys, args)
        if eval_swics['recall'] == 0:
            print('Final: No ICME detected!')
            return None
        plot_swics(args,icmes,ys,eval_swics)
    else:
        print('Final: No ICME detected!')




if __name__ == '__main__':
    xdata,ydata,eventTimes,eventSteps = loaddata_swics()
    for i in range(eventSteps.shape[1]):
        eventTest_swics(i)
print('SWICS')
