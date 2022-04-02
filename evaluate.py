# 彭镜宇 2022-03-06
# 对不同的算法进行评价
import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from aced_utils import *
import datetime
import os

def checkIcme(icme,args):
    '''
    把间隔很近的分撒的icme放在一起，如果实在是比较分散，再识别为多个icme; 输出每个icme事件的起始和结束时间
    icme: boolean array
    args: dict
    '''

    if sum(icme) < 1:
        print('No icme detected!')
        return None
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

def evaluateIcme(icmes,ys):
    # icmes: list of tuple (start,end), icme detected
    # ys: list of tuple (start,end), icme list given
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
    elif np.sum(precision) < datetime.timedelta(seconds=1): # if overlap is too small, ignore it
        precision = 0
        recall = 0
        return {'icme_detected':icme_detected, 'icme_list':icme_list, 'overlap':overlap, 'precision':precision, 'recall':recall}
    precision = np.sum(precision)/np.sum([icme_detected[i][1]-icme_detected[i][0] for i in range(len(icme_detected))])
    recall = np.sum(recall)/np.sum([icme_list[i][1]-icme_list[i][0] for i in range(len(icme_list))])
    return {'icme_detected':icme_detected, 'icme_list':icme_list, 'overlap':overlap, 'precision':precision, 'recall':recall}


#################### SWICS ####################




def loaddata_swics(fileName = 'data/eval/SWICS/data.mat'):
    # event
    # file = h5py.File(fileName)  # "datatot","ytot","timetot",'epochtot'
    # xdata = file['xdata']
    # ydata = file['ydata']
    # eventTimes = file['eventEpochs']
    # eventSteps = file['eventSteps']
    # return xdata,ydata,eventTimes,eventSteps

    # tot
    file = h5py.File(fileName)  # "datatot","ytot","timetot",'epochtot'
    xdata = file['datatot'][:]
    ydata = file['ytot'][:]
    eventTimes = file['epochtot'][:]
    return xdata,ydata,eventTimes



def SWICS(args):
    icme = args['O76'] >= 6.008*np.exp(-0.00578*args['Vp']) # Vp km/s
    return icme


# def plot_swics(args, icmes, ys=None, eval=None):
#     fig = plt.figure(figsize=(12, 6))
#     ax1 = fig.add_subplot(311)
#     ax1.plot(args['time'], args['O76'])
#     # set ylabel
#     ax1.set_ylabel('O76', fontsize=16)
#     # close x ticks
#     ax1.set_xticklabels([])
#     # set title
#     ax1.set_title('SWICS', fontsize=16)
#     # set xlim
#     ax1.set_xlim(args['time'][0], args['time'][-1])
#     ax2 = fig.add_subplot(312)
#     ax2.plot(args['time'], args['Vp'], label='V$_{p}$')
#     # set ylabel
#     ax2.set_ylabel('V$_{p}$ [Km/s]', fontsize=16)
#     # close x ticks
#     ax2.set_xticklabels([])
#     # set xlim
#     ax2.set_xlim(args['time'][0], args['time'][-1])
#     ax3 = fig.add_subplot(313)
#     idx = 0
#     for icme in icmes:
#         if idx == 0:
#             # set label
#             line1, = ax3.plot([icme[0], icme[1]], [3 / 4, 3 / 4], 'r-', linewidth=3)
#             idx += 1
#         else:
#             ax3.plot([icme[0], icme[1]], [3 / 4, 3 / 4], 'r-', linewidth=3)
#
#     idx = 0
#     if ys is not None:  # if ys is not None, plot ys
#         for y in ys:
#             if idx == 0:
#                 # set label
#                 line2, = ax3.plot([y[0], y[1]], [2 / 4, 2 / 4], 'b-', linewidth=3)
#                 idx += 1
#             else:
#                 ax3.plot([y[0], y[1]], [2 / 4, 2 / 4], 'b-', linewidth=3)
#         if eval is None:    # if eval is None, 标注就只有两个
#             ax3.legend(handles=[line1, line2], labels=['SWICS', 'R&C'])
#         else:
#             overlapLen = len(eval['overlap'])
#             ax1.set_title('SWICS P={:.2f} R={:.2f}'.format(eval['precision'], eval['recall']), fontsize=16)
#             # plot overlap
#             idx = 0
#             for i in range(overlapLen):
#                 if idx == 0:
#                     line3, = ax3.plot([eval['overlap'][i][0], eval['overlap'][i][1]], [1 / 4, 1 / 4], 'g-',
#                                       linewidth=3)
#                     idx += 1
#                 else:
#                     ax3.plot([eval['overlap'][i][0], eval['overlap'][i][1]], [1 / 4, 1 / 4], 'g-', linewidth=3)
#             ax3.legend(handles=[line1, line2, line3], labels=['SWICS', 'R&C', 'overlap'])
#
#
#
#
#     # set ylabel
#     ax3.set_ylabel('ICME',fontsize=16)
#     # set xlabel
#     ax3.set_xlabel('Time',fontsize=16)
#     # set xlim
#     ax3.set_xlim(args['time'][0],args['time'][-1])
#     # set ylim
#     ax3.set_ylim([0,1])
#     # close y ticks
#     ax3.set_yticklabels([])
#
#     # save figure
#     if not os.path.exists('image/eval/SWICS'):
#         os.makedirs('image/eval/SWICS')
#     plt.savefig('image/eval/SWICS/'+args['time'][0].strftime('%Y%m%d%H%M')+'_'+args['time'][-1].strftime('%Y%m%d%H%M')+'.png')
#
# def eventTest_swics(eventIdx,eventTimes,eventSteps,xdata,ydata):
#     eventTime = eventTimes[:eventSteps[0, eventIdx], eventIdx]
#     # convert to datetime
#     eventTime = (eventTime - 719529.0) * 86400.0 - 8.0 * 3600.0
#     eventTime = [datetime.datetime.fromtimestamp(t) for t in eventTime]
#     eventO76 = xdata[0, :eventSteps[0, eventIdx], eventIdx]
#     eventVp = xdata[1, :eventSteps[0, eventIdx], eventIdx]
#     args = {'time': eventTime, 'O76': eventO76, 'Vp': eventVp, 'y': ydata[:eventSteps[0, eventIdx], eventIdx]}
#     icme = SWICS(args)
#     icmes = checkIcme(icme, args)
#     ys = checkIcme(args['y'], args)
#     if icmes is not None:
#         eval_swics = evaluateIcme(icmes, ys)
#         if eval_swics['recall'] == 0:
#             print('Final: No ICME detected!')
#             return None
#         plot_swics(args,icmes,ys,eval_swics)
#     else:
#         print('Final: No ICME detected!')

##################### XB #####################

# K to eV
def K2eV(K):
    return K * 8.6173324e-5

def loaddata_xb(fileName = 'data/eval/XB/datatot.mat'):

    # events
    # file = h5py.File(fileName)  # "eventSteps","eventEpochs","xdata","ydata"
    # xdata = file['xdata']
    # ydata = file['ydata']
    # eventTimes = file['eventEpochs']
    # eventSteps = file['eventSteps']
    # return xdata,ydata,eventTimes,eventSteps

    # tot
    file = h5py.File(fileName)  # "datatot","ytot","timetot",'epochtot'
    xdata = file['datatot'][:]
    ydata = file['ytot'][:]
    eventTimes = file['epochtot'][:]
    return xdata,ydata,eventTimes


def XB(args):
    icme = 0.841*args['Mag']*(args['Np']**-0.315)*(K2eV(args['Tp'])**-0.0222)*(args['Vp']**-0.171) > 1 # Vp in km/s, np in cm-3, B in nT, Tp in K(通过K2eV转换成eV)
    return icme


# def plot_xb(args, icmes, ys=None, eval=None):
#     fig = plt.figure(figsize=(16, 9))
#     ax1 = fig.add_subplot(511)
#     ax1.plot(args['time'], args['Np'], label='Np')
#     ax1.set_ylabel('Np cm$^{-3}$', fontsize=16)
#     ax1.set_xlim(args['time'][0], args['time'][-1])
#     # close x ticks
#     ax1.set_xticklabels([])
#     ax2 = fig.add_subplot(512)
#     ax2.plot(args['time'], args['Tp'], label='Tp')
#     ax2.set_ylabel('Tp eV', fontsize=16)
#     ax2.set_xlim(args['time'][0], args['time'][-1])
#     # close x ticks
#     ax2.set_xticklabels([])
#     ax2 = fig.add_subplot(513)
#     ax2.plot(args['time'], args['Vp'], label='Vp')
#     ax2.set_ylabel('Vp km/s', fontsize=16)
#     ax2.set_xlim(args['time'][0], args['time'][-1])
#     # close x ticks
#     ax2.set_xticklabels([])
#     ax2 = fig.add_subplot(514)
#     ax2.plot(args['time'], args['Mag'], label='Mag')
#     ax2.set_ylabel('B nT', fontsize=16)
#     ax2.set_xlim(args['time'][0], args['time'][-1])
#     # close x ticks
#     ax2.set_xticklabels([])
#     ax3 = fig.add_subplot(515)
#     idx = 0
#     for icme in icmes:
#         if idx == 0:
#             line1, = ax3.plot([icme[0], icme[1]], [3 / 4, 3 / 4], 'r-', linewidth=3)
#             idx += 1
#         else:
#             ax3.plot([icme[0], icme[1]], [3 / 4, 3 / 4], 'r-', linewidth=3)
#     if ys is not None:
#         idx = 0
#         for y in ys:
#             if idx == 0:
#                 line2, = ax3.plot([y[0], y[1]], [2 / 4, 2 / 4], 'b-', linewidth=3)
#                 idx += 1
#             else:
#                 ax3.plot([y[0], y[1]], [2 / 4, 2 / 4], 'b-', linewidth=3)
#         if eval is None:
#             ax3.legend(handles=[line1, line2], labels=['XB', 'R&C'])
#         else:
#             overlapLen = len(eval['overlap'])
#             ax1.set_title('XB P={:.2f} R={:.2f}'.format(eval['precision'], eval['recall']), fontsize=16)
#             # plot overlap
#             idx = 0
#             for i in range(overlapLen):
#                 if idx == 0:
#                     line3, = ax3.plot([eval['overlap'][i][0], eval['overlap'][i][1]], [1 / 4, 1 / 4], 'g-',
#                                       linewidth=3)
#                     idx += 1
#                 else:
#                     ax3.plot([eval['overlap'][i][0], eval['overlap'][i][1]], [1 / 4, 1 / 4], 'g-', linewidth=3)
#             ax3.legend(handles=[line1, line2, line3], labels=['XB', 'R&C', 'overlap'])
#
#     # set ylabel
#     ax3.set_ylabel('ICME',fontsize=16)
#     # set xlabel
#     ax3.set_xlabel('Time',fontsize=16)
#     # set xlim
#     ax3.set_xlim(args['time'][0],args['time'][-1])
#     # set ylim
#     ax3.set_ylim([0,1])
#     # close y ticks
#     ax3.set_yticklabels([])
#
#     # save figure
#     if not os.path.exists('image/eval/XB'):
#         os.makedirs('image/eval/XB')
#     plt.savefig('image/eval/XB/'+args['time'][0].strftime('%Y%m%d%H%M')+'_'+args['time'][-1].strftime('%Y%m%d%H%M')+'.png')
#
# def eventTest_xb(eventIdx,eventTimes,eventSteps,xdata,ydata):
#     eventTime = eventTimes[:eventSteps[0, eventIdx], eventIdx]
#     # convert to datetime
#     eventTime = (eventTime - 719529.0) * 86400.0 - 8.0 * 3600.0
#     eventTime = [datetime.datetime.fromtimestamp(t) for t in eventTime]
#     eventNp = xdata[0, :eventSteps[0, eventIdx], eventIdx] # in cm-3
#     eventTp = xdata[1, :eventSteps[0, eventIdx], eventIdx]  # in k
#     eventVp = xdata[2, :eventSteps[0, eventIdx], eventIdx] # in Km/s
#     eventMag = xdata[3, :eventSteps[0, eventIdx], eventIdx] # in nT
#
#     args = {'time': eventTime, 'Np': eventNp, 'Tp':eventTp, 'Vp': eventVp,'Mag':eventMag, 'y': ydata[:eventSteps[0, eventIdx], eventIdx]}
#     icme = XB(args)
#     icmes = checkIcme(icme, args)
#     ys = checkIcme(args['y'], args)
#     if icmes is not None:
#         eval_xb = evaluateIcme(icmes, ys)
#         if eval_xb['recall'] == 0:
#             print('Final: No ICME detected!')
#             return None
#         plot_xb(args,icmes,ys,eval_xb)
#     else:
#         print('Final: No ICME detected!')

################## Genesis ##################
from preprocessing import *

def ToCME(tnow,
          Tshock,
          TextoTp,
          cons,
          NHetoNp=None,
          Be=None,
          Wa=1,
          Wb=0,):

    # 很多参数都是待定的，需要调试

   # if tnow is between ctime1 and ctime2 after any Tshock

   if Tshock.size==0:
       Yt = cons['CT3'] * TextoTp - cons['CT4']
       if Wa:
           Ya = cons['CA3'] * NHetoNp - cons['CA4']
       else:
           Ya = 0
       if Wb:
           Yb = cons['CB3'] * Be - cons['CB4']
       else:
           Yb = 0
   elif any( (tnow > (Tshock + cons['ctime1'])) * (tnow < (Tshock + cons['ctime2'])) ):
       Yt = cons['CT1'] * TextoTp - cons['CT2']
       if Wa:
           Ya = cons['CA1'] * NHetoNp - cons['CA2']
       else:
           Ya = 0
       if Wb:
           Yb = cons['CB1'] * Be - cons['CB2']
       else:
           Yb = 0
   else:
       Yt = cons['CT3'] * TextoTp - cons['CT4']
       if Wa:
           Ya = cons['CA3'] * NHetoNp - cons['CA4']
       else:
           Ya = 0
       if Wb:
           Yb = cons['CB3'] * Be - cons['CB4']
       else:
           Yb = 0
   Yt = Yt > 1.0
   Ya = Ya > 1.0
   Yb = Yb > 1.0
   return (Yt + Ya * Wa + Yb * Wb) / (1 + Wa + Wb)

def Genesis(args,cons,Wa=1,Wb=0):

    def Genesis_inner(lastype,
                      TOCME,
                      TextoTp,
                      tnow,
                      tcme,
                      tcme_end,
                      cons,
                      NHetoNp=0,
                      Be=0,
                      Wa=1,
                      Wb=0):
        '''

        :param lastype: 0: None 1: CME 2: Not CME
        :param TOCME:
        :param TextoTp:
        :param NHetoNp:
        :param Be:
        :param tocme_threshold:
        :param Wa:
        :param Wb:
        :return:
        '''
        if lastype == 1:
            if (TextoTp>cons['Tout']) or (Wa*NHetoNp>cons['Aout']) or (Wb*Be>cons['Bout']) or (TOCME>cons['tocme_threshold']):
                tcme_end = max(tcme+cons['tstay'],tnow+cons['tlag'])
                newtype = 1
            elif tnow<tcme_end:
                newtype = 1
            else:
                newtype = 2
        else:
            if TOCME>cons['tocme_threshold']:
                tcme = tnow
                tcme_end = max(tcme+cons['tstay'],tnow+cons['tlag'])
                newtype = 1
            else:
                newtype = 2
        return newtype, tcme, tcme_end

    # 初始化
    tcme = np.nan
    tcme_end = np.nan
    icme = np.zeros(len(args['time']), dtype=bool)
    lastype = 0
    TOCMEs = np.zeros(len(args['time']))

    # 循环
    for i in range(len(args['time'])):
        if Wa and Wb:
            TOCME = ToCME(args['time'][i],
                          args['Tshock'],
                          args['TextoTp'][i],
                          cons,
                          NHetoNp=args['NHetoNp'][i],
                          Be=args['Be'][i],
                          Wa=Wa,
                          Wb=Wb,)
        elif Wa:
            TOCME = ToCME(args['time'][i],
                          args['Tshock'],
                          args['TextoTp'][i],
                          cons,
                          NHetoNp=args['NHetoNp'][i],
                          Wa=Wa,
                          Wb=Wb,)
        elif Wb:
            TOCME = ToCME(args['time'][i],
                          args['Tshock'],
                          args['TextoTp'][i],
                          cons,
                          Be=args['Be'][i],
                          Wa=Wa,
                          Wb=Wb,)
        else:
            TOCME = ToCME(args['time'][i],
                          args['Tshock'],
                          args['TextoTp'][i],
                          cons,
                          Wa=Wa,
                          Wb=Wb,)
        TOCMEs[i] = TOCME
        if Wa and Wb:
            lastype, tcme, tcme_end = Genesis_inner(lastype,
                                                    TOCME,
                                                    args['TextoTp'][i],
                                                    args['time'][i],
                                                    tcme,
                                                    tcme_end,
                                                    cons,
                                                    NHetoNp=args['NHetoNp'][i],
                                                    Be=args['Be'][i],
                                                    Wa=Wa,
                                                    Wb=Wb,)
        elif Wa:
            lastype, tcme, tcme_end = Genesis_inner(lastype,
                                                    TOCME,
                                                    args['TextoTp'][i],
                                                    args['time'][i],
                                                    tcme,
                                                    tcme_end,
                                                    cons,
                                                    NHetoNp=args['NHetoNp'][i],
                                                    Wa=Wa,
                                                    Wb=Wb,)
        elif Wb:
            lastype, tcme, tcme_end = Genesis_inner(lastype,
                                                    TOCME,
                                                    args['TextoTp'][i],
                                                    args['time'][i],
                                                    tcme,
                                                    tcme_end,
                                                    cons,
                                                    Be=args['Be'][i],
                                                    Wa=Wa,
                                                    Wb=Wb,)
        else:
            lastype, tcme, tcme_end = Genesis_inner(lastype,
                                                    TOCME,
                                                    args['TextoTp'][i],
                                                    args['time'][i],
                                                    tcme,
                                                    tcme_end,
                                                    cons,
                                                    Wa=Wa,
                                                    Wb=Wb,)

        icme[i] = lastype==1
    args['TOCME'] = TOCMEs
    return icme

# def plot_genesis(args, icmes, ys=None, eval=None):
#     fig = plt.figure(figsize=(10, 12))
#
#     ax1 = fig.add_subplot(711)
#     ax1.plot(args['time'], args['Vp'], label='Vp')
#     ax1.set_ylabel('Vp Km/s', fontsize=16)
#     ax1.set_xlim(args['time'][0], args['time'][-1])
#     # close x ticks
#     ax1.set_xticklabels([])
#     ax2 = fig.add_subplot(712)
#     ax2.plot(args['time'], args['Tp']*1e-4, label='Tp')
#     ax2.set_ylabel('Tp 10$^4$K', fontsize=16)
#     ax2.set_xlim(args['time'][0], args['time'][-1])
#     # close x ticks
#     ax2.set_xticklabels([])
#     ax2 = fig.add_subplot(713)
#     ax2.plot(args['time'], args['TextoTp'], label='TextoTp')
#     ax2.set_ylabel('T$_{ex}$/T$_p$', fontsize=16)
#     ax2.set_xlim(args['time'][0], args['time'][-1])
#     # close x ticks
#     ax2.set_xticklabels([])
#     ax2 = fig.add_subplot(714)
#     ax2.plot(args['time'], args['NHetoNp'], label='NhetoNp')
#     ax2.set_ylabel('N$_{He}$/N$_p$', fontsize=16)
#     ax2.set_xlim(args['time'][0], args['time'][-1])
#     # close x ticks
#     ax2.set_xticklabels([])
#     ax2 = fig.add_subplot(715)
#     ax2.plot(args['time'], args['Be'], label='BDE')
#     ax2.set_ylabel('BDE', fontsize=16)
#     ax2.set_xlim(args['time'][0], args['time'][-1])
#     ax2.set_ylim(0, 1)
#     # close x ticks
#     ax2.set_xticklabels([])
#     ax2 = fig.add_subplot(716)
#     ax2.plot(args['time'], args['TOCME'], label='TOCME')
#     ax2.set_ylabel('TOCME', fontsize=16)
#     ax2.set_xlim(args['time'][0], args['time'][-1])
#     ax2.set_ylim(0, 1)
#     # close x ticks
#     ax2.set_xticklabels([])
#     ax3 = fig.add_subplot(717)
#     idx = 0
#     for icme in icmes:
#         if idx == 0:
#             line1, = ax3.plot([icme[0], icme[1]], [3 / 4, 3 / 4], 'r-', linewidth=3)
#             idx += 1
#         else:
#             ax3.plot([icme[0], icme[1]], [3 / 4, 3 / 4], 'r-', linewidth=3)
#     if ys is not None:
#         idx = 0
#         for y in ys:
#             if idx == 0:
#                 line2, = ax3.plot([y[0], y[1]], [2 / 4, 2 / 4], 'b-', linewidth=3)
#                 idx += 1
#             else:
#                 ax3.plot([y[0], y[1]], [2 / 4, 2 / 4], 'b-', linewidth=3)
#         if eval is None:
#             ax3.legend(handles=[line1, line2], labels=['Genesis', 'R&C'])
#         else:
#             overlapLen = len(eval['overlap'])
#             ax1.set_title('Genesis P={:.2f} R={:.2f}'.format(eval['precision'], eval['recall']), fontsize=16)
#             # plot overlap
#             idx = 0
#             for i in range(overlapLen):
#                 if idx == 0:
#                     line3, = ax3.plot([eval['overlap'][i][0], eval['overlap'][i][1]], [1 / 4, 1 / 4], 'g-',
#                                       linewidth=3)
#                     idx += 1
#                 else:
#                     ax3.plot([eval['overlap'][i][0], eval['overlap'][i][1]], [1 / 4, 1 / 4], 'g-', linewidth=3)
#             ax3.legend(handles=[line1, line2, line3], labels=['Genesis', 'R&C', 'overlap'])
#
#     # set ylabel
#     ax3.set_ylabel('ICME',fontsize=16)
#     # set xlabel
#     ax3.set_xlabel('Time',fontsize=16)
#     # set xlim
#     ax3.set_xlim(args['time'][0],args['time'][-1])
#     # set ylim
#     ax3.set_ylim([0,1])
#     # close y ticks
#     ax3.set_yticklabels([])
#
#     # save figure
#     if not os.path.exists('image/eval/Genesis'):
#         os.makedirs('image/eval/Genesis')
#     plt.savefig('image/eval/Genesis/'+args['time'][0].strftime('%Y%m%d%H%M')+'_'+args['time'][-1].strftime('%Y%m%d%H%M')+'.png')

# def eventTest_genesis(args):
#
#     icme = Genesis(args,Wa=1,Wb=1,ctime1=datetime.timedelta(hours=10),ctime2=datetime.timedelta(hours=10))
#     icmes = checkIcme(icme, args)
#     ys = checkIcme(args['y'], args)
#     if icmes is not None:
#         eval_genesis = evaluateIcme(icmes, ys)
#         if eval_genesis['recall'] == 0:
#             print('Final: No ICME detected!')
#             return None
#         plot_genesis(args,icmes,ys,eval_genesis)
#     else:
#         print('Final: No ICME detected!')


if __name__ == '__main__':
    # xdata,ydata,eventTimes,eventSteps = loaddata_swics()
    # for i in range(eventSteps.shape[1]):
    #     eventTest_swics(i,eventTimes,eventSteps,xdata,ydata)

    # xdata,ydata,eventTimes,eventSteps = loaddata_xb()
    # for i in range(eventSteps.shape[1]):
    #     eventTest_xb(i,eventTimes,eventSteps,xdata,ydata)

    eventSteps, eventSteps_swe, eventSteps_pa, swedata, padata, magdata, ydata, event_epoch, event_epoch_swe, event_epoch_pa = load_original_data_genesis()
    for i in range(eventSteps.shape[1]):
        print(i)
        args = pre_genesis(i, eventSteps, eventSteps_swe, eventSteps_pa, swedata, padata, ydata, event_epoch, event_epoch_swe, event_epoch_pa)
        # eventTest_genesis(args)
    print('SWICS')
