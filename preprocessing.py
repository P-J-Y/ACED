import datetime

def running_avg(x,Tx,T,t1=datetime.timedelta(minutes=30),t2=datetime.timedelta(minutes=30)):
    assert len(x) == len(Tx), 'x and Tx must have the same length'
    x_means = np.zeros(len(T))
    for i in range(len(T)):
        x_means[i] = np.nanmean(x[(np.array(Tx)>=T[i]-t1) & (np.array(Tx)<=T[i]+t2)])
    return x_means




################### Genesis ######################


import h5py
import numpy as np


def load_original_data_genesis(fileName = 'data/eval/Genesis/origin_data.mat'):

    file = h5py.File(fileName)  # "eventSteps","eventSteps_swe","eventSteps_pa","swedata","padata","ydata",eventEpochs,eventEpochs_swe,eventEpochs_pa
    eventSteps = file['eventSteps'][:]
    eventSteps_swe = file['eventSteps_swe'][:]
    if 'eventSteps_pa' in file.keys():
        eventSteps_pa = file['eventSteps_pa'][:]
    else:
        eventSteps_pa = None
    swedata = file['swedata'][:]
    if "padata" in file.keys():
        padata = file['padata'][:]
    else:
        padata = None
    if 'ydata' in file.keys():
        ydata = file['ydata'][:]
    else:
        ydata = None
    eventEpochs = file['eventEpochs'][:]
    eventEpochs_swe = file['eventEpochs_swe'][:]
    if 'eventEpochs_pa' in file.keys():
        eventEpochs_pa = file['eventEpochs_pa'][:]
    else:
        eventEpochs_pa = None
    if 'magdata' in file.keys():
        magdata = file['magdata'][:]
    else:
        magdata = None
    file.close()
    return eventSteps, eventSteps_swe, eventSteps_pa, swedata, padata, magdata, ydata, eventEpochs, eventEpochs_swe, eventEpochs_pa
    # return eventSteps, eventSteps_swe, swedata, ydata, eventEpochs, eventEpochs_swe

def Tex(Vp,cons):
    '''

    :param Vp: in km/s
    :return: Tp in K
    '''
    # some constants from OMNI data, modified with new data might be better
    # 具体的参数可能还需要根据所需要的数据进行拟合，实际上就是对太阳风（CME排除）的Tp用Vp二次拟合
    # Vcut = cons['Vcut'] # km/s
    # C1 = 2.6e4
    # C2 = 316.2
    # C3 = 0.961
    # C4 = -1.42e5
    # C5 = 510.0
    # C6 = 0

    # calculate Tex
    Tex = (
          (Vp<cons['Vcut']) * ( cons['C3'] * Vp**2 + cons['C2'] * Vp + cons['C1']) +
          (Vp>=cons['Vcut']) * (cons['C6'] * Vp**2 + cons['C5'] * Vp + cons['C4'])
           )

    return Tex

def BDE(PA,threshold=2):
    # bi-directional electron stream
    # PA 20*timepoints
    ############ 阈值还需要确定 #############

    # 2:3:2
    pachs = np.shape(PA)[0]
    c13_len = int(pachs*2/7)
    c2_len = pachs - c13_len*2
    C1 = np.nanmean(PA[0:c13_len],axis=0)
    C2 = np.nanmean(PA[c13_len:c13_len+c2_len],axis=0)
    C3 = np.nanmean(PA[c13_len+c2_len:],axis=0)

    Be = (C1>(C2*threshold)) * (C3>(C2*threshold))
    return Be

def avg_genesis(input,t_avg=datetime.timedelta(hours=1),Wa=0,Wb=0):

    '''
    :param input: 包含以下数据的dict，Be和NHetoNp可以没有
    Vp: in km/s
    NHetoNp:
    TextoTp:
    Be: 是否有BDE
    Np: in cm-3
    Tp:  in K
    以及各自的时间
    :return: 返回以eventT的点为中心，前后半小时数据的均值；下标1表示对前半个小时取平均，下标2表示对后半个小时取平均
    '''

    V1 = running_avg(input['Vp'],input['Vp_time'],input['time'],t1=t_avg/2,t2=datetime.timedelta(minutes=0))
    V2 = running_avg(input['Vp'],input['Vp_time'],input['time'],t1=datetime.timedelta(minutes=0),t2=t_avg/2)
    N1 = running_avg(input['Np'],input['Np_time'],input['time'],t1=t_avg/2,t2=datetime.timedelta(minutes=0))
    N2 = running_avg(input['Np'],input['Np_time'],input['time'],t1=datetime.timedelta(minutes=0),t2=t_avg/2)
    T1 = running_avg(input['Tp'],input['Tp_time'],input['time'],t1=t_avg/2,t2=datetime.timedelta(minutes=0))
    T2 = running_avg(input['Tp'],input['Tp_time'],input['time'],t1=datetime.timedelta(minutes=0),t2=t_avg/2)

    Vp = running_avg(input['Vp'],input['Vp_time'],input['time'],t1=t_avg/2,t2=t_avg/2)
    Tp = running_avg(input['Tp'],input['Tp_time'],input['time'],t1=t_avg/2,t2=t_avg/2)
    Np = running_avg(input['Np'],input['Np_time'],input['time'],t1=t_avg/2,t2=t_avg/2)
    TextoTp = running_avg(input['TextoTp'], input['Vp_time'], input['time'], t1=t_avg/2, t2=t_avg/2)
    args = {'V1':V1,
            'V2':V2,
            'N1':N1,
            'N2':N2,
            'T1':T1,
            'T2':T2,
            'Vp':Vp,
            'Tp':Tp,
            'Np':Np,
            'TextoTp':TextoTp,
            'time':input['time']}
    if Wa:
        NHetoNp = running_avg(input['NHetoNp'],input['NHetoNp_time'],input['time'],t1=t_avg/2,t2=t_avg/2)
        args['NHetoNp'] = NHetoNp
    if Wb:
        Be = running_avg(input['Be'],input['PA_time'],input['time'],t1=t_avg/2,t2=t_avg/2)
        args['Be'] = Be

    return args

def shock_genesis(args,cons):
    '''

    :param args: 事件参数
    :return: 不return 但是会在args里加上isshock 判读是否是shock的bool; Tshock里面则是shock的时间
    '''
    # some threshold


    isshock = ((args['V2']-args['V1'])>cons['Vjump']) * (args['N2']>cons['RN']*args['N1']) * (args['T2']>cons['RT']*args['T1'])
    Tshock = np.array(args['time'])[isshock]-cons['t_avg']/2
    args['isshock'] = isshock
    args['Tshock'] = Tshock

def pre_genesis(eventIdx,eventSteps, eventSteps_swe, eventSteps_pa, swedata, padata, ydata, event_epoch, event_epoch_swe, event_epoch_pa):
    ## 弃用了
    pass
    # eventPA = padata[:, :eventSteps_pa[0, eventIdx], eventIdx]
    # eventSWE = swedata[:, :eventSteps_swe[0, eventIdx], eventIdx]  # Np;Tp;Vp;He4top
    #
    # eventpaT = event_epoch_pa[:eventSteps_pa[0, eventIdx], eventIdx]
    # eventpaT = (eventpaT - 719529.0) * 86400.0 - 8.0 * 3600.0
    # eventpaT = [datetime.datetime.fromtimestamp(t) for t in eventpaT]
    #
    # eventsweT = event_epoch_swe[:eventSteps_swe[0, eventIdx], eventIdx]
    # eventsweT = (eventsweT - 719529.0) * 86400.0 - 8.0 * 3600.0
    # eventsweT = [datetime.datetime.fromtimestamp(t) for t in eventsweT]
    #
    # eventT = event_epoch[:eventSteps[0, eventIdx], eventIdx]
    # eventT = (eventT - 719529.0) * 86400.0 - 8.0 * 3600.0
    # eventT = [datetime.datetime.fromtimestamp(t) for t in eventT]
    #
    # eventTex = Tex(eventSWE[2, :])
    # eventBe = BDE(eventPA)
    #
    # args = avg_genesis(eventSWE[2, :], eventSWE[3, :], eventTex / eventSWE[1, :], eventBe, eventSWE[0, :],
    #                    eventSWE[1, :], eventsweT, eventT, eventpaT)
    # shock_genesis(args,cons)
    # args['y'] = ydata[:eventSteps[0, eventIdx],eventIdx]

    #
    # nanpoints = (
    # np.isnan(args['V1']) +
    # np.isnan(args['V2']) +
    # np.isnan(args['N1']) +
    # np.isnan(args['N2']) +
    # np.isnan(args['T1']) +
    # np.isnan(args['T2']) +
    # np.isnan(args['Vp']) +
    # np.isnan(args['NHetoNp']) +
    # np.isnan(args['TextoTp']) +
    # np.isnan(args['Be']) +
    # np.isnan(args['isshock']) +
    # )

    return args


if __name__ == '__main__':
    eventSteps, eventSteps_swe, eventSteps_pa, swedata, padata, magdata, ydata, event_epoch, event_epoch_swe, event_epoch_pa = load_original_data_genesis()
    eventIdx = 1
    eventPA = padata[:,:eventSteps_pa[0,eventIdx],eventIdx]
    eventSWE = swedata[:, :eventSteps_swe[0, eventIdx], eventIdx]   # Np;Tp;Vp;He4top

    eventpaT = event_epoch_pa[:eventSteps_pa[0, eventIdx],eventIdx]
    eventpaT = (eventpaT - 719529.0) * 86400.0 - 8.0 * 3600.0
    eventpaT = [datetime.datetime.fromtimestamp(t) for t in eventpaT]

    eventsweT = event_epoch_swe[:eventSteps_swe[0, eventIdx],eventIdx]
    eventsweT = (eventsweT - 719529.0) * 86400.0 - 8.0 * 3600.0
    eventsweT = [datetime.datetime.fromtimestamp(t) for t in eventsweT]


    eventT = event_epoch[:eventSteps[0,eventIdx],eventIdx]
    eventT = (eventT - 719529.0) * 86400.0 - 8.0 * 3600.0
    eventT = [datetime.datetime.fromtimestamp(t) for t in eventT]


    eventTex =  Tex(eventSWE[2,:])
    eventBe = BDE(eventPA)

    args = avg_genesis(eventSWE[2,:],eventSWE[3,:],eventTex/eventSWE[1,:],eventBe,eventSWE[0,:],eventSWE[1,:],eventsweT,eventT,eventpaT)
    # shock_genesis(args,cons)

    print('test')