

# 彭镜宇(Peng jingyu)
# 2022-03-09
# 1 genesis在Wb=0的时候，输出TOcme可能还要调试一下
# 2 应该确保t_avg和输出数据的时间点time一致




import datetime
import preprocessing
import evaluate


constants_genesis = {'CA1':23,'CA2':1.15,'CA3':16.67,'CA4':1.0,
                     'CB1':23,'CB2':1.15,'CB3':16.67,'CB4':1.0,
                     'CT1':23,'CT2':1.15,'CT3':16.67,'CT4':1.0,
                     'ctime1':datetime.timedelta(hours=10),'ctime2':datetime.timedelta(hours=10), # for tocme
                     'BDE_threshold':2.0,
                     'Vcut':500.0,'C1':2.6e4,'C2':316.2,'C3':0.961,'C4':-1.42e5,'C5':510.0,'C6':0, # for Tex
                     't_avg':datetime.timedelta(hours=1), # for averaging
                     'Aout':0.06,'Tout':1.5,'Bout':0.4,
                     'tstay':datetime.timedelta(hours=18), 'tlag':datetime.timedelta(hours=6), 'tocme_threshold':0.4, # for genesis
                     'Vjump':40,'RN':1.4,'RT':1.5, # for shock
                     }


################ Genesis ################
def genesis_io(args,test=False,constants=constants_genesis,Wa=1,Wb=1):
    '''
    ——必须的数据7个：
    Vp: 质子速度 Km/s, shape=#timepoints
    Vp_time: datetime数据类型
    Np: 质子数密度cm-3, shape=#timepoints
    Np_time: datetime数据类型
    Tp: 质子温度 K, shape=#timepoints
    Tp_time: datetime数据类型
    time: 间隔为1hr（或2hr），datetime数据类型
    ——缺失，Wa设置为0
    NHetoNp: 阿尔法粒子（氦离子）和质子数密度之比, shape=#timepoints
    NHetoNp_time: datetime数据类型
    ——缺失，Wb设置为0
    PA: 电子投掷角，数据类型为numpy的array，shape=(#angles, #timepoints)，即第一个维度为不同的投掷角区间，第二个维度为时间点；PA用于判断双向电子散射
    PA_time: datetime数据类型
    ——测试模式才需要提供
    y: icme权威标记, 和time一一对应, shape=#timepoints


    :param args: 字典，包含程序所需的特征时间序列，以及每个特征的时间点（datetime数据类型）；此算法具有四种不同的工作模式，缺失电子投掷角或者He/p数密度比的情况下都可以工作
    算法会自动检测有没有这些数据，并根据所给的数据自动选择工作模式
    输入的数据不必是同分辨率的，但是数据和其时间必须一一对应，例如Vp与Vp_time必须一致
    另外，由于算法设计对数据求1hr窗口均值，数据时间分辨率应该高于1hr
    时间序列应该是按照时间单调的
    :param test: 是否是测试模式，如果是，则args里必须包含y的数据，及人工标记的time每个时间点的icme标记；y应该和最后输出的时间time一致
    :param constants: 算法中用到的参数；可能需要根据数据的不同进行调整。
    :param Wa: 如果输入相应数据，是否默认使用阿尔法粒子（氦离子）和质子数密度之比，0为不使用，1为使用；如果没有输入NHetoNp数据，可以不管这个参数，程序会自动设置为0
    :param Wb: 如果输入相应数据，是否默认使用电子投掷角，0为不使用，1为使用；如果没有输入PA数据，可以不管这个参数，程序会自动设置为0
    :return:
    '''

    ### 检查数据是否完整
    if 'Vp' not in args or 'Vp_time' not in args or 'Np' not in args or 'Np_time' not in args or 'Tp' not in args or 'Tp_time' not in args or 'time' not in args:
        raise ValueError('缺少必要的数据，请检查数据是否完整，或字典key的名称是否正确("Vp","Vp_time","Np","Np_time","Tp","Tp_time","time")')
    ### 设置Wa
    if 'NHetoNp' not in args:
        Wa = 0
        print('没有检测到"NHetoNp"数据，Wa设置为0. 如果输入了此数据，请检查字典key名称是否正确')
    else:
        assert 'NHetoNp_time' in args, '检测到"NHetoNp"数据, 但没有"NHetoNp_time"数据，请检查是否输入该时间数据，或检查字典key名称是否正确'
    ### 设置Wb
    if 'PA' not in args:
        Wb = 0
        print('没有检测到"PA"数据，Wb设置为0. 如果输入了此数据，请检查字典key名称是否正确')
    else:
        assert 'PA_time' in args, '检测到"PA"数据, 但没有"PA_time"数据，请检查是否输入该时间数据，或检查字典key名称是否正确'

    ### 计算 Tex （根据Vp估计的非ICME太阳风质子温度，对于ICME，质子温度应该远远低于Tex）
    Tp_ex = preprocessing.Tex(args['Vp'],constants) # in K
    assert args['Tp_time'] == args['Vp_time'], "Tp_time和Vp_time应该是相同的，如果这两个数据时间不同，还应该再插值到Vp_time"
    args['TextoTp'] = Tp_ex/args['Tp']
    ### 如果Wb，计算BDE
    if Wb:
        Be = preprocessing.BDE(args['PA'],threshold=constants['BDE_threshold'])
        args['Be'] = Be
    ### 把数据滑动平均到time上
    args_avg = preprocessing.avg_genesis(args,constants['t_avg'],Wa=Wa,Wb=Wb)
    ### 判断是否有shock
    preprocessing.shock_genesis(args_avg,constants)
    ### 给出time每个时间点的icme判定
    icme = evaluate.Genesis(args_avg,constants,Wa=Wa,Wb=Wb)

    return icme

if __name__ == '__main__':
    eventSteps, eventSteps_swe, eventSteps_pa, swedata, padata, ydata, event_epoch, event_epoch_swe, event_epoch_pa = preprocessing.load_original_data_genesis()
    eventIdx = 112
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

    args = {
        'Np':eventSWE[0,:],'Np_time':eventsweT,
        'Tp':eventSWE[1,:],'Tp_time':eventsweT,
        'Vp':eventSWE[2,:],'Vp_time':eventsweT,
        'NHetoNp':eventSWE[3,:],'NHetoNp_time':eventsweT,
        'PA':eventPA,'PA_time':eventpaT,
        'time':eventT,
        'y':ydata[:eventSteps[0, eventIdx],eventIdx],
    }

    icme = genesis_io(args)
    print('genesis')

