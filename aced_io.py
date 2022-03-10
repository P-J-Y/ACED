

# 彭镜宇(Peng jingyu)
# 2022-03-09
# 1 genesis在Wb=0的时候，输出TOcme可能还要调试一下
# 2 应该确保t_avg和输出数据的时间点time一致




import datetime
import os

import numpy as np
from matplotlib import pyplot as plt

import preprocessing
import evaluate



################ constants ################
constants_genesis = {'CA1':23,'CA2':1.15,'CA3':16.67,'CA4':1.0,
                     'CB1':23,'CB2':1.15,'CB3':16.67,'CB4':1.0,
                     'CT1':23,'CT2':1.15,'CT3':16.67,'CT4':1.0,
                     'ctime1':datetime.timedelta(hours=5),'ctime2':datetime.timedelta(hours=25), # for tocme
                     'BDE_threshold':2.0,
                     'Vcut':500.0,'C1':2.6e4,'C2':316.2,'C3':0.961,'C4':-1.42e5,'C5':510.0,'C6':0, # for Tex
                     't_avg':datetime.timedelta(hours=1), # for averaging
                     'Aout':0.06,'Tout':1.5,'Bout':0.4,
                     'tstay':datetime.timedelta(hours=18), 'tlag':datetime.timedelta(hours=6), 'tocme_threshold':0.4, # for genesis
                     'Vjump':40,'RN':1.4,'RT':1.5, # for shock
                     }

plot_features_genesis = ['Vp','Tp','TextoTp','NHetoNp','PA','Be','TOCME']
# plot_features_genesis = ['Vp','Tp','TextoTp','NHetoNp','PA']
plot_features_swics = ['Vp','O76']
plot_features_xb = ['Vp','Np','Tp','Mag']




################ functions ################
def plot_genesis(args,args_avg,icmes,ys=None,eval=None,
                 plot_features=plot_features_genesis,
                 figpath='image/eval/Genesis/test'):

    plot_in_args = ['Vp','TextoTp','NHetoNp',]
    plot_in_args_avg = ['TOCME','Be']
    feature2title = {'Vp':'Vp Km/s','Tp':'Tp 10$^4$K','TextoTp':'Tex/Tp','NHetoNp':'N$_{He}$/Np','Be':'BDE','TOCME':'TOCME'}
    feature2time = {'Vp': 'Vp_time',
                    'Tp': 'Tp_time',
                    'TextoTp': 'Vp_time',
                    'NHetoNp': 'NHetoNp_time',
                    'PA': 'PA_time',
                    'Be': 'time',
                    'TOCME': 'time'}

    panelnum = len(plot_features)+1
    eventnum = len(icmes)
    fig,axes = plt.subplots(nrows=panelnum,ncols=1,figsize=(7,2*panelnum))

    if eval is not None:
        axes[0].set_title('Genesis P={:.2f} R={:.2f}'.format(eval['precision'], eval['recall']), fontsize=16)
        for i in range(eventnum):
            if i == 0:
                line1, = axes[-1].plot([icmes[i][0], icmes[i][1]], [3 / 4, 3 / 4], 'r-', linewidth=3)
            else:
                axes[-1].plot([icmes[i][0], icmes[i][1]], [3 / 4, 3 / 4], 'r-', linewidth=3)
        for i in range(len(ys)):
            if i ==0:
                line2, = axes[-1].plot([ys[i][0], ys[i][1]], [2 / 4, 2 / 4], 'b-', linewidth=3)
            else:
                axes[-1].plot([ys[i][0], ys[i][1]], [2 / 4, 2 / 4], 'b-', linewidth=3)
        for i in range(len(eval['overlap'])):
            if i == 0:
                line3, = axes[-1].plot([eval['overlap'][i][0], eval['overlap'][i][1]], [1 / 4, 1 / 4], 'g-', linewidth=3)
            else:
                axes[-1].plot([eval['overlap'][i][0], eval['overlap'][i][1]], [1 / 4, 1 / 4], 'g-', linewidth=3)
        axes[-1].legend(handles=[line1, line2, line3], labels=['Genesis', 'list', 'overlap'])

    else:
        axes[0].set_title('Genesis', fontsize=16)
        for i in range(eventnum):
            axes[-1].plot([icmes[i][0], icmes[i][1]], [2 / 4, 2 / 4], 'r-', linewidth=3)
    # set ylabel
    axes[-1].set_ylabel('ICME', fontsize=16)
    # set xlabel
    axes[-1].set_xlabel('Time', fontsize=16)
    # set xlim
    axes[-1].set_xlim(args['time'][0], args['time'][-1])
    # set ylim
    axes[-1].set_ylim([0, 1])
    # close y ticks
    axes[-1].set_yticklabels([])

    for i in range(len(plot_features)):
        if plot_features[i] in plot_in_args:
            axes[i].plot(args[feature2time[plot_features[i]]], args[plot_features[i]], 'b-', linewidth=1)
            axes[i].set_ylabel(feature2title[plot_features[i]], fontsize=16)
            axes[i].set_xlim(args['time'][0], args['time'][-1])
            axes[i].set_xticklabels([])
            if plot_features[i] in ['Be','TOCME']:
                axes[i].set_ylim([0, 1])
        elif plot_features[i] in plot_in_args_avg:
            axes[i].plot(args_avg[feature2time[plot_features[i]]], args_avg[plot_features[i]], 'b-', linewidth=1)
            axes[i].set_ylabel(feature2title[plot_features[i]], fontsize=16)
            axes[i].set_xlim(args['time'][0], args['time'][-1])
            axes[i].set_xticklabels([])
            if plot_features[i] in ['Be','TOCME']:
                axes[i].set_ylim([0, 1])
        elif plot_features[i] == 'PA':
            anglenum = np.shape(args['PA'])[0]
            deltaangle = 180 / anglenum
            angles = np.arange(0, 180, deltaangle)+deltaangle/2
            X,Y = np.meshgrid(args['PA_time'],angles)
            axes[i].contourf(X,Y,10*np.log10(args['PA']),cmap='jet')
            axes[i].set_ylabel('PA(#) °(dB)', fontsize=16)
            axes[i].set_xticklabels([])
            axes[i].set_xlim(args['time'][0], args['time'][-1])
            axes[i].set_ylim([0, 180])
        elif plot_features[i] == 'Tp':
            axes[i].plot(args['Tp_time'], args['Tp'] * 1e-4, 'b-', linewidth=1)
            axes[i].set_ylabel(feature2title['Tp'], fontsize=16)
            axes[i].set_xlim(args['time'][0], args['time'][-1])
            axes[i].set_xticklabels([])


    # save figure
    if not os.path.exists(figpath):
        os.makedirs(figpath)
    plt.savefig(figpath+'/'+args['time'][0].strftime('%Y%m%d%H%M')+'_'+args['time'][-1].strftime('%Y%m%d%H%M')+'.png')

def plot_swics(args,icmes,ys=None,eval=None,
                 plot_features=plot_features_swics,
                 figpath='image/eval/SWICS/test'):
    feature2title = {'Vp':'Vp Km/s','O76':'O7/O6'}

    panelnum = len(plot_features)+1
    eventnum = len(icmes)
    fig,axes = plt.subplots(nrows=panelnum,ncols=1,figsize=(10,4*panelnum))

    if eval is not None:
        axes[0].set_title('SWICS P={:.2f} R={:.2f}'.format(eval['precision'], eval['recall']), fontsize=16)
        for i in range(eventnum):
            if i == 0:
                line1, = axes[-1].plot([icmes[i][0], icmes[i][1]], [3 / 4, 3 / 4], 'r-', linewidth=3)
            else:
                axes[-1].plot([icmes[i][0], icmes[i][1]], [3 / 4, 3 / 4], 'r-', linewidth=3)
        for i in range(len(ys)):
            if i ==0:
                line2, = axes[-1].plot([ys[i][0], ys[i][1]], [2 / 4, 2 / 4], 'b-', linewidth=3)
            else:
                axes[-1].plot([ys[i][0], ys[i][1]], [2 / 4, 2 / 4], 'b-', linewidth=3)
        for i in range(len(eval['overlap'])):
            if i == 0:
                line3, = axes[-1].plot([eval['overlap'][i][0], eval['overlap'][i][1]], [1 / 4, 1 / 4], 'g-', linewidth=3)
            else:
                axes[-1].plot([eval['overlap'][i][0], eval['overlap'][i][1]], [1 / 4, 1 / 4], 'g-', linewidth=3)
        axes[-1].legend(handles=[line1, line2, line3], labels=['SWICS', 'list', 'overlap'])

    else:
        axes[0].set_title('SWICS', fontsize=16)
        for i in range(eventnum):
            axes[-1].plot([icmes[i][0], icmes[i][1]], [2 / 4, 2 / 4], 'r-', linewidth=3)
    # set ylabel
    axes[-1].set_ylabel('ICME', fontsize=16)
    # set xlabel
    axes[-1].set_xlabel('Time', fontsize=16)
    # set xlim
    axes[-1].set_xlim(args['time'][0], args['time'][-1])
    # set ylim
    axes[-1].set_ylim([0, 1])
    # close y ticks
    axes[-1].set_yticklabels([])

    for i in range(len(plot_features)):
        axes[i].plot(args['time'], args[plot_features[i]], 'b-', linewidth=1)
        axes[i].set_ylabel(feature2title[plot_features[i]], fontsize=16)
        axes[i].set_xlim(args['time'][0], args['time'][-1])
        axes[i].set_xticklabels([])

    # save figure
    if not os.path.exists(figpath):
        os.makedirs(figpath)
    plt.savefig(figpath+'/'+args['time'][0].strftime('%Y%m%d%H%M')+'_'+args['time'][-1].strftime('%Y%m%d%H%M')+'.png')

def plot_xb(args,icmes,ys=None,eval=None,
                 plot_features=plot_features_xb,
                 figpath='image/eval/XB/test'):
    feature2title = {'Vp':'Vp Km/s','Np':'Np cm$^{-3}$','Tp':'Tp 10$^4$K','Mag':'B nT'}

    panelnum = len(plot_features)+1
    eventnum = len(icmes)
    fig,axes = plt.subplots(nrows=panelnum,ncols=1,figsize=(10,4*panelnum))

    if eval is not None:
        axes[0].set_title('XB P={:.2f} R={:.2f}'.format(eval['precision'], eval['recall']), fontsize=16)
        for i in range(eventnum):
            if i == 0:
                line1, = axes[-1].plot([icmes[i][0], icmes[i][1]], [3 / 4, 3 / 4], 'r-', linewidth=3)
            else:
                axes[-1].plot([icmes[i][0], icmes[i][1]], [3 / 4, 3 / 4], 'r-', linewidth=3)
        for i in range(len(ys)):
            if i ==0:
                line2, = axes[-1].plot([ys[i][0], ys[i][1]], [2 / 4, 2 / 4], 'b-', linewidth=3)
            else:
                axes[-1].plot([ys[i][0], ys[i][1]], [2 / 4, 2 / 4], 'b-', linewidth=3)
        for i in range(len(eval['overlap'])):
            if i == 0:
                line3, = axes[-1].plot([eval['overlap'][i][0], eval['overlap'][i][1]], [1 / 4, 1 / 4], 'g-', linewidth=3)
            else:
                axes[-1].plot([eval['overlap'][i][0], eval['overlap'][i][1]], [1 / 4, 1 / 4], 'g-', linewidth=3)
        axes[-1].legend(handles=[line1, line2, line3], labels=['XB', 'list', 'overlap'])

    else:
        axes[0].set_title('XB', fontsize=16)
        for i in range(eventnum):
            axes[-1].plot([icmes[i][0], icmes[i][1]], [2 / 4, 2 / 4], 'r-', linewidth=3)
    # set ylabel
    axes[-1].set_ylabel('ICME', fontsize=16)
    # set xlabel
    axes[-1].set_xlabel('Time', fontsize=16)
    # set xlim
    axes[-1].set_xlim(args['time'][0], args['time'][-1])
    # set ylim
    axes[-1].set_ylim([0, 1])
    # close y ticks
    axes[-1].set_yticklabels([])

    for i in range(len(plot_features)):
        if plot_features[i] == 'Tp':
            axes[i].plot(args['time'], args['Tp']*1e-4, 'b-', linewidth=1)
            axes[i].set_ylabel(feature2title[plot_features[i]], fontsize=16)
            axes[i].set_xlim(args['time'][0], args['time'][-1])
            axes[i].set_xticklabels([])
        else:
            axes[i].plot(args['time'], args[plot_features[i]], 'b-', linewidth=1)
            axes[i].set_ylabel(feature2title[plot_features[i]], fontsize=16)
            axes[i].set_xlim(args['time'][0], args['time'][-1])
            axes[i].set_xticklabels([])

    # save figure
    if not os.path.exists(figpath):
        os.makedirs(figpath)
    plt.savefig(figpath+'/'+args['time'][0].strftime('%Y%m%d%H%M')+'_'+args['time'][-1].strftime('%Y%m%d%H%M')+'.png')


############### SWICS ###############
def swics_io(args,test=False,ifplot=False,plot_features=plot_features_swics):
    '''

    :param args: 应该包含如下数据，质子速度与O76的时间分辨率应该是一样的（建议1hr或2hr），即应该在预处理时插值到所需的时间点
    'Vp': km/s
    'O76': O7/O6价态粒子数之比
    'time': 输出数据的时间点
    ——测试模式才需要提供
    y: icme权威标记, 和time一一对应, shape=#timepoints
    :param test: 是否是测试模式，如果是，则args里必须包含y的数据，即人工标记的time每个时间点的icme标记；y应该和最后输出的时间time一致
    :param ifplot: 是否画图
    :param plot_features: 需要画出哪些参数
    :param feature2time: 每个参数画图时对应的时间是哪个
    :return:
    '''

    ### 检查数据是否完整
    if 'Vp' not in args or 'O76' not in args or 'time' not in args:
        raise ValueError('缺少必要的数据，请检查数据是否完整，或字典key的名称是否正确("Vp","O76","time")')
    ### 给出time每个时间点的icme判定
    icme = evaluate.SWICS(args)
    icmes = evaluate.checkIcme(icme,args)
    if icmes is None:
        print('Final: No ICME detected!')
        return None,None
    if test:
        assert 'y' in args, "没有检测到输入权威icme标记y，请关闭测试模式，或检查输入数据"
        ys = evaluate.checkIcme(args['y'],args)
        eval_swics = evaluate.evaluateIcme(icmes, ys)
        if eval_swics['recall'] == 0:
            print('Final: No ICME detected!')
            return None,None
        elif ifplot:
            plot_swics(args,icmes,ys=ys,eval=eval_swics,plot_features=plot_features)
    else:
        if ifplot:
            plot_swics(args,icmes,plot_features=plot_features)

    return icme, args


################ XB ################
def xb_io(args,test=False,ifplot=False,plot_features=plot_features_xb):
    '''

    :param args: 应该包含如下数据，时间分辨率应该是一样的（建议1hr或2hr），即应该在预处理时插值到所需的时间点
    'Vp': km/s
    'Np': cm-3
    'Tp': K
    'Mag': nT, 磁场绝对值
    'time': 输出数据的时间点
    ——测试模式才需要提供
    y: icme权威标记, 和time一一对应, shape=#timepoints
    :param test: 是否是测试模式，如果是，则args里必须包含y的数据，即人工标记的time每个时间点的icme标记；y应该和最后输出的时间time一致
    :param ifplot: 是否画图
    :param plot_features: 需要画出哪些参数
    :param feature2time: 每个参数画图时对应的时间是哪个
    :return:
    '''

    ### 检查数据是否完整
    if 'Vp' not in args or 'Np' not in args or 'Tp' not in args or 'Mag' not in args or 'time' not in args:
        raise ValueError('缺少必要的数据，请检查数据是否完整，或字典key的名称是否正确("Vp","Np","Tp","Mag","time")')
    ### 给出time每个时间点的icme判定
    icme = evaluate.XB(args)
    icmes = evaluate.checkIcme(icme,args)
    if icmes is None:
        print('Final: No ICME detected!')
        return None, None
    if test:
        assert 'y' in args, "没有检测到输入权威icme标记y，请关闭测试模式，或检查输入数据"
        ys = evaluate.checkIcme(args['y'],args)
        eval_xb = evaluate.evaluateIcme(icmes, ys)
        if eval_xb['recall'] == 0:
            print('Final: No ICME detected!')
            return None, None
        elif ifplot:
            plot_xb(args,icmes,ys=ys,eval=eval_xb,plot_features=plot_features)
    else:
        if ifplot:
            plot_xb(args,icmes,plot_features=plot_features)

    return icme,args










################ Genesis ################
def genesis_io(args,test=False,constants=constants_genesis,Wa=1,Wb=1,ifplot=False,plot_features=plot_features_genesis):
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
    :param test: 是否是测试模式，如果是，则args里必须包含y的数据，即人工标记的time每个时间点的icme标记；y应该和最后输出的时间time一致
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
    icmes = evaluate.checkIcme(icme,args_avg)
    if icmes is None:
        print('Final: No ICME detected!')
        return None,None
    if test:
        assert 'y' in args, "没有检测到输入权威icme标记y，请关闭测试模式，或检查输入数据"
        ys = evaluate.checkIcme(args['y'],args_avg)
        eval_genesis = evaluate.evaluateIcme(icmes, ys)
        if eval_genesis['recall'] == 0:
            print('Final: No ICME detected!')
            return None,None
        elif ifplot:
            plot_genesis(args,args_avg,icmes,ys=ys,eval=eval_genesis,plot_features=plot_features)
    else:
        if ifplot:
            plot_genesis(args,args_avg,icmes,plot_features=plot_features)

    return icme,args_avg

if __name__ == '__main__':
    # eventSteps, eventSteps_swe, eventSteps_pa, swedata, padata, ydata, event_epoch, event_epoch_swe, event_epoch_pa = preprocessing.load_original_data_genesis()
    # eventIdx = 100
    # eventPA = padata[:,:eventSteps_pa[0,eventIdx],eventIdx]
    # eventSWE = swedata[:, :eventSteps_swe[0, eventIdx], eventIdx]   # Np;Tp;Vp;He4top
    #
    # eventpaT = event_epoch_pa[:eventSteps_pa[0, eventIdx],eventIdx]
    # eventpaT = (eventpaT - 719529.0) * 86400.0 - 8.0 * 3600.0
    # eventpaT = [datetime.datetime.fromtimestamp(t) for t in eventpaT]
    #
    # eventsweT = event_epoch_swe[:eventSteps_swe[0, eventIdx],eventIdx]
    # eventsweT = (eventsweT - 719529.0) * 86400.0 - 8.0 * 3600.0
    # eventsweT = [datetime.datetime.fromtimestamp(t) for t in eventsweT]
    #
    #
    # eventT = event_epoch[:eventSteps[0,eventIdx],eventIdx]
    # eventT = (eventT - 719529.0) * 86400.0 - 8.0 * 3600.0
    # eventT = [datetime.datetime.fromtimestamp(t) for t in eventT]
    #
    # args = {
    #     'Np':eventSWE[0,:],'Np_time':eventsweT,
    #     'Tp':eventSWE[1,:],'Tp_time':eventsweT,
    #     'Vp':eventSWE[2,:],'Vp_time':eventsweT,
    #     'NHetoNp':eventSWE[3,:],'NHetoNp_time':eventsweT,
    #     'PA':eventPA,'PA_time':eventpaT,
    #     'time':eventT,
    #     'y':ydata[:eventSteps[0, eventIdx],eventIdx],
    # }
    #
    # icme,args_avg = genesis_io(args,test=True,Wa=1,Wb=1,ifplot=1,)

    # xdata,ydata,eventTimes,eventSteps = evaluate.loaddata_swics()
    # eventIdx = 20
    # eventTime = eventTimes[:eventSteps[0, eventIdx], eventIdx]
    # eventTime = (eventTime - 719529.0) * 86400.0 - 8.0 * 3600.0
    # eventTime = [datetime.datetime.fromtimestamp(t) for t in eventTime]
    # args = {
    #     'time':eventTime,
    #     'Vp':xdata[1, :eventSteps[0, eventIdx], eventIdx],
    #     'O76':xdata[0, :eventSteps[0, eventIdx], eventIdx],
    #     'y':ydata[:eventSteps[0, eventIdx], eventIdx],
    # }
    # icme,args = swics_io(args,test=True,ifplot=True,plot_features=plot_features_swics)

    xdata,ydata,eventTimes,eventSteps = evaluate.loaddata_xb()
    eventIdx = 20
    eventTime = eventTimes[:eventSteps[0, eventIdx], eventIdx]
    eventTime = (eventTime - 719529.0) * 86400.0 - 8.0 * 3600.0
    eventTime = [datetime.datetime.fromtimestamp(t) for t in eventTime]
    args = {
        'time':eventTime,
        'Vp':xdata[2, :eventSteps[0, eventIdx], eventIdx],
        'Tp':xdata[1, :eventSteps[0, eventIdx], eventIdx],
        'Np':xdata[0, :eventSteps[0, eventIdx], eventIdx],
        'Mag':xdata[3, :eventSteps[0, eventIdx], eventIdx],
        'y':ydata[:eventSteps[0, eventIdx], eventIdx],
    }
    icme,args = xb_io(args,test=True,ifplot=True,plot_features=plot_features_xb)

    print('genesis')

