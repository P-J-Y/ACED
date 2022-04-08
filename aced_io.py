

# 彭镜宇(Peng jingyu)
# 2022-03-09
# 1 genesis在Wb=0的时候，输出TOcme可能还要调试一下；包括拟合Tex(Vp)
# 2 应该确保t_avg和输出数据的时间点time一致
# 3 genesis Vp_time和Tp_time不一致的时候还没考虑
# 4 NN 模型还没有封装完成




import datetime
import os

import h5py
import numpy as np

from matplotlib import pyplot as plt
import plotly.offline as py
import plotly.graph_objects as go
import preprocessing
import evaluate
import loadData



################ constants ################
constants_genesis = {'CA1':23,'CA2':1.15,'CA3':16.67,'CA4':1.0,
                     'CB1':3,'CB2':1.5,'CB3':2,'CB4':1.,
                     'CT1':0.2,'CT2':1.3,'CT3':0.15,'CT4':1,
                     'ctime1':datetime.timedelta(hours=5),'ctime2':datetime.timedelta(hours=25), # for tocme
                     'BDE_threshold':2.5,# 这是我根据ACE的数据调整的
                     'Vcut':500.0,'C1':2.6e4,'C2':316.2,'C3':0.961,'C4':-1.42e5,'C5':510.0,'C6':0, # for Tex
                     't_avg':datetime.timedelta(hours=1), # for averaging
                     'Aout':0.06,'Tout':1.5,'Bout':0.4,
                     # Tout=1.5 可能有点太小了，容易把ICME过多的识别
                     'tstay':datetime.timedelta(hours=18), 'tlag':datetime.timedelta(hours=6), 'tocme_threshold':0.4, # for genesis
                     'Vjump':40,'RN':1.4,'RT':1.5, # for shock
                     }

# plot_features_genesis = ['Vp','Mag','Tp','TextoTp','NHetoNp','PA','Be','TOCME']
plot_features_genesis = ['Vp','Mag','Tp','TextoTp','TOCME']
plot_features_swics = ['Vp','O76']
plot_features_xb = ['Vp','Np','Tp','Mag']
plot_features_nn = ['Vp','Np','Tp','Mag','dbrms','PA','delta','lambda']

list_features = ['Vp','Tp','Np','Mag','O76','Be','TextoTp','NHetoNp']
figpath_swics = 'image/eval/SWICS/test/tot'
figpath_xb = 'image/eval/xb/test/tot'
figpath_genesis = 'image/eval/genesis/test/tot'


################ functions ################

def listIcmes(args,
              list_features=list_features,
              savejson=True,
              savepath='icmelist/test',
              filename='icmetest.json'):
    '''
    计算每个识别出的icme的参数均值，并输出一个json表格
    :param icmes:
    :param args:
    :param list_features:
    :return:
    '''
    icmes = args['icmes']
    icmelist = []
    for i in range(len(icmes)):
        features = {
            'start_time':icmes[i][0], 'end_time':icmes[i][1],
        }
        timepoints = (args['time']>icmes[i][0]) * (args['time']<icmes[i][1])
        for feature in list_features:
            if feature not in args:
                features[feature] = None
            else:
                features[feature] = np.nanmean(args[feature][timepoints])

        icmelist.append(features)
    if savejson:
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        with open(savepath+'/'+filename,'w') as f:
            f.write(str(icmelist))

    return icmelist


def plot_genesis(args,args_avg,icmes,ys=None,eval=None,
                 plot_features=plot_features_genesis,
                 figpath=figpath_genesis,) :

    plot_in_args = ['Vp','TextoTp','NHetoNp','Mag']
    plot_in_args_avg = ['TOCME','Be']
    feature2title = {'Vp':'Vp Km/s','Tp':'Tp 10<sup>4</sup>K','TextoTp':'Tex/Tp','NHetoNp':'N<sub>He</sub>/Np','Be':'BDE','TOCME':'TOCME','PA':'PA(#) °(dB)','Mag':'B nT'}
    feature2time = {'Vp': 'Vp_time',
                    'Tp': 'Tp_time',
                    'TextoTp': 'Vp_time',
                    'NHetoNp': 'NHetoNp_time',
                    'PA': 'PA_time',
                    'Be': 'time',
                    'TOCME': 'time',
                    'Mag': 'Mag_time',
                    }

    panelnum = len(plot_features)+1
    eventnum = len(icmes)
    # fig,axes = plt.subplots(nrows=panelnum,
    #                         ncols=1,
    #                         figsize=(9,1.5*panelnum),
    #                         gridspec_kw = {'wspace':0, 'hspace':0})
    #
    # if eval is not None:
    #     axes[0].set_title('Genesis P={:.2f} R={:.2f}'.format(eval['precision'], eval['recall']), fontsize=16)
    #     for i in range(eventnum):
    #         if i == 0:
    #             line1, = axes[-1].plot([icmes[i][0], icmes[i][1]], [3 / 4, 3 / 4], 'r-', linewidth=3)
    #         else:
    #             axes[-1].plot([icmes[i][0], icmes[i][1]], [3 / 4, 3 / 4], 'r-', linewidth=3)
    #     for i in range(len(ys)):
    #         if i ==0:
    #             line2, = axes[-1].plot([ys[i][0], ys[i][1]], [2 / 4, 2 / 4], 'b-', linewidth=3)
    #         else:
    #             axes[-1].plot([ys[i][0], ys[i][1]], [2 / 4, 2 / 4], 'b-', linewidth=3)
    #     for i in range(len(eval['overlap'])):
    #         if i == 0:
    #             line3, = axes[-1].plot([eval['overlap'][i][0], eval['overlap'][i][1]], [1 / 4, 1 / 4], 'g-', linewidth=3)
    #         else:
    #             axes[-1].plot([eval['overlap'][i][0], eval['overlap'][i][1]], [1 / 4, 1 / 4], 'g-', linewidth=3)
    #     axes[-1].legend(handles=[line1, line2, line3], labels=['Genesis', 'list', 'overlap'])
    #
    # else:
    #     axes[0].set_title('Genesis', fontsize=16)
    #     for i in range(eventnum):
    #         axes[-1].plot([icmes[i][0], icmes[i][1]], [2 / 4, 2 / 4], 'r-', linewidth=3)
    # # set ylabel
    # axes[-1].set_ylabel('ICME', fontsize=16)
    # # set xlabel
    # axes[-1].set_xlabel('Time', fontsize=16)
    # # set xlim
    # axes[-1].set_xlim(args['time'][0], args['time'][-1])
    # # set ylim
    # axes[-1].set_ylim([0, 1])
    # # close y ticks
    # axes[-1].set_yticklabels([])
    #
    # for i in range(len(plot_features)):
    #     if plot_features[i] not in args and plot_features[i] not in args_avg:
    #         continue
    #     if plot_features[i] in plot_in_args:
    #         axes[i].plot(args[feature2time[plot_features[i]]], args[plot_features[i]], 'b-', linewidth=1)
    #         axes[i].set_ylabel(feature2title[plot_features[i]], fontsize=16)
    #         axes[i].set_xlim(args['time'][0], args['time'][-1])
    #         axes[i].set_xticklabels([])
    #         if plot_features[i] in ['Be','TOCME']:
    #             axes[i].set_ylim([0, 1])
    #     elif plot_features[i] in plot_in_args_avg:
    #         axes[i].plot(args_avg[feature2time[plot_features[i]]], args_avg[plot_features[i]], 'b-', linewidth=1)
    #         axes[i].set_ylabel(feature2title[plot_features[i]], fontsize=16)
    #         axes[i].set_xlim(args['time'][0], args['time'][-1])
    #         axes[i].set_xticklabels([])
    #         if plot_features[i] in ['Be','TOCME']:
    #             axes[i].set_ylim([0, 1])
    #     elif plot_features[i] == 'PA':
    #         anglenum = np.shape(args['PA'])[0]
    #         deltaangle = 180 / anglenum
    #         angles = np.arange(0, 180, deltaangle)+deltaangle/2
    #         X,Y = np.meshgrid(args['PA_time'],angles)
    #         axes[i].contourf(X,Y,10*np.log10(args['PA']),cmap='jet')
    #         axes[i].set_ylabel('PA(#) °(dB)', fontsize=16)
    #         axes[i].set_xticklabels([])
    #         axes[i].set_xlim(args['time'][0], args['time'][-1])
    #         axes[i].set_ylim([0, 180])
    #     elif plot_features[i] == 'Tp':
    #         axes[i].plot(args['Tp_time'], args['Tp'] * 1e-4, 'b-', linewidth=1)
    #         axes[i].set_ylabel(feature2title['Tp'], fontsize=16)
    #         axes[i].set_xlim(args['time'][0], args['time'][-1])
    #         axes[i].set_xticklabels([])
    #
    #
    # # save figure
    # if not os.path.exists(figpath):
    #     os.makedirs(figpath)
    # plt.savefig(figpath+'/'+args['time'][0].strftime('%Y%m%d%H%M')+'_'+args['time'][-1].strftime('%Y%m%d%H%M')+'.png')


    fig = go.Figure().set_subplots(rows=panelnum,
                                   cols=1,
                                   shared_xaxes=True,

                                   )
    if eval is not None:
        fig.update_layout(title_text='Genesis P={:.2f} R={:.2f}'.
                          format(eval['precision'], eval['recall']),
                          )
        for i in range(eventnum):
            if i == 0:
                fig.add_trace(
                    go.Scatter(
                        x=[icmes[i][0], icmes[i][1]], y=[3 / 4, 3 / 4],
                        mode='lines',
                        line=dict(color='red', width=10),
                        name='ICME',
                    ),
                    row=panelnum,
                    col=1,
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=[icmes[i][0], icmes[i][1]],
                        y=[3 / 4, 3 / 4],
                        mode='lines',
                        line=dict(color='red', width=10),
                        name='ICME',
                        showlegend=False,
                    ),
                    row=panelnum,
                    col=1,
                )
        for i in range(len(ys)):
            if i ==0:
                fig.add_trace(
                    go.Scatter(
                        x=[ys[i][0], ys[i][1]],
                        y=[2 / 4, 2 / 4],
                        mode='lines',
                        line=dict(color='blue', width=10),
                        name='list',
                    ),
                    row=panelnum,
                    col=1,
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=[ys[i][0], ys[i][1]],
                        y=[2 / 4, 2 / 4],
                        mode='lines',
                        line=dict(color='blue', width=10),
                        name='list',
                        showlegend=False,
                    ),
                    row=panelnum,
                    col=1,
                )
        for i in range(len(eval['overlap'])):
            if i == 0:
                fig.add_trace(
                    go.Scatter(
                        x=[eval['overlap'][i][0],
                           eval['overlap'][i][1]],
                        y=[1 / 4, 1 / 4],
                        mode='lines',
                        line=dict(color='green', width=10),
                        name='overlap',
                    ),
                    row=panelnum,
                    col=1,
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=[eval['overlap'][i][0], eval['overlap'][i][1]],
                        y=[1 / 4, 1 / 4],
                        mode='lines',
                        line=dict(color='green', width=10),
                        name='overlap',
                        showlegend=False,
                    ),
                    row=panelnum,
                    col=1,
                )
    else:
        fig.update_layout(title_text='Genesis',
                          )
        for i in range(eventnum):
            if i==0:
                fig.add_trace(
                    go.Scatter(
                        x=[icmes[i][0], icmes[i][1]],
                        y=[2 / 4, 2 / 4],
                        mode='lines',
                        line=dict(color='red', width=10),
                        name='ICME',
                    ),
                    row=panelnum,
                    col=1,
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=[icmes[i][0], icmes[i][1]],
                        y=[2 / 4, 2 / 4],
                        mode='lines',
                        line=dict(color='red', width=10),
                        name='ICME',
                        showlegend=False,
                    ),
                    row=panelnum,
                    col=1,
                )
    fig.update_layout(legend=dict(
        yanchor="bottom",
        y=0.01,
        xanchor="left",
        x=0.01
    ),
    )
    fig.update_yaxes(title_text='ICME', range=[0, 1], row=panelnum, col=1)

    for i in range(len(plot_features)):
        if plot_features[i] not in args and plot_features[i] not in args_avg:
            continue
        if plot_features[i] in plot_in_args:
            fig.add_trace(
                go.Scatter(
                    x=args[feature2time[plot_features[i]]],
                    y=args[plot_features[i]],
                    mode='lines',
                    line=dict(color='black', width=1),
                    name=plot_features[i],
                    showlegend=False,
                ),
                row=i+1,
                col=1,
            )
        elif plot_features[i] in plot_in_args_avg:
            fig.add_trace(
                go.Scatter(
                    x=args_avg[feature2time[plot_features[i]]],
                    y=args_avg[plot_features[i]],
                    mode='lines',
                    line=dict(color='black', width=1),
                    name=plot_features[i],
                    showlegend=False,
                ),
                row=i+1,
                col=1,
            )
        elif plot_features[i] == 'PA':
            anglenum = np.shape(args['PA'])[0]
            deltaangle = 180 / anglenum
            angles = np.arange(0, 180, deltaangle)+deltaangle/2
            # X,Y = np.meshgrid(args['PA_time'],angles)
            fig.add_trace(
                go.Heatmap(
                    x=args['PA_time'],
                    y=angles,
                    z=10*np.log10(args['PA']),
                    colorscale='Jet',
                    showscale=False,
                    name=plot_features[i],
                    showlegend=False,
                    type='heatmap',
                ),
                row=i+1,
                col=1,
            )
        elif plot_features[i] == 'Tp':
            fig.add_trace(
                go.Scatter(
                    x=args['time'],
                    y=args[plot_features[i]]/10000,
                    mode='lines',
                    line=dict(color='black', width=1),
                    name=plot_features[i],
                    showlegend=False,
                ),
                row=i+1,
                col=1)
            fig.update_yaxes(title_text=feature2title[plot_features[i]], row=i+1, col=1)

        if plot_features[i] in ['Be', 'TOCME']:
            fig.update_yaxes(title_text=feature2title[plot_features[i]], range=[0, 1], row=i + 1, col=1)
        elif plot_features[i] == 'PA':
            fig.update_yaxes(title_text=feature2title[plot_features[i]], range=[0, 180], row=i + 1, col=1)
        else:
            fig.update_yaxes(title_text=feature2title[plot_features[i]], row=i + 1, col=1)


    # save fig
    if not os.path.exists(figpath):
        os.makedirs(figpath)
    py.plot(fig,
            filename=figpath+'/'+args['time'][0].strftime('%Y%m%d%H%M')+'_'+args['time'][-1].strftime('%Y%m%d%H%M')+'.html',
            image='svg',
            )


def plot_swics(args,icmes,ys=None,eval=None,
                 plot_features=plot_features_swics,
                 figpath=figpath_swics):
    feature2title = {'Vp':'Vp Km/s','O76':'O7/O6'}

    panelnum = len(plot_features)+1
    eventnum = len(icmes)
    # fig,axes = plt.subplots(nrows=panelnum,
    #                         ncols=1,
    #                         figsize=(10,3*panelnum),
    #                         gridspec_kw = {'wspace':0, 'hspace':0})
    #
    # if eval is not None:
    #     axes[0].set_title('SWICS P={:.2f} R={:.2f}'.format(eval['precision'], eval['recall']), fontsize=16)
    #     for i in range(eventnum):
    #         if i == 0:
    #             line1, = axes[-1].plot([icmes[i][0], icmes[i][1]], [3 / 4, 3 / 4], 'r-', linewidth=3)
    #         else:
    #             axes[-1].plot([icmes[i][0], icmes[i][1]], [3 / 4, 3 / 4], 'r-', linewidth=3)
    #     for i in range(len(ys)):
    #         if i ==0:
    #             line2, = axes[-1].plot([ys[i][0], ys[i][1]], [2 / 4, 2 / 4], 'b-', linewidth=3)
    #         else:
    #             axes[-1].plot([ys[i][0], ys[i][1]], [2 / 4, 2 / 4], 'b-', linewidth=3)
    #     for i in range(len(eval['overlap'])):
    #         if i == 0:
    #             line3, = axes[-1].plot([eval['overlap'][i][0], eval['overlap'][i][1]], [1 / 4, 1 / 4], 'g-', linewidth=3)
    #         else:
    #             axes[-1].plot([eval['overlap'][i][0], eval['overlap'][i][1]], [1 / 4, 1 / 4], 'g-', linewidth=3)
    #     axes[-1].legend(handles=[line1, line2, line3], labels=['SWICS', 'list', 'overlap'])
    #
    # else:
    #     axes[0].set_title('SWICS', fontsize=16)
    #     for i in range(eventnum):
    #         axes[-1].plot([icmes[i][0], icmes[i][1]], [2 / 4, 2 / 4], 'r-', linewidth=3)
    # # set ylabel
    # axes[-1].set_ylabel('ICME', fontsize=16)
    # # set xlabel
    # axes[-1].set_xlabel('Time', fontsize=16)
    # # set xlim
    # axes[-1].set_xlim(args['time'][0], args['time'][-1])
    # # set ylim
    # axes[-1].set_ylim([0, 1])
    # # close y ticks
    # axes[-1].set_yticklabels([])
    #
    # for i in range(len(plot_features)):
    #     axes[i].plot(args['time'], args[plot_features[i]], 'b-', linewidth=1)
    #     axes[i].set_ylabel(feature2title[plot_features[i]], fontsize=16)
    #     axes[i].set_xlim(args['time'][0], args['time'][-1])
    #     axes[i].set_xticklabels([])
    #
    # # save figure
    # if not os.path.exists(figpath):
    #     os.makedirs(figpath)
    # plt.savefig(figpath+'/'+args['time'][0].strftime('%Y%m%d%H%M')+'_'+args['time'][-1].strftime('%Y%m%d%H%M')+'.png')

    fig = go.Figure().set_subplots(rows=panelnum,
                                   cols=1,
                                   shared_xaxes=True,

                                   )
    if eval is not None:
        fig.update_layout(title_text='SWICS P={:.2f} R={:.2f}'.
                          format(eval['precision'], eval['recall']),
                          )
        for i in range(eventnum):
            if i == 0:
                fig.add_trace(
                    go.Scatter(
                        x=[icmes[i][0], icmes[i][1]], y=[3 / 4, 3 / 4],
                        mode='lines',
                        line=dict(color='red', width=10),
                        name='ICME',
                    ),
                    row=panelnum,
                    col=1,
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=[icmes[i][0], icmes[i][1]],
                        y=[3 / 4, 3 / 4],
                        mode='lines',
                        line=dict(color='red', width=10),
                        name='ICME',
                        showlegend=False,
                    ),
                    row=panelnum,
                    col=1,
                )
        for i in range(len(ys)):
            if i ==0:
                fig.add_trace(
                    go.Scatter(
                        x=[ys[i][0], ys[i][1]],
                        y=[2 / 4, 2 / 4],
                        mode='lines',
                        line=dict(color='blue', width=10),
                        name='list',
                    ),
                    row=panelnum,
                    col=1,
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=[ys[i][0], ys[i][1]],
                        y=[2 / 4, 2 / 4],
                        mode='lines',
                        line=dict(color='blue', width=10),
                        name='list',
                        showlegend=False,
                    ),
                    row=panelnum,
                    col=1,
                )
        for i in range(len(eval['overlap'])):
            if i == 0:
                fig.add_trace(
                    go.Scatter(
                        x=[eval['overlap'][i][0],
                           eval['overlap'][i][1]],
                        y=[1 / 4, 1 / 4],
                        mode='lines',
                        line=dict(color='green', width=10),
                        name='overlap',
                    ),
                    row=panelnum,
                    col=1,
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=[eval['overlap'][i][0], eval['overlap'][i][1]],
                        y=[1 / 4, 1 / 4],
                        mode='lines',
                        line=dict(color='green', width=10),
                        name='overlap',
                        showlegend=False,
                    ),
                    row=panelnum,
                    col=1,
                )
    else:
        fig.update_layout(title_text='SWICS',
                          )
        for i in range(eventnum):
            if i==0:
                fig.add_trace(
                    go.Scatter(
                        x=[icmes[i][0], icmes[i][1]],
                        y=[2 / 4, 2 / 4],
                        mode='lines',
                        line=dict(color='red', width=10),
                        name='ICME',
                    ),
                    row=panelnum,
                    col=1,
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=[icmes[i][0], icmes[i][1]],
                        y=[2 / 4, 2 / 4],
                        mode='lines',
                        line=dict(color='red', width=10),
                        name='ICME',
                        showlegend=False,
                    ),
                    row=panelnum,
                    col=1,
                )
    fig.update_yaxes(title_text='ICME', range=[0, 1], row=panelnum, col=1)
    for i in range(len(plot_features)):
        fig.add_trace(
            go.Scatter(x=args['time'],
                       y=args[plot_features[i]],
                       mode='lines',
                       line=dict(color='black', width=1),
                       name=plot_features[i],
                       showlegend=False,
                       ),
            row=i + 1,
            col=1,
        )
        fig.update_yaxes(title_text=feature2title[plot_features[i]],
                         row=i + 1, col=1)

    fig.update_layout(legend=dict(
        yanchor="bottom",
        y=0.01,
        xanchor="left",
        x=0.01
    ),
    )
    # save fig
    if not os.path.exists(figpath):
        os.makedirs(figpath)
    py.plot(fig,
            filename=figpath+'/'+args['time'][0].strftime('%Y%m%d%H%M')+'_'+args['time'][-1].strftime('%Y%m%d%H%M')+'.html',
            image='svg',
            )


def plot_xb(args,icmes,ys=None,eval=None,
                 plot_features=plot_features_xb,
                 figpath=figpath_xb):
    feature2title = {'Vp':'Vp Km/s','Np':'Np cm<sup>-3</sup>','Tp':'Tp 10<sup>4</sup>K','Mag':'B nT'}

    panelnum = len(plot_features)+1
    eventnum = len(icmes)

    fig = go.Figure().set_subplots(rows=panelnum,
                                   cols=1,
                                   shared_xaxes=True,

                                   )
    if eval is not None:
        fig.update_layout(title_text='XB P={:.2f} R={:.2f}'.
                          format(eval['precision'], eval['recall']),
                          )
        for i in range(eventnum):
            if i == 0:
                fig.add_trace(
                    go.Scatter(
                        x=[icmes[i][0], icmes[i][1]], y=[3 / 4, 3 / 4],
                        mode='lines',
                        line=dict(color='red', width=10),
                        name='ICME',
                    ),
                    row=panelnum,
                    col=1,
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=[icmes[i][0], icmes[i][1]],
                        y=[3 / 4, 3 / 4],
                        mode='lines',
                        line=dict(color='red', width=10),
                        name='ICME',
                        showlegend=False,
                    ),
                    row=panelnum,
                    col=1,
                )
        for i in range(len(ys)):
            if i ==0:
                fig.add_trace(
                    go.Scatter(
                        x=[ys[i][0], ys[i][1]],
                        y=[2 / 4, 2 / 4],
                        mode='lines',
                        line=dict(color='blue', width=10),
                        name='list',
                    ),
                    row=panelnum,
                    col=1,
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=[ys[i][0], ys[i][1]],
                        y=[2 / 4, 2 / 4],
                        mode='lines',
                        line=dict(color='blue', width=10),
                        name='list',
                        showlegend=False,
                    ),
                    row=panelnum,
                    col=1,
                )
        for i in range(len(eval['overlap'])):
            if i == 0:
                fig.add_trace(
                    go.Scatter(
                        x=[eval['overlap'][i][0],
                           eval['overlap'][i][1]],
                        y=[1 / 4, 1 / 4],
                        mode='lines',
                        line=dict(color='green', width=10),
                        name='overlap',
                    ),
                    row=panelnum,
                    col=1,
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=[eval['overlap'][i][0], eval['overlap'][i][1]],
                        y=[1 / 4, 1 / 4],
                        mode='lines',
                        line=dict(color='green', width=10),
                        name='overlap',
                        showlegend=False,
                    ),
                    row=panelnum,
                    col=1,
                )
    else:
        fig.update_layout(title_text='XB',
                          )
        for i in range(eventnum):
            if i == 0:
                fig.add_trace(
                    go.Scatter(
                        x=[icmes[i][0], icmes[i][1]],
                        y=[2 / 4, 2 / 4],
                        mode='lines',
                        line=dict(color='red', width=10),
                        name='ICME',
                    ),
                    row=panelnum,
                    col=1,
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=[icmes[i][0], icmes[i][1]],
                        y=[2 / 4, 2 / 4],
                        mode='lines',
                        line=dict(color='red', width=10),
                        name='ICME',
                        showlegend=False,
                    ),
                    row=panelnum,
                    col=1,
                )
    fig.update_yaxes(title_text='ICME', range=[0, 1], row=panelnum, col=1)

    for i in range(len(plot_features)):
        if plot_features[i] == 'Tp':
            fig.add_trace(
                go.Scatter(
                    x=args['time'],
                    y=args[plot_features[i]]/10000,
                    mode='lines',
                    line=dict(color='black', width=1),
                    name=plot_features[i],
                    showlegend=False,
                ),
                row=i+1,
                col=1)
            fig.update_yaxes(title_text=feature2title[plot_features[i]], row=i+1, col=1)
        else:
            fig.add_trace(
                go.Scatter(x=args['time'],
                           y=args[plot_features[i]],
                           mode='lines',
                           line=dict(color='black', width=1),
                           name=plot_features[i],
                           showlegend=False,
                           ),
                row=i+1,
                col=1,
            )
            fig.update_yaxes(title_text=feature2title[plot_features[i]],
                             row=i+1, col=1)
    fig.update_layout(legend=dict(
        yanchor="bottom",
        y=0.01,
        xanchor="left",
        x=0.01
    ),
    )
    # save fig
    if not os.path.exists(figpath):
        os.makedirs(figpath)
    py.plot(fig,
            filename=figpath+'/'+args['time'][0].strftime('%Y%m%d%H%M')+'_'+args['time'][-1].strftime('%Y%m%d%H%M')+'.html',
            image='svg',
            )



    # fig,axes = plt.subplots(nrows=panelnum,
    #                         ncols=1,
    #                         figsize=(10,1.5*panelnum),
    #                         gridspec_kw = {'wspace':0, 'hspace':0})
    #
    # if eval is not None:
    #     axes[0].set_title('XB P={:.2f} R={:.2f}'.format(eval['precision'], eval['recall']), fontsize=16)
    #     for i in range(eventnum):
    #         if i == 0:
    #             line1, = axes[-1].plot([icmes[i][0], icmes[i][1]], [3 / 4, 3 / 4], 'r-', linewidth=3)
    #         else:
    #             axes[-1].plot([icmes[i][0], icmes[i][1]], [3 / 4, 3 / 4], 'r-', linewidth=3)
    #     for i in range(len(ys)):
    #         if i ==0:
    #             line2, = axes[-1].plot([ys[i][0], ys[i][1]], [2 / 4, 2 / 4], 'b-', linewidth=3)
    #         else:
    #             axes[-1].plot([ys[i][0], ys[i][1]], [2 / 4, 2 / 4], 'b-', linewidth=3)
    #     for i in range(len(eval['overlap'])):
    #         if i == 0:
    #             line3, = axes[-1].plot([eval['overlap'][i][0], eval['overlap'][i][1]], [1 / 4, 1 / 4], 'g-', linewidth=3)
    #         else:
    #             axes[-1].plot([eval['overlap'][i][0], eval['overlap'][i][1]], [1 / 4, 1 / 4], 'g-', linewidth=3)
    #     axes[-1].legend(handles=[line1, line2, line3], labels=['XB', 'list', 'overlap'])
    #
    # else:
    #     axes[0].set_title('XB', fontsize=16)
    #     for i in range(eventnum):
    #         axes[-1].plot([icmes[i][0], icmes[i][1]], [2 / 4, 2 / 4], 'r-', linewidth=3)
    # # set ylabel
    # axes[-1].set_ylabel('ICME', fontsize=16)
    # # set xlabel
    # axes[-1].set_xlabel('Time', fontsize=16)
    # # set xlim
    # axes[-1].set_xlim(args['time'][0], args['time'][-1])
    # # set ylim
    # axes[-1].set_ylim([0, 1])
    # # close y ticks
    # axes[-1].set_yticklabels([])
    #
    # for i in range(len(plot_features)):
    #     if plot_features[i] == 'Tp':
    #         axes[i].plot(args['time'], args['Tp']*1e-4, 'b-', linewidth=1)
    #         axes[i].set_ylabel(feature2title[plot_features[i]], fontsize=16)
    #         axes[i].set_xlim(args['time'][0], args['time'][-1])
    #         axes[i].set_xticklabels([])
    #     else:
    #         axes[i].plot(args['time'], args[plot_features[i]], 'b-', linewidth=1)
    #         axes[i].set_ylabel(feature2title[plot_features[i]], fontsize=16)
    #         axes[i].set_xlim(args['time'][0], args['time'][-1])
    #         axes[i].set_xticklabels([])
    #
    # # save figure
    # if not os.path.exists(figpath):
    #     os.makedirs(figpath)
    # plt.savefig(figpath+'/'+args['time'][0].strftime('%Y%m%d%H%M')+'_'+args['time'][-1].strftime('%Y%m%d%H%M')+'.png')

def plot_nn(args,icmes,ys=None,eval=None,
                 plot_features=plot_features_nn,
                 figpath='image/eval/NN/v7/test'):
    feature2title = {'Vp': 'Vp Km/s',
                     'Np': 'Np cm$^{-3}$',
                     'Tp': 'Tp 10$^4$K',
                     'Mag': 'B nT',
                     'lambda':'λ °',
                     'delta':'δ °',
                     'PA':'PA(#) °(dB)',
                     'dbrms':'dbrms nT'}

    panelnum = len(plot_features)+1
    eventnum = len(icmes)
    fig,axes = plt.subplots(nrows=panelnum,
                            ncols=1,
                            figsize=(9,2*panelnum),
                            gridspec_kw = {'wspace':0, 'hspace':0})

    if eval is not None:
        axes[0].set_title('NN P={:.2f} R={:.2f}'.format(eval['precision'], eval['recall']), fontsize=16)
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
        axes[-1].legend(handles=[line1, line2, line3], labels=['NN', 'list', 'overlap'])

    else:
        axes[0].set_title('NN', fontsize=16)
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
        if plot_features[i] == 'PA':
            anglenum = np.shape(args['PA'])[0]
            deltaangle = 180 / anglenum
            angles = np.arange(0, 180, deltaangle)+deltaangle/2
            X,Y = np.meshgrid(args['time'],angles)
            axes[i].contourf(X,Y,10*np.log10(args['PA']),cmap='jet')
            axes[i].set_ylabel('PA(#) °(dB)', fontsize=16)
            axes[i].set_xticklabels([])
            axes[i].set_xlim(args['time'][0], args['time'][-1])
            axes[i].set_ylim([0, 180])
        elif plot_features[i] == 'Tp':
            axes[i].plot(args['time'], args['Tp'] * 1e-4, 'b-', linewidth=1)
            axes[i].set_ylabel(feature2title['Tp'], fontsize=16)
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
def swics_io(args,test=False,ifplot=False,plot_features=plot_features_swics,figpath=figpath_swics):
    '''

    :param args: 应该包含如下数据，质子速度与O76的时间分辨率应该是一样的（建议1hr或2hr），即应该在预处理时插值到所需的时间点, 这些一维的数据格式是np array，shape=(n,)注意不要把数据搞成向量输入了
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
    args['icme'] = icme
    args['icmes'] = icmes
    if icmes is None:
        print('Final: No ICME detected!')
        return None
    if test:
        assert 'y' in args, "没有检测到输入权威icme标记y，请关闭测试模式，或检查输入数据"
        ys = evaluate.checkIcme(args['y'],args)
        eval_swics = evaluate.evaluateIcme(icmes, ys)
        if eval_swics['recall'] == 0:
            print('Final: No ICME detected!')
            return None
        elif ifplot:
            plot_swics(args,icmes,ys=ys,eval=eval_swics,plot_features=plot_features,figpath=figpath)
    else:
        if ifplot:
            plot_swics(args,icmes,plot_features=plot_features,figpath=figpath)


################ XB ################
def xb_io(args,test=False,ifplot=False,plot_features=plot_features_xb,figpath=figpath_xb):
    '''

    :param args: 应该包含如下数据，时间分辨率应该是一样的（建议1hr或2hr），即应该在预处理时插值到所需的时间点, 这些一维的数据格式是np array，shape=(n,)注意不要把数据搞成向量输入了
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
    args['icme'] = icme
    args['icmes'] = icmes
    if icmes is None:
        print('Final: No ICME detected!')
        return None
    if test:
        assert 'y' in args, "没有检测到输入权威icme标记y，请关闭测试模式，或检查输入数据"
        ys = evaluate.checkIcme(args['y'],args)
        eval_xb = evaluate.evaluateIcme(icmes, ys)
        if eval_xb['recall'] == 0:
            print('Final: No ICME detected!')
            return None
        elif ifplot:
            plot_xb(args,icmes,ys=ys,eval=eval_xb,plot_features=plot_features,figpath=figpath)
    else:
        if ifplot:
            plot_xb(args,icmes,plot_features=plot_features,figpath=figpath)











################ Genesis ################
def genesis_io(args,
               test=False,
               constants=constants_genesis,
               Wa=1,
               Wb=1,
               ifplot=False,
               plot_features=plot_features_genesis,
               figpath=figpath_genesis):
    '''
    ——必须的数据7个：
    Vp: 质子速度 Km/s, shape=#timepoints
    Vp_time: datetime数据类型,numpy的array
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
    assert (args['Tp_time'] == args['Vp_time']).all, "Tp_time和Vp_time应该是相同的，如果这两个数据时间不同，还应该再插值到Vp_time"
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
    args_avg['icme'] = icme
    args_avg['icmes'] = icmes
    if icmes is None:
        print('Final: No ICME detected!')
        return None
    if test:
        assert 'y' in args, "没有检测到输入权威icme标记y，请关闭测试模式，或检查输入数据"
        ys = evaluate.checkIcme(args['y'],args_avg)
        eval_genesis = evaluate.evaluateIcme(icmes, ys)
        if eval_genesis['recall'] == 0:
            print('Final: No ICME detected!')
            return None
        elif ifplot:
            plot_genesis(args,args_avg,icmes,ys=ys,eval=eval_genesis,plot_features=plot_features,figpath=figpath)
    else:
        if ifplot:
            plot_genesis(args,args_avg,icmes,plot_features=plot_features,figpath=figpath)

    return args_avg

############### Machine Learning ################
def nn_io(args,model,test=False,ifplot=False,plot_features=plot_features_genesis):
    ### 检查数据是否完整
    # if 'Vp' not in args or 'Vp_time' not in args or 'Np' not in args or 'Np_time' not in args or 'Tp' not in args or 'Tp_time' not in args or 'time' not in args:
    #     raise ValueError('缺少必要的数据，请检查数据是否完整，或字典key的名称是否正确("Vp","Vp_time","Np","Np_time","Tp","Tp_time","time")')
    ### 给出time每个时间点的icme判定
    eventstep = len(args['Vp'])
    xdata = np.hstack((args['Vp'].reshape(eventstep,1),args['Np'].reshape(eventstep,1),
                       args['Tp'].reshape(eventstep,1),args['delta'].reshape(eventstep,1),
                       args['lambda'].reshape(eventstep,1),args['Mag'].reshape(eventstep,1),
                       args['dbrms'].reshape(eventstep,1),args['PA'].T))
    means = np.nanmean(xdata,axis=0)
    maxmins = np.nanmax(xdata,axis=0)-np.nanmin(xdata,axis=0)
    xdata = (xdata-means)/maxmins
    icme = model.predict(xdata,).reshape(eventstep,)>0.5
    icmes = evaluate.checkIcme(icme,args)
    args['icme'] = icme
    args['icmes'] = icmes
    if icmes is None:
        print('Final: No ICME detected!')
        return None
    if test:
        assert 'y' in args, "没有检测到输入权威icme标记y，请关闭测试模式，或检查输入数据"
        ys = evaluate.checkIcme(args['y'],args)
        eval_nn = evaluate.evaluateIcme(icmes, ys)
        if eval_nn['recall'] == 0:
            print('Final: No ICME detected!')
            return None
        elif ifplot:
            plot_nn(args,icmes,ys=ys,eval=eval_nn,plot_features=plot_features)
    else:
        if ifplot:
            plot_nn(args,icmes,plot_features=plot_features)





if __name__ == '__main__':
    ### genesis
    def test_genesis(eventIdx = 42,
                     fileName = 'data/eval/Genesis/origin_data.mat',
                     list_features = list_features,
                     plot_features = plot_features_genesis,
                     figpath=figpath_genesis):

        # event
        # eventSteps, eventSteps_swe, eventSteps_pa, swedata, padata, magdata, ydata, event_epoch, event_epoch_swe, event_epoch_pa = preprocessing.load_original_data_genesis(fileName=fileName) # for event
        # eventSteps, eventSteps_swe, swedata, ydata, event_epoch, event_epoch_swe = preprocessing.load_original_data_genesis(fileName=fileName)
        # eventSWE = swedata[:, :eventSteps_swe[0, eventIdx], eventIdx]  # Np;Tp;Vp;He4top
        # eventsweT = event_epoch_swe[:eventSteps_swe[0, eventIdx], eventIdx]
        # eventsweT = (eventsweT - 719529.0) * 86400.0 - 8.0 * 3600.0
        # eventsweT = [datetime.datetime.fromtimestamp(t) for t in eventsweT]
        #
        # eventT = event_epoch[:eventSteps[0, eventIdx], eventIdx]
        # eventT = (eventT - 719529.0) * 86400.0 - 8.0 * 3600.0
        # eventT = np.array([datetime.datetime.fromtimestamp(t) for t in eventT])
        #
        # args = {
        #     'Np': eventSWE[0, :], 'Np_time': eventsweT,
        #     'Tp': eventSWE[1, :], 'Tp_time': eventsweT,
        #     'Vp': eventSWE[2, :], 'Vp_time': eventsweT,
        #     'time': eventT,
        # }
        # if ydata is not None:
        #     args['y'] = ydata[:eventSteps[0, eventIdx], eventIdx]
        # if padata is not None:
        #     eventPA = padata[:, :eventSteps_pa[0, eventIdx], eventIdx]
        #     eventpaT = event_epoch_pa[:eventSteps_pa[0, eventIdx], eventIdx]
        #     eventpaT = (eventpaT - 719529.0) * 86400.0 - 8.0 * 3600.0
        #     eventpaT = [datetime.datetime.fromtimestamp(t) for t in eventpaT]
        #     args['PA'] = eventPA
        #     args['PA_time'] = eventpaT
        # if eventSWE.shape[0] == 4:
        #     args['NHetoNp'] = eventSWE[3, :]
        #     args['NHetoNp_time'] = eventsweT

        # tot
        # swedata, padata, magdata, ydata, event_epoch, event_epoch_swe, event_epoch_pa = preprocessing.load_original_data_genesis(
        #     fileName=fileName)  # for tot
        # swet = (event_epoch_swe - 719529.0) * 86400.0 - 8.0 * 3600.0
        # swet = np.array([datetime.datetime.fromtimestamp(t) for t in swet[0,:]])
        # yt = (event_epoch - 719529.0) * 86400.0 - 8.0 * 3600.0
        # yt = np.array([datetime.datetime.fromtimestamp(t) for t in yt[0,:]])

        ########### dscovr from loadData ###########
        swedata, magdata, yt, swet = loadData.outputdata_genesis(filepath=fileName)
        swedata = swedata.T
        ydata = None
        padata = None
        event_epoch_pa = None

        args = {
            'Np': swedata[0, :], 'Np_time': swet,
            'Tp': swedata[1, :], 'Tp_time': swet,
            'Vp': swedata[2, :], 'Vp_time': swet,
            'time': yt,
        }
        if ydata is not None:
            args['y'] = ydata[0,:]
        if padata is not None:
            pat = (event_epoch_pa - 719529.0) * 86400.0 - 8.0 * 3600.0
            pat = np.array([datetime.datetime.fromtimestamp(t) for t in pat[0, :]])
            args['PA'] = padata
            args['PA_time'] = pat
        if swedata.shape[0] == 4:
            args['NHetoNp'] = swedata[3, :]
            args['NHetoNp_time'] = swet
        if magdata is not None:
            # args['Mag'] = magdata[0, :]
            args['Mag'] = magdata
            args['Mag_time'] = yt
        args_avg = genesis_io(args, test=False, Wa=1,Wb=1,ifplot=1,plot_features=plot_features,figpath=figpath)
        if 'Mag' in args.keys():
            args_avg['Mag'] = args['Mag']
        if args_avg is not None:
            cmelist = listIcmes(args_avg, list_features=list_features,savejson=True,filename='genesis_icme.json')

        print('genesis test done!')

    ### SWICS
    def test_swics(eventIdx = 42,
                   fileName = 'data/eval/SWICS/data.mat',
                   list_features = list_features,
                   plot_features=plot_features_swics,
                   figpath=figpath_swics):
        # xdata, ydata, eventTimes, eventSteps = evaluate.loaddata_swics(fileName=fileName) # for event

        # eventTime = eventTimes[:eventSteps[0, eventIdx], eventIdx]
        # eventTime = (eventTime - 719529.0) * 86400.0 - 8.0 * 3600.0
        # eventTime = np.array([datetime.datetime.fromtimestamp(t) for t in eventTime])
        # args = {
        #     'time': eventTime,
        #     'Vp': xdata[1, :eventSteps[0, eventIdx], eventIdx],
        #     'O76': xdata[0, :eventSteps[0, eventIdx], eventIdx],
        #     'y': ydata[:eventSteps[0, eventIdx], eventIdx],
        # }

        xdata, ydata, eventTime = evaluate.loaddata_swics(fileName=fileName)  # for tot
        eventTime = (eventTime - 719529.0) * 86400.0 - 8.0 * 3600.0
        eventTime = np.array([datetime.datetime.fromtimestamp(t) for t in eventTime[:,0]])
        args = {
            'time': eventTime,
            'Vp': xdata[:, 1],
            'O76': xdata[:, 0],
            'y': ydata[:,0],
        }

        swics_io(args, test=False, ifplot=True, plot_features=plot_features,figpath=figpath)
        if args['icmes'] is not None:
            icmelist = listIcmes(args, list_features=list_features,savejson=True,filename='swics_icmes.json')
        print('swics test done!')

    ### xb
    def test_xb(eventIdx = 42,
                fileName = 'data/eval/XB/data.mat',
                list_features = list_features,
                figpath=figpath_xb):
        # xdata, ydata, eventTimes, eventSteps = evaluate.loaddata_xb(fileName=fileName) # for event
        # eventTime = eventTimes[:eventSteps[0, eventIdx], eventIdx]
        # eventTime = (eventTime - 719529.0) * 86400.0 - 8.0 * 3600.0
        # eventTime = np.array([datetime.datetime.fromtimestamp(t) for t in eventTime])
        # args = {
        #     'time': eventTime,
        #     'Vp': xdata[2, :eventSteps[0, eventIdx], eventIdx],
        #     'Tp': xdata[1, :eventSteps[0, eventIdx], eventIdx],
        #     'Np': xdata[0, :eventSteps[0, eventIdx], eventIdx],
        #     'Mag': xdata[3, :eventSteps[0, eventIdx], eventIdx],
        #     'y': ydata[:eventSteps[0, eventIdx], eventIdx],
        # }
        xdata, ydata, eventTime = evaluate.loaddata_xb(fileName=fileName)  # for tot
        eventTime = (eventTime - 719529.0) * 86400.0 - 8.0 * 3600.0
        # eventTime = np.array([datetime.datetime.fromtimestamp(t) for t in eventTime[0,:]])
        eventTime = np.array([datetime.datetime.fromtimestamp(t) for t in eventTime[:, 0]])
        args = {
            'time': eventTime,
            # 'Vp': xdata[2, :],
            'Vp': xdata[:, 2],
            # 'Tp': xdata[1, :],
            'Tp': xdata[:, 1],
            # 'Np': xdata[0, :],
            'Np': xdata[:, 0],
            # 'Mag': xdata[3, :],
            'Mag': xdata[:, 3],
            # 'y': ydata[:,0],
        }
        if ydata is not None:
            args['y'] = ydata[:,0]
            xb_io(args, test=True, ifplot=True, plot_features=plot_features_xb,figpath=figpath)
        else:
            xb_io(args, test=False, ifplot=True, plot_features=plot_features_xb,figpath=figpath)
        if args['icmes'] is not None:
            icmelist = listIcmes(args, list_features=list_features,savejson=True,filename='xb_icmes.json')
        print('xb test done!')

    #### NN
    def test_nn(eventIdx = 42):
        import tensorflow
        file = h5py.File('data/eval/ML/v7/data.mat')  # "eventSteps","eventTimes","xdata","ydata","means","stds"
        xdata = np.array(file['xdata'])
        means = np.mean(xdata, axis=0)
        maxmins = np.max(xdata, axis=0) - np.min(xdata, axis=0)
        ydata = np.array(file['ydata'])
        eventTimes = file['eventEpochs']
        eventSteps = np.array(file['eventSteps'])
        print(eventIdx)
        eventTime = eventTimes[:eventSteps[0, eventIdx], eventIdx]
        eventTime = (eventTime - 719529.0) * 86400.0 - 8.0 * 3600.0
        eventTime = np.array([datetime.datetime.fromtimestamp(t) for t in eventTime])
        args = {
            'time': eventTime,
            'Vp': xdata[0, :eventSteps[0, eventIdx], eventIdx],
            'Tp': xdata[2, :eventSteps[0, eventIdx], eventIdx],
            'Np': xdata[1, :eventSteps[0, eventIdx], eventIdx],
            'delta': xdata[3, :eventSteps[0, eventIdx], eventIdx],
            'lambda': xdata[4, :eventSteps[0, eventIdx], eventIdx],
            'Mag': xdata[5, :eventSteps[0, eventIdx], eventIdx],
            'dbrms': xdata[6, :eventSteps[0, eventIdx], eventIdx],
            'PA': xdata[7:, :eventSteps[0, eventIdx], eventIdx],
            'y': ydata[:eventSteps[0, eventIdx], eventIdx],
        }
        if len(eventTime) == 0:
            print('eventTime is None')
            return None
        model = tensorflow.keras.models.load_model('model/v7/model_v7_1_1.h5')
        nn_io(args,model, test=True, ifplot=True, plot_features=plot_features_nn)
        print('nn test done!')


    # test_genesis(eventIdx=11,fileName='data/eval/Genesis/datatot_2002.mat',list_features=list_features,plot_features=plot_features_genesis,figpath=figpath_genesis)
    test_genesis(eventIdx=11, fileName='data/origin/DSCOVR/data/2022/01', list_features=list_features,
                 plot_features=plot_features_genesis, figpath=figpath_genesis)
    # test_xb(eventIdx=11,fileName='data/eval/XB/datatot.mat',list_features=list_features)
    # test_swics(eventIdx=200,fileName='data/eval/SWICS/datatot2.mat',list_features=list_features,plot_features=plot_features_swics,figpath=figpath_swics)
