#高精度，以cme起始信号作为触发的触发检测机
#可采用只输入磁场的方式
import kerastuner.tuners
import numpy as np
import tensorflow
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
import h5py
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv1D
from tensorflow.keras.layers import GRU, Bidirectional, BatchNormalization, Reshape,Embedding,Masking
from tensorflow.keras.optimizers import Adam

import V1_utils
import getdataset
import matplotlib.pyplot as plt
import tensorflow.keras.regularizers as tfkreg
import aced_utils
from kerastuner import HyperModel
from kerastuner.tuners import RandomSearch


# class MyHyperModel(HyperModel):
#
#     def __init__(self,input_shape):
#         self.input_shape = input_shape
#
#     def build(self,hp):
#         lr = hp.Float("lr",min_value=1e-5,max_value=1e-1,sampling="log")
#         lambda_l2 = hp.Float("lambda_l2",min_value=1e-5,max_value=1e-1,sampling="log")
#         X_input = Input(shape=self.input_shape)
#         X = Dense(units=hp.Int('units0',min_value=8,max_value=32,step=4), activation='relu', kernel_regularizer=tfkreg.l2(lambda_l2))(X_input)
#         X = BatchNormalization()(X)
#         X = Dense(units=hp.Int('units1',min_value=16,max_value=64,step=4), activation='relu', kernel_regularizer=tfkreg.l2(lambda_l2))(X)
#         X = BatchNormalization()(X)
#         X = Dense(units=hp.Int('units2',min_value=8,max_value=32,step=4), activation='relu', kernel_regularizer=tfkreg.l2(lambda_l2))(X)
#         X = BatchNormalization()(X)
#         X = Dense(units=hp.Int('units3',min_value=4,max_value=32,step=4), activation='relu', kernel_regularizer=tfkreg.l2(lambda_l2))(X)
#         X = BatchNormalization()(X)
#         X = Dense(1, activation='sigmoid', kernel_regularizer=tfkreg.l2(lambda_l2))(X)
#
#         model = Model(inputs=X_input, outputs=X)
#         opt = Adam(learning_rate=lr)
#         model.compile(optimizer=opt,loss='binary_crossentropy',metrics=["accuracy"])
#         return model
#
#
# fileName = 'data/train_v7_1.mat'
# file = h5py.File(fileName)  # "eventSteps","eventTimes","xdata","ydata","means","stds"
# xdata = np.array(file['xdata'])
# means = np.mean(xdata,axis=0)
# maxmins = np.max(xdata,axis=0)-np.min(xdata,axis=0)
# xdata = (xdata-means)/maxmins
# ydata = np.array(file['ydata'])
# eventTimes = file['times']
# eventSteps = np.array(file['eventSteps'])
# devnum = np.sum(eventSteps[0:40])
# testnum = np.sum(eventSteps[40:80])
# xdev = xdata[:devnum]
# ydev = ydata[:devnum]
# xtest = xdata[devnum:(devnum+testnum)]
# ytest = ydata[devnum:(devnum+testnum)]
# xtrain = xdata[(devnum+testnum):]
# ytrain = ydata[(devnum+testnum):]
#
# hypermodel = MyHyperModel(input_shape=xtrain.shape[1:])
#
# class MyTuner(kerastuner.tuners.BayesianOptimization):
#     def run_trial(self, trial, *args, **kwargs):
#         kwargs['batch_size'] = trial.hyperparameters.Int('batch_size',32,256,step=32)
#         super(MyTuner,self).run_trial(trial,*args,**kwargs)
#
# tuner = MyTuner(
#     hypermodel,
#     objective='val_loss',
#     max_trials=10,
#     directory='my_dir',
#     project_name='model_v7_2',
# )
# tuner.search(xtrain,ytrain,
#              epochs=20,
#              validation_data=(xdev,ydev),
#              )
# tuner.results_summary()
# #tf.compat.v1.disable_v2_behavior() # model trained in tf1
# #model = tf.compat.v1.keras.models.load_model('./model/v1/my_model.h5')









def preprocess(fileName = 'data/train_v7_1.mat'):
    file = h5py.File(fileName)  # "eventSteps","eventTimes","xdata","ydata","means","stds"
    xdata = np.array(file['xdata'])
    means = np.mean(xdata,axis=0)
    maxmins = np.max(xdata,axis=0)-np.min(xdata,axis=0)
    xdata = (xdata-means)/maxmins
    ydata = np.array(file['ydata'])
    eventTimes = file['times']
    eventSteps = np.array(file['eventSteps'])
    devnum = np.sum(eventSteps[0:40])
    testnum = np.sum(eventSteps[40:80])
    xdev = xdata[:devnum]
    ydev = ydata[:devnum]
    xtest = xdata[devnum:(devnum+testnum)]
    ytest = ydata[devnum:(devnum+testnum)]
    xtrain = xdata[(devnum+testnum):]
    ytrain = ydata[(devnum+testnum):]
    return xtrain,ytrain,xdev,ydev,xtest,ytest




def model_v7(input_shape,params): #params: batch_size,lr,lambda_l2
    X_input = Input(input_shape)
    X = Dense(8, activation='relu',
              kernel_regularizer=tfkreg.l2(params['lambda_l2']))(X_input)
    X = BatchNormalization()(X)
    X = Dense(16, activation='relu',
              kernel_regularizer=tfkreg.l2(params['lambda_l2']))(X)
    X = BatchNormalization()(X)
    X = Dense(8, activation='relu',
              kernel_regularizer=tfkreg.l2(params['lambda_l2']))(X)
    X = BatchNormalization()(X)
    X = Dense(4, activation='relu',
              kernel_regularizer=tfkreg.l2(params['lambda_l2']))(X)
    X = BatchNormalization()(X)
    X = Dense(1, activation='sigmoid', kernel_regularizer=tfkreg.l2(params['lambda_l2']))(X)

    model = Model(inputs=X_input, outputs=X)
    return model


if __name__=='__main__':
    classes = [0, 1]

    xtrain,ytrain,xdev,ydev,xtest,ytest = preprocess()
    ################################# hyperopt model #####################################
    from hyperopt import hp, STATUS_OK, Trials, fmin, tpe
    from tensorflow.keras.callbacks import EarlyStopping

    space = {
        'lr': hp.loguniform('lr', -10, -0),
        'lambda_l2': hp.loguniform('lambda_l2', -10, -0),
        'batch_size': hp.choice('batch_size', [32,])
    }

    f1 = 0
    workidx = 0
    print('work {}'.format(workidx))
    maxtrailnum = 50
    def trainAmodel(params):
        global xtrain, xdev, ytrain, ydev
        global f1, workidx
        print('Params testing: ', params)
        aModel = model_v7(xtrain.shape[1:], params)
        opt = tensorflow.keras.optimizers.Adam(lr=params['lr'])
        aModel.compile(loss="binary_crossentropy", metrics=['accuracy'], optimizer=opt)
        steps_per_epoch = (np.shape(xtrain)[0] + params['batch_size'] - 1) // params['batch_size']
        # metrics = V1_utils.Metrics(test_data=(X_test[::10], Y_test[::10]), train_data=(X_train[::100], Y_train[::100]))

        from sklearn.utils import class_weight
        import pandas as pd

        class_weight = class_weight.compute_class_weight(class_weight='balanced',
                                                         classes=classes,
                                                         y=ytrain[:, 0])
        cw = dict(enumerate(class_weight))
        # early_stopping = EarlyStopping(monitor='val_loss', patience=8, min_delta=0.8, mode='min')
        # history = aModel.fit_generator(generator=data_generator(xtrain, ytrain, params['batch_size']),
        #                                  steps_per_epoch=steps_per_epoch,
        #                                  epochs=30,
        #                                  verbose=0,
        #                                  validation_data=(xdev[::80], ydev[::80]),
        #                                  callbacks=[early_stopping],
        #                                  # callbacks=[metrics],
        #                                  class_weight=cw,
        #                                  )
        history = aModel.fit(xtrain, ytrain,
                             batch_size=params['batch_size'],
                             epochs=30,
                             verbose=0,
                             # validation_data=(xdev[::10000], ydev[::10000]),
                             class_weight=cw,
                             )
        # 评估模型
        batch_size_test = 32
        # preds = aModel.evaluate_generator(generator=data_generator(xdev, ydev, batch_size_test, cycle=False),verbose=0)
        preds = aModel.evaluate(xdev, ydev, batch_size=batch_size_test, verbose=0)
        print("误差值 = " + str(preds[0]))
        print("准确度 = " + str(preds[1]))
        # cvres = aModel.predict_generator(data_generator(xdev,None,4,cycle=False,givey=False), verbose=0)
        # cvf1s, cache = V1_utils.fmeasure(ydev, cvres)
        cvres = aModel.predict(xdev, batch_size=batch_size_test, verbose=0)
        cvf1s, cache = V1_utils.fmeasure(ydev, cvres)
        p, r = cache
        print("f1 = {}, precision = {}, recall = {}".format(cvf1s, p, r))
        if cvf1s > f1:
            f1 = cvf1s
            aModel.save('model/v7/model_v7_1_{}.h5'.format(workidx))
            print(f1)
            plt.figure()
            plt.plot(history.history['loss'], 'b', label='Training loss')
            # plt.plot(history.history['val_loss'], 'r', label='Validation val_loss')
            plt.title('Traing loss')
            plt.legend()
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.savefig('image/v7/v7_1/log/loss_v2_3_{}.jpg'.format(workidx))
        return {
            'loss': -cvf1s,
            'status': STATUS_OK
        }

    trials = Trials()
    best = fmin(trainAmodel, space, algo=tpe.suggest, max_evals=maxtrailnum, trials=trials)

    filename = 'model/v7/log_v7_{}.npz'.format(workidx)
    np.savez(filename, trials=trials, best=best)

    print('best')
    print(best)

    trialNum = len(trials.trials)
    l2s = np.zeros(trialNum)
    lrs = np.zeros(trialNum)
    losses = np.zeros(trialNum)
    bzs = np.zeros(trialNum)
    for trialidx in range(trialNum):
        thevals = trials.trials[trialidx]['misc']['vals'] #如果是从文件中读取，这一行改成trials[]，即不需要后面那个.trails,下面losses那一行同理
        l2s[trialidx] = thevals['lambda_l2'][0]
        lrs[trialidx] = thevals['lr'][0]
        bzs[trialidx] = (thevals['batch_size'][0] + 1)
        losses[trialidx] = -trials.trials[trialidx]['result']['loss']

    plt.figure()
    # plt.scatter(np.log(lrs), np.log(l2s), c=bzs, s=losses * 100, cmap=mpl.colors.ListedColormap(
    #     ["darkorange", "gold", "lawngreen", "lightseagreen"]
    # ))
    plt.scatter(np.log(lrs), np.log(l2s), c=losses, )
    plt.xlabel('ln[lr]')
    plt.ylabel('ln[λ]')
    plt.title('f1')
    cb = plt.colorbar()
    # cb.set_label('log2[BatchSize]', labelpad=-1)
    plt.savefig('image/v7/v7_1/hyparams_v7_{}.jpg'.format(workidx))
    print('done')
