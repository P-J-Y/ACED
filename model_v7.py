#高精度，以cme起始信号作为触发的触发检测机
#可采用只输入磁场的方式
import kerastuner.tuners
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
import h5py
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv1D
from tensorflow.keras.layers import GRU, Bidirectional, BatchNormalization, Reshape,Embedding,Masking
from tensorflow.keras.optimizers import Adam
import getdataset
import matplotlib.pyplot as plt
import tensorflow.keras.regularizers as tfkreg
import aced_utils
from kerastuner import HyperModel
from kerastuner.tuners import RandomSearch


class MyHyperModel(HyperModel):

    def __init__(self,input_shape):
        self.input_shape = input_shape

    def build(self,hp):
        lr = hp.Float("lr",min_value=1e-5,max_value=1e-1,sampling="log")
        lambda_l2 = hp.Float("lambda_l2",min_value=1e-5,max_value=1e-1,sampling="log")
        X_input = Input(shape=self.input_shape)
        X = Dense(units=hp.Int('units0',min_value=8,max_value=32,step=4), activation='relu', kernel_regularizer=tfkreg.l2(lambda_l2))(X_input)
        X = BatchNormalization()(X)
        X = Dense(units=hp.Int('units1',min_value=16,max_value=64,step=4), activation='relu', kernel_regularizer=tfkreg.l2(lambda_l2))(X)
        X = BatchNormalization()(X)
        X = Dense(units=hp.Int('units2',min_value=8,max_value=32,step=4), activation='relu', kernel_regularizer=tfkreg.l2(lambda_l2))(X)
        X = BatchNormalization()(X)
        X = Dense(units=hp.Int('units3',min_value=4,max_value=32,step=4), activation='relu', kernel_regularizer=tfkreg.l2(lambda_l2))(X)
        X = BatchNormalization()(X)
        X = Dense(1, activation='sigmoid', kernel_regularizer=tfkreg.l2(lambda_l2))(X)

        model = Model(inputs=X_input, outputs=X)
        opt = Adam(learning_rate=lr)
        model.compile(optimizer=opt,loss='binary_crossentropy',metrics=["accuracy"])
        return model


fileName = 'data/train_v7_1.mat'
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

hypermodel = MyHyperModel(input_shape=xtrain.shape[1:])

class MyTuner(kerastuner.tuners.BayesianOptimization):
    def run_trial(self, trial, *args, **kwargs):
        kwargs['batch_size'] = trial.hyperparameters.Int('batch_size',32,256,step=32)
        super(MyTuner,self).run_trial(trial,*args,**kwargs)

tuner = MyTuner(
    hypermodel,
    objective='val_loss',
    max_trials=10,
    directory='my_dir',
    project_name='model_v7_2',
)
tuner.search(xtrain,ytrain,
             epochs=20,
             validation_data=(xdev,ydev),
             )
tuner.results_summary()
#tf.compat.v1.disable_v2_behavior() # model trained in tf1
#model = tf.compat.v1.keras.models.load_model('./model/v1/my_model.h5')
