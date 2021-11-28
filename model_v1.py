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

timeSteps = 600
features = 9
lambda_l2 = 0.008
lr = 0.0004

def model(input_shape):
    """
    用 Keras 创建模型的图 Function creating the model's graph in Keras.

    参数：
    input_shape -- 模型输入数据的维度（使用Keras约定）

    返回：
    model -- Keras 模型实例
    """

    X_input = Input(shape=input_shape)
    X = Masking(mask_value=0.)(X_input)
    # # 第一步：卷积层 (≈4 lines)
    # X = Conv1D(196, 15, strides=4)(X_input)  # CONV1D
    # X = BatchNormalization()(X)  # Batch normalization 批量标准化
    # X = Activation('relu')(X)  # ReLu activation ReLu 激活
    # X = Dropout(0.8)(X)  # dropout (use 0.8)

    # 第二步：第一个 GRU 层 (≈4 lines)
    X = GRU(units=64, return_sequences=True,kernel_regularizer=tfkreg.l2(lambda_l2),)(X)  # GRU (使用128个单元并返回序列)
    #X = Dropout(0.6)(X)  # dropout (use 0.8)
    X = BatchNormalization()(X)  # Batch normalization 批量标准化

    # 第三步: 第二个 GRU 层  (≈4 lines)
    X = GRU(units=64, return_sequences=True,kernel_regularizer=tfkreg.l2(lambda_l2),)(X)  # GRU (使用128个单元并返回序列)
    #X = Dropout(0.6)(X)  # dropout (use 0.8)
    X = BatchNormalization()(X)  # Batch normalization 批量标准化
    #X = Dropout(0.6)(X)  # dropout (use 0.8)

    # 第四步： 时间分布全连接层 (≈1 line)
    X = Dense(16,activation="relu",kernel_regularizer=tfkreg.l2(lambda_l2))(X)
    #X = Dropout(0.6)(X)
    X = BatchNormalization()(X)
    X = TimeDistributed(Dense(1, activation="sigmoid"))(X)  # time distributed  (sigmoid)

    model = Model(inputs=X_input, outputs=X)

    return model


fileName = 'data/train_v1_2.mat'
file = h5py.File(fileName)  # "eventSteps","eventTimes","xdata","ydata","means","stds"
xdata = file['xdata']
ydata = file['ydata']
xtrain, ytrain = getdataset.creat_train_data(xdata, ydata)

model = model(input_shape=(timeSteps, features))
#tf.compat.v1.disable_v2_behavior() # model trained in tf1
#model = tf.compat.v1.keras.models.load_model('./model/v1/my_model.h5')
model.summary()
opt = Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, decay=0.01)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])
history = model.fit(xtrain,ytrain,batch_size=16,epochs=40,verbose=1,validation_split=0.1)
model.save('./model/v1/my_model.h5')
plt.figure()
plt.plot(history.history['loss'],'b',label='Training loss')
plt.plot(history.history['val_loss'], 'r', label='Validation val_loss')
plt.title('Traing and Validation loss')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.savefig('image/log/v1/loss.jpg')

ypre = model.predict(xtrain)
f1s,cache = aced_utils.fmeasure(ytrain,ypre)
p,r = cache
print("f1 = {}, precision = {}, recall = {}".format(f1s,p,r))