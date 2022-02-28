import numpy as np
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt


def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def fbeta_score(y_true, y_pred, beta=1):
    y_true = np.float32(y_true)
    y_pred = np.float32(y_pred)
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    cache = (p,r)
    return fbeta_score,cache


def fmeasure(y_true, y_pred):
    return fbeta_score(y_true, y_pred, beta=1)

def testEvent(model,xtest,ytest,figname='1'):
    '''
    plot testevent (AR or global)
    :param model:
    :param xtest:
    :param ytest:
    :return:
    '''
    ypre = model.predict(xtest,verbose=1)
    print("y={}, probility={}".format(ytest,ypre))
    plt.figure()
    nc = xtest.shape[-1]
    for idx in range(nc):
        plt.subplot((nc+1)//2,2,idx+1)
        plt.imshow(xtest[0,:,:,idx])
    plt.savefig("figure/test/{}.jpg".format(figname))
    return ypre,ytest




if __name__ == '__main__':
    #creat_dataset()
    #creat_dataset_tot()
    #creat_dataset_single()
    # xtrain_orig, ytrain, xtest_orig, ytest, classes = load_dataset(filename='data/data60to30/data60to30.h5')
    # del xtrain_orig
    # del ytrain
    # #Y_train = ytrain.T
    # X_test = xtest_orig / 255.
    # Y_test = ytest.T
    # model = tensorflow.keras.models.load_model('model/v1/model_v1_3.h5')
    # testidx = 69
    # ypre,ytest = testEvent(model,X_test[testidx,None,:,:,:],Y_test[testidx],figname=testidx)
    # cvres = model.predict(X_test, verbose=1)
    # cvf1s, cache = fmeasure(Y_test, cvres)
    # p, r = cache
    #xtrain_orig, ytrain, classes = load_dataset_tot('data/data60/data60tot.h5')
    #model = modelV1([256,256,6])
    #model.summary()
    print("test down")


