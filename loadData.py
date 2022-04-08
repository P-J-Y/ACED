import datetime
from os import listdir
from os.path import isfile, join

import pandas as pd
from netCDF4 import Dataset
import numpy as np

time_19700101 = datetime.datetime(1970, 1, 1)

# load all nc files
def loadData(path):
    # search for all files ending with .nc
    files = [f for f in listdir(path) if isfile(join(path, f)) and f.endswith('.nc')]
    # load all files
    iondata = []
    magdata = []
    for file in files:
        if 'm1s' in file:
            magdata.append(Dataset(join(path, file), 'r'))
        elif 'f3s' in file:
            iondata.append(Dataset(join(path, file), 'r'))
    return iondata, magdata

def loadIonData(iondata):
    time = []
    Vp = []
    Np = []
    Tp = []
    NHe = []
    for ion in iondata:
        # get time
        thetime = [time_19700101 + datetime.timedelta(milliseconds=t) for t in ion.variables['time'][:]]
        time.extend(thetime)
        # get Vp
        Vp.extend(ion.variables['proton_speed'][:])
        # get Np
        Np.extend(ion.variables['proton_density'][:])
        # get Tp
        Tp.extend(ion.variables['proton_temperature'][:])
        # get NHe
        NHe.extend(ion.variables['alpha_density'][:])
    # merge lists
    if sum(np.isnan(NHe))/len(NHe)>=0.5:
        swe = [np.array(Np), np.array(Tp), np.array(Vp)]
        swe = np.array(swe).T
    else:
        swe = [np.array(Np), np.array(Tp), np.array(Vp), np.array(NHe) / np.array(Np)]
        swe = np.array(swe).T
    nanpoints = np.isnan(np.sum(swe, axis=1))
    swe = swe[~nanpoints, :]
    time = np.array(time)[~nanpoints]
    sortperm = np.argsort(time)
    swe = swe[sortperm, :]
    time = time[sortperm]
    return swe, time

def loadMagData(magdata):
    time = []
    B = []
    for mag in magdata:
        # get time
        thetime = [time_19700101 + datetime.timedelta(milliseconds=t) for t in mag.variables['time'][:]]
        time.extend(thetime)
        # get B
        B.extend(mag.variables['bt'][:])
    nanpoints = np.isnan(B)
    B = np.array(B)[~nanpoints]
    time = np.array(time)[~nanpoints]
    sortperm = np.argsort(time)
    B = B[sortperm]
    time = time[sortperm]
    return B, time

def datetime_range(t1, t2, delta):
    time = []
    while t1 < t2:
        time.append(t1)
        t1 += delta
    return np.array(time)

def outputdata_genesis(filepath='data/origin/DSCOVR/data/2022/01'):
    iondata, magdata = loadData(filepath)
    swe, swet = loadIonData(iondata)
    mag, magt = loadMagData(magdata)
    t1 = max(swet[0],magt[0])
    t2 = min(swet[-1], magt[-1])
    yt = datetime_range(t1,t2,datetime.timedelta(hours=1))
    b = np.interp([datetime.datetime.timestamp(t) for t in yt], [datetime.datetime.timestamp(t) for t in magt], mag)
    return swe, b, yt, swet

def outputdata_xb(filepath='data/origin/DSCOVR/data/2022'):
    iondata = []
    magdata = []
    for path in listdir(filepath):
        thepath = join(filepath, path)
        theiondata, themagdata = loadData(thepath)
        iondata.extend(theiondata)
        magdata.extend(themagdata)
    swe, swet = loadIonData(iondata)
    mag, magt = loadMagData(magdata)
    t1 = max(swet[0],magt[0])
    t2 = min(swet[-1], magt[-1])
    yt = datetime_range(t1,t2,datetime.timedelta(hours=1))
    b = np.interp([datetime.datetime.timestamp(t) for t in yt], [datetime.datetime.timestamp(t) for t in magt], mag)
    swe = [np.interp([datetime.datetime.timestamp(t) for t in yt], [datetime.datetime.timestamp(t) for t in swet], swe[:,i]) for i in range(swe.shape[1])]
    swe = np.array(swe).T
    xdata = np.append(swe,b.reshape(len(b),1),axis=1)
    return xdata, yt


outputdata_xb()



# time
# proton_speed
# proton_density
# proton_temperature
# alpha_density
#
# time
# bt
print('test')