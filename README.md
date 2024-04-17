# Epileptic_seizure_detection_CNN
## Overview
This study investigates the well-performed EEG-based ES detection method by decomposing EEG signals.
## Prerequisites
The Python programming language is used to conduct the experiments. Decomposition and feature extraction, with model training, are powered by a P100 GPU of Kaggle's platform. 5-fold cross-validation (CV) is used to split training and test sets.
## Implementation Demo
### 1. Import library
```
!pip install EMD-signal
from scipy import stats
from scipy import fftpack
from PyEMD import EMD, EEMD # https://pyemd.readthedocs.io/en/latest/usage.html
from scipy.signal import find_peaks
from scipy.stats import kurtosis, skew
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale
import pandas as pd
import numpy as np
import statistics
import pickle
import math
import cmath
import os
import math 
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import tensorflow as tf
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Dense, Activation,Conv2D,MaxPooling2D, Dropout, Flatten
from sklearn.model_selection import *
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
import os
from sklearn.feature_selection import SelectKBest, SelectPercentile, mutual_info_classif
from sklearn.neural_network import MLPClassifier
from numpy import mean
```
### 2. Create IMFs using EMD
```
def cal_IMF(start, channel, data): 
    # print(data.shape)
    sample_rate = 256
    seconds = 10
    num_samples = sample_rate*seconds
    time_vect = (data.iloc[start:start+num_samples, channel])
    time_vect = np.array(time_vect)
    emd = EMD()
    imfs = emd.emd(time_vect)
    return imfs
```
### 3. Create features
#### Calculate Fluctuation Index
```
def fluctuation_index(x):
    sum = 0
    # print(x.shape[0]-1)
    for i in range(0, x.shape[0]-1):
        sum+=abs(x[i+1]-x[i])
        # print(sum)
    return sum/(x.shape[0]-1)
```
#### Calculate Ellipse Area of SODP
```
def calc_area_of_sodp(X,Y,i,channel):
        #Area of Second Order Difference Plot
        SX = math.sqrt(np.sum(np.multiply(X,X))/len(X))
        SY = math.sqrt(np.sum(np.multiply(Y,Y))/len(Y))
        SXY = np.sum(np.multiply(X,Y))/len(X)
        D = cmath.sqrt((SX*SX) + (SY*SY) - (4*(SX*SX*SY*SY - SXY*SXY)))
        # print(D)
        a = 1.7321 *cmath.sqrt(SX*SX + SY*SY + D)
        b = 1.7321 * cmath.sqrt(SX*SX + SY*SY - D)
        Area = math.pi *a *b
        # print(SX,SY,SXY, D, a, b, Area)
        # print("Channel=  ",channel,"Area of SODP of IMF number= ",i, " is ", Area.real, " ", Area.imag)
        return Area.real
```
```
def SODP(y, i, channel):
        #remove outliers
        upper_quartile = np.percentile(y,80)
        lower_quartile = np.percentile(y,20)
        IQR = (upper_quartile - lower_quartile) * 1.5
        quartileSet = (lower_quartile- IQR, upper_quartile +IQR)
        y = y[np.where((y >= quartileSet[0]) & (y <= quartileSet[1]))]
        
        #plotting SODP
        X = np.subtract(y[1:],y[0:-1]) #x(n+1)-x(n)
        Y = np.subtract(y[3:],y[0:-3]).tolist()#x(n+2)-x(n-1)
        Y.extend([0])
        Y.extend([0])
        Area = calc_area_of_sodp(X,Y,i,channel) 
        return Area
```
### 4. Set Segmentation, samlping rate
```
seg = 10
samp = 256
non_overlap = 3
overlap = 7
step = non_overlap*samp
dist=seg*samp
```
### 5. Start segmentation, create features and store for seizure data
```
final_feature=[]
seizure_data = pd.read_csv("/kaggle/input/prev-chbmit-raw-data-sc/merge_all_seizure.csv",header=None)
seizure = seizure_data.shape[0]
print(seizure)
cnt =0
step = non_overlap*samp
for k in range (0,data_shape-(overlap*samp)-step,step):
# for k in range(0,2):
#     print(k//samp)
    cnt=cnt+1
    # print(cnt)
    feature = []
    ok = 1
    for i in range(0, 22):               
        imf = cal_IMF(k,i,seizure_data)
        imf = np.array(imf)
        if(imf.shape[0]<6):
            print(k,i,"boom")
            i=22
            ok = 0
            continue
        row = []
        for j in range(0,6):
            m = fluctuation_index(imf[j])
            n = statistics.variance(imf[j]) # entropy korte hbe
            o = SODP(imf[j], j, i)
            # final[k//step][i][3*j+0]=float(m)
            # final[k//step][i][3*j+1]=float(n)
            # final[k//step][i][3*j+2]=float(o)
            row.append(m)
            row.append(n)
            row.append(o)
        feature.append(row)
    if ok:
        feature=np.array(feature)
        final_feature.append(feature)
        print("final feature len ", len(final_feature))
                
print(cnt)
```
### 6. Repeate step 5 for Non-seizure data and append features to 'final_feature' array
### 7. Add label and create 5 fold train-test dataset
```
temp = []
seizure = int(final.shape[0]/2)
for i in range(0, final.shape[0]):
    tt = final[i].reshape((396,))
    tt=tt.tolist()
    if(i<seizure):
        tt.append(1)
    else:
        tt.append(0)
    temp.append(tt)
temp=np.array(temp)
temp.shape
```
### 8. Train test using 2D CNN
```
for idx in range (1,6):
    file = open(f"/kaggle/input/5f-train-test-val-dataset/x_train{idx}.txt","rb")
    x_train = pickle.load(file)
    file.close()

    file = open(f"/kaggle/input/5f-train-test-val-dataset/x_test{idx}.txt","rb")
    x_test = pickle.load( file)
    file.close()

    file = open(f"/kaggle/input/5f-train-test-val-dataset/y_train{idx}.txt","rb")
    y_train = pickle.load(file)
    file.close()

    file = open(f"/kaggle/input/5f-train-test-val-dataset/y_test{idx}.txt","rb")
    y_test = pickle.load( file)
    file.close()
    
    tx=[]
    for i in range (x_train.shape[0]):
        x1=x_train[i,:].reshape((22,18))
        tx.append(x1)
    x_train=np.asarray(tx)
    print(x_train.shape)
    tx=[]
    for i in range (x_test.shape[0]):
        x1=x_test[i,:].reshape((22,18))
        tx.append(x1)
    x_test=np.asarray(tx)
    print(x_test.shape)

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    mi_max = 100
    acc_mx = -100
    k="nofs"
    name = "CNN2d10"
    X_train_top = x_train
    X_test_top = x_test

    #################################  Standard Scaler #####################
    from sklearn.preprocessing import StandardScaler
    for i in range(X_train_top.shape[0]):
        sc = StandardScaler()
        # X = sc.fit_transform(X)
        X_train_top[i] = sc.fit_transform(X_train_top[i])
        
    for i in range(X_test_top.shape[0]):
        sc = StandardScaler()
        # X = sc.fit_transform(X)
        X_test_top[i] = sc.fit_transform(X_test_top[i])

    from keras.models import Sequential
    from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization

    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=(22,18,1)))
    model.add(MaxPooling2D())
    model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(10))
    model.add(Dense(2, activation='softmax'))

    ################################# compile model #####################
    lr = .001
    batch_size = 128
    opt = tf.keras.optimizers.Adam(learning_rate = lr)
    model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])
    ################################# fit model #####################

    res_path = f'/kaggle/working/{name}_{idx}/mi_{k}/{idx}_MI_{k}.h5'
    mc = ModelCheckpoint(res_path, monitor='val_accuracy', mode='max', verbose=0, save_best_only=True)
    history = model.fit(X_train_top,y_train,validation_data = (X_test_top, y_test),batch_size=batch_size,epochs=700,verbose=0,callbacks=[mc])

    ################################# history data #####################
    train_loss=history.history['loss']
    val_loss=history.history['val_loss']
    accuracy=history.history['accuracy']
    val_accuracy=history.history['val_accuracy']


    ################################# dump history data #####################
    x = np.array(list(range(1,len(train_loss)+1)))
    loss = 'model1'
    datas = {'epoch':x, 'Training loss':train_loss}
    lossepoch = pd.DataFrame(datas)
    lossepoch.to_csv(f'/kaggle/working/{name}_{idx}/mi_{k}/{loss}_train_loss.csv', index = False)

    datas = {'epoch':x, 'Test loss':val_loss}
    lossepoch = pd.DataFrame(datas)
    lossepoch.to_csv(f'/kaggle/working/{name}_{idx}/mi_{k}/{loss}_test_loss.csv', index = False)

    datas = {'epoch':x, 'Training accuracy':accuracy}
    lossepoch = pd.DataFrame(datas)
    lossepoch.to_csv(f'/kaggle/working/{name}_{idx}/mi_{k}/{loss}_train_accuracy.csv', index = False)

    datas = {'epoch':x, 'Test accuracy':val_accuracy}
    lossepoch = pd.DataFrame(datas)
    lossepoch.to_csv(f'/kaggle/working/{name}_{idx}/mi_{k}/{loss}_test_accuracy.csv', index = False)

    ########################## performance evalution #############################
    model.load_weights(res_path)
    scores = model.evaluate(X_test_top, y_test, verbose=1, batch_size=batch_size)
    pred=model.predict(X_test_top, verbose=0, batch_size=batch_size)
    pred=np.round(pred)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

    os.rename(f'/kaggle/working/{name}_{idx}/mi_{k}', f'/kaggle/working/{name}_{idx}/mi_{k}_{round(scores[1],4)}')
    acc_mx = round(scores[1],4)
    os.rename(f'/kaggle/working/{name}_{idx}', f'/kaggle/working/{name}_{idx}_{acc_mx}')

```
