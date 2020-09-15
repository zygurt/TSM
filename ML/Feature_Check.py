# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 12:29:13 2020

@author: s2599923
"""


import pickle
import librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np


def prep(x,threshold):
    #Convert to Mono by summing channels
    print('Before Summing')
    print(x.shape)
    if len(x.shape)>1:
        x = np.sum(x,axis=0)
        print('After Summing')
        print(x.shape)
    #Normalise
    print('Before Normalisation')
    print(np.min(x))
    print(np.max(x))
    x = np.divide(x,np.max(np.abs(x)))
    print('After Normalisation')
    print(np.min(x))
    print(np.max(x))
    #Trim Start and End
    print('Before Trimming')
    print(x.shape[0])
    start_sample = 0
    while np.sum(x[start_sample:start_sample+3]<threshold) and start_sample+3<x.shape[0]:
        start_sample+=1
    end_sample = x.shape[0]
    while np.sum(x[end_sample:end_sample+3]<threshold) and end_sample+3>0:
        end_sample -= 1
    x = x[start_sample:end_sample]
    print('After Trimming')
    print(x.shape[0])
    return x
    
    
# f = './data/Mexican_Flute_02_Elastique_49.7286_per.wav'
f = './data/audio/ref_train/Ardour_1.wav'
# f = './data/Alto_Sax_06_FESOLA_82.58_per.wav'
# f = './data/Alto_Sax_15_Elastique_92.6238_per.wav'
x,fs = librosa.load(f, mono=1, sr=None)
x_prep = prep(x,0.0061) #Using PEAQ threshold

x_MFCC = librosa.feature.mfcc(x,sr=fs, n_mfcc=128)
x_prep_MFCC = librosa.feature.mfcc(x_prep,sr=fs, n_mfcc=128)

#Adjust MFCC 0
x_MFCC[0,:] = np.mean(x_MFCC[1:-1,:],axis=0)
x_prep_MFCC[0,:] = np.mean(x_prep_MFCC[1:-1,:],axis=0)

x_Delta = librosa.feature.delta(x_MFCC, width = 9, order=1)
x_prep_Delta = librosa.feature.delta(x_prep_MFCC, width = 9, order=1)

x_Delta_Delta = librosa.feature.delta(x_Delta, width = 9, order=1)
x_prep_Delta_Delta = librosa.feature.delta(x_prep_Delta, width = 9, order=1)

#Normalise the output
x_MFCC = (x_MFCC-np.min(x_MFCC))/(np.max(x_MFCC)-np.min(x_MFCC))
x_prep_MFCC = (x_prep_MFCC-np.min(x_prep_MFCC))/(np.max(x_prep_MFCC)-np.min(x_prep_MFCC))

x_Delta = (x_MFCC-np.min(x_MFCC))/(np.max(x_MFCC)-np.min(x_MFCC))
x_prep_Delta = (x_prep_MFCC-np.min(x_prep_MFCC))/(np.max(x_prep_MFCC)-np.min(x_prep_MFCC))

x_Delta_Delta = (x_MFCC-np.min(x_MFCC))/(np.max(x_MFCC)-np.min(x_MFCC))
x_prep_Delta_Delta = (x_prep_MFCC-np.min(x_prep_MFCC))/(np.max(x_prep_MFCC)-np.min(x_prep_MFCC))

fig,ax = plt.subplots(nrows=2, ncols=1, sharex=True)
img = librosa.display.specshow(x_MFCC, y_axis='log',x_axis='time', sr=44100, ax=ax[0])
ax[0].set(title='x MFCCs')
ax[0].label_outer()
img = librosa.display.specshow(x_prep_MFCC, y_axis='log',x_axis='time', sr=44100, ax=ax[1])
ax[1].set(title='x prep MFCCs')
ax[1].label_outer()
fig.suptitle(f)
fig.colorbar(img, ax=ax)

fig,ax = plt.subplots(nrows=2, ncols=1, sharex=True)
img = librosa.display.specshow(x_Delta, y_axis='log',x_axis='time', sr=44100, ax=ax[0])
ax[0].set(title='x Delta')
ax[0].label_outer()
img = librosa.display.specshow(x_prep_Delta, y_axis='log',x_axis='time', sr=44100, ax=ax[1])
ax[1].set(title='x prep Delta')
ax[1].label_outer()
fig.suptitle(f)
fig.colorbar(img, ax=ax)

fig,ax = plt.subplots(nrows=2, ncols=1, sharex=True)
img = librosa.display.specshow(x_Delta_Delta, y_axis='log',x_axis='time', sr=44100, ax=ax[0])
ax[0].set(title='x Delta Delta')
ax[0].label_outer()
img = librosa.display.specshow(x_prep_Delta_Delta, y_axis='log',x_axis='time', sr=44100, ax=ax[1])
ax[1].set(title='x prep Delta Delta')
ax[1].label_outer()
fig.suptitle(f)
fig.colorbar(img, ax=ax)











# with open('Feat.p', 'rb') as pickle_file:
#     train_list, val_list, test_list, Norm_vals = pickle.load(pickle_file)
    
# for n in range(len(test_list)):
#     MFCC = test_list[n].get('MFCCs')
#     Deltas = test_list[n].get('Deltas')
#     fig,ax = plt.subplots(nrows=2, ncols=1, sharex=True)
#     img = librosa.display.specshow(MFCC, y_axis='log',x_axis='time', sr=44100, ax=ax[0])
#     ax[0].set(title='MFCCs')
#     ax[0].label_outer()
#     img = librosa.display.specshow(Deltas, y_axis='log',x_axis='time', sr=44100, ax=ax[1])
#     ax[1].set(title='Deltas')
#     ax[1].label_outer()
#     fig.suptitle(test_list[n].get('file'))
#     fig.colorbar(img, ax=ax)
#     plt.savefig('./Test/'+test_list[n].get('file')[0:-4]+'.png',format='png')
#     plt.close()

# for n in range(len(train_list)):
#     MFCC = train_list[n].get('MFCCs')
#     Deltas = train_list[n].get('Deltas')
#     fig,ax = plt.subplots(nrows=2, ncols=1, sharex=True)
#     img = librosa.display.specshow(MFCC, y_axis='log',x_axis='time', sr=44100, ax=ax[0])
#     ax[0].set(title='MFCCs')
#     ax[0].label_outer()
#     img = librosa.display.specshow(Deltas, y_axis='log',x_axis='time', sr=44100, ax=ax[1])
#     ax[1].set(title='Deltas')
#     ax[1].label_outer()
#     fig.suptitle(train_list[n].get('file'))
#     fig.colorbar(img, ax=ax)
#     plt.savefig('./Train/'+train_list[n].get('file')[0:-4]+'.png',format='png')
#     plt.close()
    
# for n in range(len(val_list)):
#     MFCC = val_list[n].get('MFCCs')
#     Deltas = val_list[n].get('Deltas')
#     fig,ax = plt.subplots(nrows=2, ncols=1, sharex=True)
#     img = librosa.display.specshow(MFCC, y_axis='log',x_axis='time', sr=44100, ax=ax[0])
#     ax[0].set(title='MFCCs')
#     ax[0].label_outer()
#     img = librosa.display.specshow(Deltas, y_axis='log',x_axis='time', sr=44100, ax=ax[1])
#     ax[1].set(title='Deltas')
#     ax[1].label_outer()
#     fig.suptitle(train_list[n].get('file'))
#     fig.colorbar(img, ax=ax)
#     plt.savefig('./Val/'+val_list[n].get('file')[0:-4]+'.png',format='png')
#     plt.close()