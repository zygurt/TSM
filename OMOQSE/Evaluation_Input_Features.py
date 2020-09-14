# Learning file import


import os, pickle, sys

from os import listdir
from os.path import isfile, join
import numpy as np
import scipy.io as sio
from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt

import librosa, librosa.display

def prep(x,threshold=0.0061):
    #Default PEAQ threshold
    #Convert to Mono by summing channels
    # print('Before Summing')
    # print(x.shape)
    if len(x.shape)>1:
        x = np.sum(x,axis=0)
        # print('After Summing')
        # print(x.shape)
    #Normalise
    # print('Before Normalisation')
    # print(np.min(x))
    # print(np.max(x))
    x = np.divide(x,np.max(np.abs(x)))
    # print('After Normalisation')
    # print(np.min(x))
    # print(np.max(x))
    #Trim Start and End
    # print('Before Trimming')
    # print(x.shape[0])
    start_sample = 0
    while np.sum(x[start_sample:start_sample+3]<threshold) and start_sample+3<x.shape[0]:
        start_sample+=1
    end_sample = x.shape[0]
    while np.sum(x[end_sample:end_sample+3]<threshold) and end_sample+3>0:
        end_sample -= 1
    x = x[start_sample:end_sample]
    # print('After Trimming')
    # print(x.shape[0])
    return x

def rec_filelist(eval_files, flist):
    for f in listdir(flist):
        if not isfile(flist+'/'+f):
            rec_filelist(eval_files, flist+'/'+f)
        elif isfile(flist+'/'+f):
            eval_files.append(flist+'/'+f)
    return eval_files



num_mfccs = 128
delta_width = 9
delta_short = 3

norm_methods = ['None','Freq Dim','Overall']
norm_method = 0


data_path = './data/Features/Eval_MFCC_Lib/'
if os.path.exists(data_path + '/Feat_Eval.p'):
    with open(data_path + '/Feat_Eval.p', 'rb') as f:
        eval = pickle.load(f)
else:
    eval_path = '../Subjective_Testing/Eval'

    #Create list of files in an arbitary order
    eval_files = []
    eval_files = rec_filelist(eval_files, eval_path)
    # print(eval_files)
    # sys.exit()
    eval_list = []

    N = 2048
    min_MFCC_len = 2000
    max_MFCC_len = 0
    min_POW_len = 2000
    max_POW_len = 0
    num_norm_files = 1000


    MFCC_vals = np.zeros((num_mfccs,1))
    Deltas_vals = np.zeros((num_mfccs,1))
    Deltas_Delta_vals = np.zeros((num_mfccs,1))
    #Evaluation files
    if os.path.exists(data_path + '/Feat_Eval.p'):
        with open(data_path + '/Feat_Eval.p', 'rb') as f:
            eval_list  = pickle.load(f)
    else:
        print("Creating Evaluation data")
        for f in range(len(eval_files)):
            print("Reading file: ", f, " ", eval_files[f])
            x, fs = librosa.load(eval_files[f], sr=None, mono=0)
            x = prep(x)
            MFCCs = librosa.feature.mfcc(x, sr=fs, n_mfcc=num_mfccs)
            Deltas = librosa.feature.delta(MFCCs,width=delta_width, order=1)
            Deltas_Short = librosa.feature.delta(MFCCs,width=delta_short, order=1)
            D_Deltas = librosa.feature.delta(Deltas,width=delta_width, order=1)
            L = MFCCs.shape[1]
            if L < min_MFCC_len:
                min_MFCC_len = L
            if L > max_MFCC_len:
                max_MFCC_len = L
            #Add to list of dictionaries
            eval_list.append({
                "file": eval_files[f],
                # "MATLAB_loc": MATLAB_NAME[nc],
                "MeanOS": 0,
                "MedianOS": 0,
                "MFCCs": MFCCs,
                "Deltas": Deltas,
                "Deltas_Short": Deltas_Short,
                "D_Deltas": D_Deltas,
                "L": MFCCs.shape[1],
                "fnum": f})

        #IF GOING TO NORMALISE< NEED TO IMPORT NORM VALUES FROM A SAVED FILE
        # if norm_method == 1:
        #     # Frequency Dimension Normalisation
        #     # Calculate the mean and std
        #     MFCC_mean = np.mean(MFCC_vals, axis=1).reshape(-1,1)
        #     MFCC_std = np.std(MFCC_vals, axis=1).reshape(-1,1)
        #     Deltas_mean = np.mean(Deltas_vals, axis=1).reshape(-1,1)
        #     Deltas_std = np.std(Deltas_vals, axis=1).reshape(-1,1)
        #     # Normalise the MFCCs and Deltas
        #     for n in range(len(train_list)):
        #         # norm = (feature-feature_mean)/feature_std #Readable explanation
        #         train_list[n]["MFCCs_Norm"] = (train_list[n].get('MFCCs')-np.broadcast_to(MFCC_mean,(train_list[n].get('MFCCs').shape)))/np.broadcast_to(MFCC_std,(train_list[n].get('MFCCs').shape))
        #         train_list[n]["Deltas_Norm"] = (train_list[n].get('Deltas')-np.broadcast_to(Deltas_mean,(train_list[n].get('Deltas').shape)))/np.broadcast_to(Deltas_std,(train_list[n].get('Deltas').shape))
        #     Norm_vals = np.concatenate((MFCC_mean, MFCC_std, Deltas_mean, Deltas_std), axis=1)
        # elif norm_method == 2:
        #     # Overall Normalisation
        #     # Calculate the mean and std
        #     MFCC_mean = np.mean(MFCC_vals)
        #     MFCC_std = np.std(MFCC_vals)
        #     Deltas_mean = np.mean(Deltas_vals)
        #     Deltas_std = np.std(Deltas_vals)
        #     # Normalise the MFCCs and Deltas
        #     for n in range(len(train_list)):
        #         # norm = (feature-feature_mean)/feature_std #Readable explanation
        #         train_list[n]["MFCCs_Norm"] = np.divide(np.subtract(train_list[n].get('MFCCs'),MFCC_mean),MFCC_std)
        #         train_list[n]["Deltas_Norm"] = np.divide(np.subtract(train_list[n].get('Deltas'),Deltas_mean),Deltas_std)
        #     Norm_vals = np.array([MFCC_mean, MFCC_std, Deltas_mean, Deltas_std])
        # else:
        #     #norm_method == 0:
        #     #No Normalisation
        #     MFCC_mean = 0
        #     MFCC_std = 1
        #     Deltas_mean = 0
        #     Deltas_std = 1
        #     D_Deltas_mean = 0
        #     D_Deltas_std = 1
        #     # for n in range(len(train_list)):
        #     #     train_list[n]["MFCCs_Norm"] = train_list[n].get('MFCCs')
        #     #     train_list[n]["Deltas_Norm"] = train_list[n].get('Deltas')
        #     Norm_vals = np.array([MFCC_mean, MFCC_std, Deltas_mean, Deltas_std, D_Deltas_mean, D_Deltas_std])

        # Save Evaluation data
        if not os.path.exists(data_path): os.makedirs(data_path)
        print("Saving Evaluation data")
        with open(data_path + 'Feat_Eval.p', 'wb') as f:
            pickle.dump(eval_list, f)


file1 = open(data_path+"/Feature_info.txt","a")
file1.write("Number of MFCCs: ")
file1.write(str(num_mfccs))
file1.write("\n")
file1.write("Delta Width: ")
file1.write(str(delta_width))
file1.write("\n")
file1.write("N: ")
file1.write(str(N))
file1.write("\n")
file1.write("Number of files used to normalise: ")
file1.write(str(num_norm_files))
file1.write("\n")
file1.write("Normalisation method: ")
file1.write(norm_methods[norm_method])
file1.write("\n")
file1.write("Minimum MFCC Length: ")
file1.write(str(min_MFCC_len))
file1.write("\n")
file1.write("Maximum MFCC Length: ")
file1.write(str(max_MFCC_len))
file1.write("\n")


file1.close()
print("Complete")
print("Min Length = ", min_MFCC_len)
print("Max Length = ", max_MFCC_len)
