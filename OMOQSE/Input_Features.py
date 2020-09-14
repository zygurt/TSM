# Learning file import


import os, pickle, sys

from os import listdir
from os.path import isfile, join
import numpy as np
import scipy.io as sio
from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt
# import filterbank as fbank
# import torchaudio
# import torch
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
    end_sample = x.shape[0]-3
    while np.sum(x[end_sample:end_sample+3]<threshold) and end_sample+3>0:
        end_sample -= 1
    x = x[start_sample:end_sample]
    # print('After Trimming')
    # print(x.shape[0])
    return x



num_mfccs = 128
delta_short = 3
delta_width = 9
N = 2048
fs = 44100
# MFCC_Calc = torchaudio.transforms.MFCC(sample_rate=fs, n_mfcc=num_mfccs, dct_type= 2, norm= 'ortho', log_mels= False, melkwargs={"hop_length":512,"n_fft":2048,"n_mels":num_mfccs})
# sample_rate: int= 16000, n_mfcc: int= 40, dct_type: int= 2, norm: str= 'ortho', log_mels: bool= False, melkwargs: Optional[dict] = None
norm_methods = ['None','Freq Dim','Overall']
norm_method = 0


data_path = './data/Features/MFCC_Lib/'
if os.path.exists(data_path + '/Feat.p'):
    with open(data_path + '/Feat.p', 'rb') as f:
        train_list, test_list, Norm_vals = pickle.load(f)
else:
    # Load the MOS data from matlab
    load_file = './data/OMOQ_CNN_MOS_Col.mat'
    features = sio.loadmat(load_file)
    Names = []
    MATLAB_NAME = []
    # print(features['Name'][1][0][0])
    # print(len(features['MeanOS'][0]))
    # print(features['Name'])
    # sys.exit()
    print("Length of features: "+str(len(features['MeanOS'][0])))
    # sys.exit()
    for n in range(0,len(features['MeanOS'][0])):
        # print(features['Name'][0][n][0])
        # print(features['MeanOS'][0][n])
        temp = features['Name'][0][n][0].split('/')[-1]
        Names.append(temp)
        temp = features['Name'][0][n][0]
        MATLAB_NAME.append(temp)
    # sys.exit()
    MeanOS = features['MeanOS'][0]
    MedianOS = features['MedianOS'][0]
    # print("Name = ", Names)
    # print("MeanOS = ", MeanOS)
    # print("MedianOS = ", MedianOS)
    # sys.exit()
    test_path = '../ML/data/audio/test/'
    train_path = '../ML/data/audio/train/'
    #Create list of files in an arbitary order
    test_files = [f for f in listdir(test_path) if isfile(join(test_path, f))]
    train_files = [f for f in listdir(train_path) if isfile(join(train_path, f))]
    # print(test_files)
    # sys.exit()
    train_list = []
    val_list = []
    test_list = []

    min_MFCC_len = 2000
    max_MFCC_len = 0
    min_POW_len = 2000
    max_POW_len = 0
    num_norm_files = 1000

    val_per = 0.1
    train_per = 1-val_per
    num_training_files = int(5280*train_per)
    rand_order = np.random.permutation(len(train_files))
    print("Number of Training Files: "+str(num_training_files))
    MFCC_vals = np.zeros((num_mfccs,1))
    Deltas_vals = np.zeros((num_mfccs,1))
    Deltas_Delta_vals = np.zeros((num_mfccs,1))
    #Training files
    if os.path.exists(data_path + '/Feat_Train.p'):
        with open(data_path + '/Feat_Train.p', 'rb') as f:
            train_list  = pickle.load(f)
    else:
        print("Creating Training data")
        for f in range(num_training_files):
            print("Reading file: ", f, " ", train_path+train_files[rand_order[f]])
            # fs, x = wavfile.read(train_path+train_files[f])
            x, fs = librosa.load(train_path+train_files[rand_order[f]], sr=None, mono=0)
            x = prep(x)
            MFCCs = librosa.feature.mfcc(x, sr=fs, n_mfcc=num_mfccs)
            Deltas = librosa.feature.delta(MFCCs,width=delta_width, order=1)
            Deltas_Short = librosa.feature.delta(MFCCs,width=delta_short, order=1)
            D_Deltas = librosa.feature.delta(Deltas,width=delta_width, order=1)

            # #Pytorch Features
            # MFCCs_tensor = MFCC_Calc(torch.Tensor(x))
            # Deltas_tensor = torchaudio.functional.compute_deltas(MFCCs_tensor, win_length=delta_width, mode= 'replicate')
            # D_Deltas_tensor = torchaudio.functional.compute_deltas(Deltas_tensor, win_length=delta_width, mode= 'replicate')
            # MFCCs = np.squeeze(np.array(torch.Tensor.tolist(MFCCs_tensor)))
            # Deltas = np.squeeze(np.array(torch.Tensor.tolist(Deltas_tensor)))
            # D_Deltas = np.squeeze(np.array(torch.Tensor.tolist(D_Deltas_tensor)))


            L = MFCCs.shape[1]
            # fnum = f
            if L < min_MFCC_len:
                min_MFCC_len = L
                # print("Min Length = ", min_len)
            if L > max_MFCC_len:
                max_MFCC_len = L
            # Find the matching MOS
            for nc in range(len(MeanOS)):
                if Names[nc] in train_files[rand_order[f]]:
                    # print("Train_files", train_files[f], "Matlab name: ", MATLAB_NAME[n], "MeanOS", MeanOS[n])
                    # sys.exit()
                    break
            #Add to list of dictionaries
            train_list.append({
                "file": train_files[rand_order[f]],
                "MATLAB_loc": MATLAB_NAME[nc],
                "MeanOS": MeanOS[nc],
                "MedianOS": MedianOS[nc],
                "MFCCs": MFCCs,
                "Deltas": Deltas,
                "Deltas_Short": Deltas_Short,
                "D_Deltas": D_Deltas,
                "L": MFCCs.shape[1],
                "fnum": f})


        if norm_method == 1:
            # Frequency Dimension Normalisation
            # Calculate the mean and std
            MFCC_mean = np.mean(MFCC_vals, axis=1).reshape(-1,1)
            MFCC_std = np.std(MFCC_vals, axis=1).reshape(-1,1)
            Deltas_mean = np.mean(Deltas_vals, axis=1).reshape(-1,1)
            Deltas_std = np.std(Deltas_vals, axis=1).reshape(-1,1)
            # Normalise the MFCCs and Deltas
            for n in range(len(train_list)):
                # norm = (feature-feature_mean)/feature_std #Readable explanation
                train_list[n]["MFCCs_Norm"] = (train_list[n].get('MFCCs')-np.broadcast_to(MFCC_mean,(train_list[n].get('MFCCs').shape)))/np.broadcast_to(MFCC_std,(train_list[n].get('MFCCs').shape))
                train_list[n]["Deltas_Norm"] = (train_list[n].get('Deltas')-np.broadcast_to(Deltas_mean,(train_list[n].get('Deltas').shape)))/np.broadcast_to(Deltas_std,(train_list[n].get('Deltas').shape))
            Norm_vals = np.concatenate((MFCC_mean, MFCC_std, Deltas_mean, Deltas_std), axis=1)
        elif norm_method == 2:
            # Overall Normalisation
            # Calculate the mean and std
            MFCC_mean = np.mean(MFCC_vals)
            MFCC_std = np.std(MFCC_vals)
            Deltas_mean = np.mean(Deltas_vals)
            Deltas_std = np.std(Deltas_vals)
            # Normalise the MFCCs and Deltas
            for n in range(len(train_list)):
                # norm = (feature-feature_mean)/feature_std #Readable explanation
                train_list[n]["MFCCs_Norm"] = np.divide(np.subtract(train_list[n].get('MFCCs'),MFCC_mean),MFCC_std)
                train_list[n]["Deltas_Norm"] = np.divide(np.subtract(train_list[n].get('Deltas'),Deltas_mean),Deltas_std)
            Norm_vals = np.array([MFCC_mean, MFCC_std, Deltas_mean, Deltas_std])
        else:
            #norm_method == 0:
            #No Normalisation
            MFCC_mean = 0
            MFCC_std = 1
            Deltas_mean = 0
            Deltas_std = 1
            D_Deltas_mean = 0
            D_Deltas_std = 1
            # for n in range(len(train_list)):
            #     train_list[n]["MFCCs_Norm"] = train_list[n].get('MFCCs')
            #     train_list[n]["Deltas_Norm"] = train_list[n].get('Deltas')
            Norm_vals = np.array([MFCC_mean, MFCC_std, Deltas_mean, Deltas_std, D_Deltas_mean, D_Deltas_std])

        #Store for later use
        if not os.path.exists(data_path): os.makedirs(data_path)
        with open(data_path + '/Feat_Train_Norm_Vals.p', 'wb') as f:
            pickle.dump(Norm_vals, f)

        # Save Training data
        print("Saving Training data")
        with open(data_path + '/Feat_Train.p', 'wb') as f:
            pickle.dump(train_list, f)

    #--------------------Validation Set------------------------------

    if os.path.exists(data_path + '/Feat_Val.p'):
        with open(data_path + '/Feat_Val.p', 'rb') as f:
            val_list  = pickle.load(f)
    else:
        print("Creating Validation data")
        for f in range(num_training_files,5280):
            print("Reading file: ", f,"fnum: ",f-num_training_files, " ", train_path+train_files[rand_order[f]])
            x, fs = librosa.load(train_path+train_files[rand_order[f]], sr=None, mono=0)
            x = prep(x)
            MFCCs = librosa.feature.mfcc(x, sr=fs, n_mfcc=num_mfccs)
            Deltas = librosa.feature.delta(MFCCs,width=delta_width, order=1)
            Deltas_Short = librosa.feature.delta(MFCCs,width=delta_short, order=1)
            D_Deltas = librosa.feature.delta(Deltas,width=delta_width, order=1)
            # # TorchAudio Features
            # MFCCs_tensor = MFCC_Calc(torch.Tensor(x))
            # Deltas_tensor = torchaudio.functional.compute_deltas(MFCCs_tensor, win_length=delta_width, mode= 'replicate')
            # D_Deltas_tensor = torchaudio.functional.compute_deltas(Deltas_tensor, win_length=delta_width, mode= 'replicate')
            # MFCCs = np.squeeze(np.array(torch.Tensor.tolist(MFCCs_tensor)))
            # Deltas = np.squeeze(np.array(torch.Tensor.tolist(Deltas_tensor)))
            # D_Deltas = np.squeeze(np.array(torch.Tensor.tolist(D_Deltas_tensor)))
            L = MFCCs.shape[1]
            if L < min_MFCC_len:
                min_MFCC_len = L
                # print("Min Length = ", min_len)
            if L > max_MFCC_len:
                max_MFCC_len = L
            # Find the matching MOS
            for nc in range(len(MeanOS)):
                if Names[nc] in train_files[rand_order[f]]:
                    # print("Train_files", train_files[f], "Matlab name: ", MATLAB_NAME[n], "MeanOS", MeanOS[n])
                    # sys.exit()
                    break

            #Add to list of dictionaries
            val_list.append({
                "file": train_files[rand_order[f]],
                "MATLAB_loc": MATLAB_NAME[nc],
                "MeanOS": MeanOS[nc],
                "MedianOS": MedianOS[nc],
                "MFCCs": MFCCs,
                "Deltas": Deltas,
                "Deltas_Short": Deltas_Short,
                "D_Deltas": D_Deltas,
                "L": MFCCs.shape[1],
                "fnum": f-num_training_files})

        if norm_method == 1:
            # Frequency Dimension Normalisation
            # Calculate the mean and std
            MFCC_mean = np.mean(MFCC_vals, axis=1).reshape(-1,1)
            MFCC_std = np.std(MFCC_vals, axis=1).reshape(-1,1)
            Deltas_mean = np.mean(Deltas_vals, axis=1).reshape(-1,1)
            Deltas_std = np.std(Deltas_vals, axis=1).reshape(-1,1)
            # Normalise the MFCCs and Deltas
            for n in range(len(val_list)):
                # norm = (feature-feature_mean)/feature_std #Readable explanation
                val_list[n]["MFCCs_Norm"] = (val_list[n].get('MFCCs')-np.broadcast_to(MFCC_mean,(val_list[n].get('MFCCs').shape)))/np.broadcast_to(MFCC_std,(val_list[n].get('MFCCs').shape))
                val_list[n]["Deltas_Norm"] = (val_list[n].get('Deltas')-np.broadcast_to(Deltas_mean,(val_list[n].get('Deltas').shape)))/np.broadcast_to(Deltas_std,(val_list[n].get('Deltas').shape))
            Norm_vals = np.concatenate((MFCC_mean, MFCC_std, Deltas_mean, Deltas_std), axis=1)
        elif norm_method == 2:
            # Overall Normalisation
            # Calculate the mean and std
            MFCC_mean = np.mean(MFCC_vals)
            MFCC_std = np.std(MFCC_vals)
            Deltas_mean = np.mean(Deltas_vals)
            Deltas_std = np.std(Deltas_vals)
            # Normalise the MFCCs and Deltas
            for n in range(len(val_list)):
                # norm = (feature-feature_mean)/feature_std #Readable explanation
                val_list[n]["MFCCs_Norm"] = np.divide(np.subtract(val_list[n].get('MFCCs'),MFCC_mean),MFCC_std)
                val_list[n]["Deltas_Norm"] = np.divide(np.subtract(val_list[n].get('Deltas'),Deltas_mean),Deltas_std)
            Norm_vals = np.array([MFCC_mean, MFCC_std, Deltas_mean, Deltas_std])
        else:
            #norm_method == 0:
            #No Normalisation
            MFCC_mean = 0
            MFCC_std = 1
            Deltas_mean = 0
            Deltas_std = 1
            D_Deltas_mean = 0
            D_Deltas_std = 1
            # for n in range(len(train_list)):
            #     train_list[n]["MFCCs_Norm"] = train_list[n].get('MFCCs')
            #     train_list[n]["Deltas_Norm"] = train_list[n].get('Deltas')
            Norm_vals = np.array([MFCC_mean, MFCC_std, Deltas_mean, Deltas_std, D_Deltas_mean, D_Deltas_std])

        #Store for later use
        if not os.path.exists(data_path): os.makedirs(data_path)
        with open(data_path + '/Feat_Val_Norm_Vals.p', 'wb') as f:
            pickle.dump(Norm_vals, f)

        # Save Validation data
        print("Saving Validation data")
        with open(data_path + '/Feat_Val.p', 'wb') as f:
            pickle.dump(val_list, f)

    if os.path.exists(data_path + '/Feat_Test.p'):
        with open(data_path + '/Feat_Test.p', 'rb') as f:
            test_list  = pickle.load(f)
    else:
        #Testing Files
        print("Creating Test data")
        for f in range(len(test_files)):
            print("Reading file: ", f, " ", test_path+test_files[f])
            x, fs = librosa.load(test_path+test_files[f], sr=None, mono=0)
            x = prep(x)
            MFCCs = librosa.feature.mfcc(x, sr=fs, n_mfcc=num_mfccs)
            Deltas = librosa.feature.delta(MFCCs,width=delta_width, order=1)
            Deltas_Short = librosa.feature.delta(MFCCs,width=delta_short, order=1)
            D_Deltas = librosa.feature.delta(Deltas,width=delta_width, order=1)
            # #TorchAudio Features
            # MFCCs_tensor = MFCC_Calc(torch.Tensor(x))
            # Deltas_tensor = torchaudio.functional.compute_deltas(MFCCs_tensor, win_length=delta_width, mode= 'replicate')
            # D_Deltas_tensor = torchaudio.functional.compute_deltas(Deltas_tensor, win_length=delta_width, mode= 'replicate')
            # MFCCs = np.squeeze(np.array(torch.Tensor.tolist(MFCCs_tensor)))
            # Deltas = np.squeeze(np.array(torch.Tensor.tolist(Deltas_tensor)))
            # D_Deltas = np.squeeze(np.array(torch.Tensor.tolist(D_Deltas_tensor)))
            L = MFCCs.shape[1]
            if L < min_MFCC_len:
                min_MFCC_len = L
                # print("Min Length = ", min_len)
            if L > max_MFCC_len:
                max_MFCC_len = L
                # print("Min Length = ", min_len)
            # Find the matching MOS
            for nc in range(len(MeanOS)):
                if Names[nc] in test_files[f]:
                    # print("Test_files", test_files[f], "Matlab name: ", MATLAB_NAME[nc], "MeanOS", MeanOS[nc])
                    break
            #Add to list of dictionaries
            test_list.append({
                "file": test_files[f],
                "MATLAB_loc": MATLAB_NAME[nc],
                "MeanOS": MeanOS[nc],
                "MedianOS": MedianOS[nc],
                "MFCCs": MFCCs,
                "Deltas": Deltas,
                "Deltas_Short": Deltas_Short,
                "D_Deltas": D_Deltas,
                "L": MFCCs.shape[1],
                "fnum": f})



        if norm_method == 1:
            # Frequency dimension normalisation
            # Normalise the MFCCs and Deltas
            for n in range(len(test_list)):
                # norm = (feature-feature_mean)/feature_std #Readable explanation
                test_list[n]["MFCCs_Norm"] = (test_list[n].get('MFCCs')-np.broadcast_to(MFCC_mean,(test_list[n].get('MFCCs').shape)))/np.broadcast_to(MFCC_std,(test_list[n].get('MFCCs').shape))
                test_list[n]["Deltas_Norm"] = (test_list[n].get('Deltas')-np.broadcast_to(Deltas_mean,(test_list[n].get('Deltas').shape)))/np.broadcast_to(Deltas_std,(test_list[n].get('Deltas').shape))
        elif norm_method == 2:
            # Overall normalisation
            # Normalise the MFCCs and Deltas
            for n in range(len(test_list)):
                # norm = (feature-feature_mean)/feature_std #Readable explanation
                test_list[n]["MFCCs_Norm"] = np.divide(np.subtract(test_list[n].get('MFCCs'),MFCC_mean),MFCC_std)
                test_list[n]["Deltas_Norm"] = np.divide(np.subtract(test_list[n].get('Deltas'),Deltas_mean),Deltas_std)
        else:
            # No Normalisation
            MFCC_mean = 0
            MFCC_std = 1
            Deltas_mean = 0
            Deltas_std = 1
            D_Deltas_mean = 0
            D_Deltas_std = 1
            # for n in range(len(train_list)):
            #     train_list[n]["MFCCs_Norm"] = train_list[n].get('MFCCs')
            #     train_list[n]["Deltas_Norm"] = train_list[n].get('Deltas')
            Norm_vals = np.array([MFCC_mean, MFCC_std, Deltas_mean, Deltas_std, D_Deltas_mean, D_Deltas_std])

        # Save Testing Data
        print("Saving Test data")
        if not os.path.exists(data_path): os.makedirs(data_path)
        with open(data_path + '/Feat_Test.p', 'wb') as f:
            pickle.dump(test_list, f)

    # Save out the train and test data
    print("Saving everything together")
    if not os.path.exists(data_path): os.makedirs(data_path)
    with open(data_path + '/Feat.p', 'wb') as f:
        pickle.dump((train_list, val_list, test_list, Norm_vals), f)


source_path = '../ML/data/audio/ref_train/'

#Create list of files in an arbitary order
source_files = [f for f in listdir(source_path) if isfile(join(source_path, f))]
source_list = []
if os.path.exists(data_path + '/Feat_Source.p'):
    with open(data_path + '/Feat_Source.p', 'rb') as f:
        source_list  = pickle.load(f)
else:
    #Source Files
    print("Creating Source data")
    for f in range(len(source_files)):
        print("Reading file: ", f, " ", source_path+source_files[f])
        x, fs = librosa.load(source_path+source_files[f], sr=None, mono=0)
        x = prep(x)
        MFCCs = librosa.feature.mfcc(x, sr=fs, n_mfcc=num_mfccs)
        Deltas = librosa.feature.delta(MFCCs,width=delta_width, order=1)
        Deltas_Short = librosa.feature.delta(MFCCs,width=delta_short, order=1)
        D_Deltas = librosa.feature.delta(Deltas,width=delta_width, order=1)
        # # TorchAudio Features
        # MFCCs_tensor = MFCC_Calc(torch.Tensor(x))
        # Deltas_tensor = torchaudio.functional.compute_deltas(MFCCs_tensor, win_length=delta_width, mode= 'replicate')
        # D_Deltas_tensor = torchaudio.functional.compute_deltas(Deltas_tensor, win_length=delta_width, mode= 'replicate')
        # MFCCs = np.squeeze(np.array(torch.Tensor.tolist(MFCCs_tensor)))
        # Deltas = np.squeeze(np.array(torch.Tensor.tolist(Deltas_tensor)))
        # D_Deltas = np.squeeze(np.array(torch.Tensor.tolist(D_Deltas_tensor)))
        L = MFCCs.shape[1]
        if L < min_MFCC_len:
            min_MFCC_len = L
            # print("Min Length = ", min_len)
        if L > max_MFCC_len:
            max_MFCC_len = L
        # print("Min Length = ", min_len)

        #Add to list of dictionaries
        source_list.append({
            "file": source_files[f],
            "MATLAB_loc": source_path+source_files[f],
            "MeanOS": 5,
            "MedianOS": 5,
            "MFCCs": MFCCs,
            "Deltas": Deltas,
            "Deltas_Short": Deltas_Short,
            "D_Deltas": D_Deltas,
            "L": MFCCs.shape[1],
            "fnum": f})


    if norm_method == 1:
        # Frequency dimension normalisation
        # Normalise the MFCCs and Deltas
        for n in range(len(source_list)):
            # norm = (feature-feature_mean)/feature_std #Readable explanation
            source_list[n]["MFCCs_Norm"] = (source_list[n].get('MFCCs')-np.broadcast_to(MFCC_mean,(source_list[n].get('MFCCs').shape)))/np.broadcast_to(MFCC_std,(source_list[n].get('MFCCs').shape))
            source_list[n]["Deltas_Norm"] = (source_list[n].get('Deltas')-np.broadcast_to(Deltas_mean,(source_list[n].get('Deltas').shape)))/np.broadcast_to(Deltas_std,(source_list[n].get('Deltas').shape))
    elif norm_method == 2:
        # Overall normalisation
        # Normalise the MFCCs and Deltas
        for n in range(len(source_list)):
            # norm = (feature-feature_mean)/feature_std #Readable explanation
            source_list[n]["MFCCs_Norm"] = np.divide(np.subtract(source_list[n].get('MFCCs'),MFCC_mean),MFCC_std)
            source_list[n]["Deltas_Norm"] = np.divide(np.subtract(source_list[n].get('Deltas'),Deltas_mean),Deltas_std)
    else:
        # No Normalisation
        # for n in range(len(source_list)):
            # source_list[n]["MFCCs_Norm"] = source_list[n].get('MFCCs')
            # source_list[n]["Deltas_Norm"] = source_list[n].get('Deltas')
        MFCC_mean = 0
        MFCC_std = 1
        Deltas_mean = 0
        Deltas_std = 1
        D_Deltas_mean = 0
        D_Deltas_std = 1
        Norm_vals = np.array([MFCC_mean, MFCC_std, Deltas_mean, Deltas_std, D_Deltas_mean, D_Deltas_std])

    # Save Source Data
    print("Saving Source data")
    if not os.path.exists(data_path): os.makedirs(data_path)
    with open(data_path + '/Feat_Source.p', 'wb') as f:
        pickle.dump(source_list, f)


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
print("Min MFCC Length = ", min_MFCC_len)
print("Max MFCC Length = ", max_MFCC_len)
