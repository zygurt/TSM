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

num_mfccs = 80
delta_width = 9

norm_methods = ['None','Freq Dim','Overall']
norm_method = 1


data_path = './data/'
if os.path.exists(data_path + '/CNN_Feat.p'):
    with open(data_path + '/CNN_Feat.p', 'rb') as f:
        train_list, test_list, Norm_vals = pickle.load(f)
else:
    # Load the MOS data from matlab
    load_file = 'data/OMOQ_CNN_MOS.mat'
    features = sio.loadmat(load_file)
    Names = []
    MATLAB_NAME = []
    # print(features['Name'][1][0][0])
    for n in range(0,len(features['MeanOS'][0])):
        # print(features['Name'][n][0][0])
        temp = features['Name'][n][0][0].split('/')[-1]
        Names.append(temp)
        temp = features['Name'][n][0][0]
        MATLAB_NAME.append(temp)
    MeanOS = features['MeanOS'][0]
    MedianOS = features['MedianOS'][0]
    # print("Name = ", Names)
    # print("MeanOS = ", MeanOS)
    # print("MedianOS = ", MedianOS)
    # sys.exit()
    test_path = './data/audio/test/'
    train_path = './data/audio/train/'
    #Create list of files in an arbitary order
    test_files = [f for f in listdir(test_path) if isfile(join(test_path, f))]
    train_files = [f for f in listdir(train_path) if isfile(join(train_path, f))]
    train_list = []
    test_list = []
    N = 2048
    min_len = 2000
    max_len = 0
    num_norm_files = 1000
    rand_order = np.random.permutation(len(train_files))
    MFCC_vals = np.zeros((80,1))
    Deltas_vals = np.zeros((80,1))
    #Training files
    if os.path.exists(data_path + '/CNN_Feat_Train.p'):
        with open(data_path + '/CNN_Feat_Train.p', 'rb') as f:
            train_list  = pickle.load(f)
    else:
        print("Creating Training data")
        for f in range(len(train_files)):
            print("Reading file: ", f, " ", train_path+train_files[f])
            # fs, x = wavfile.read(train_path+train_files[f])
            x, fs = librosa.load(train_path+train_files[f], sr=None)
            MFCCs = librosa.feature.mfcc(x, sr=fs, n_mfcc=num_mfccs)
            Deltas = librosa.feature.delta(MFCCs,width=delta_width, order=1)
            L = MFCCs.shape[1]
            if L < min_len:
                min_len = L
                # print("Min Length = ", min_len)
            if L > max_len:
                max_len = L
            # print("MFCCs: ", MFCCs.shape)
            # print("Deltas: ", Deltas.shape)
            # F,T,X = signal.stft(x, fs, window='hann', nperseg=N, noverlap=N/2) #Returns length N/2+1
            # X_POW = np.square(abs(X))

            # Find the matching MOS
            for n in range(len(MeanOS)):
                if Names[n] in train_files[f]:
                    # print("Train_files", train_files[f], "Matlab name: ", MATLAB_NAME[n], "MeanOS", MeanOS[n])
                    # sys.exit()
                    break

            # plt.figure(figsize=(10, 4))
            # plt.imshow(MFCCs)
            # # plt.colorbar()
            # plt.title('MFCC')
            # plt.tight_layout()
            # plt.colorbar()
            # plt.savefig("MFCCsimshow.png",dpi=300,format='png')
            # sys.exit()

            #Add to list of dictionaries
            train_list.append({
                "file": train_files[f],
                "MATLAB_loc": MATLAB_NAME[n],
                "MeanOS": MeanOS[n],
                "MedianOS": MedianOS[n],
                #"x": x,
                "fs": fs,
                "MFCCs": MFCCs,
                "Deltas": Deltas,
                "MFCCs_Norm": 0,
                "Deltas_Norm": 0,
                # "F": F,
                # "T": T,
                "L": MFCCs.shape[1],
                #"X": X,
                # "X_POW": X_POW })
                })
            # print("MFCCs_Norm: ", train_list[f].get('MFCCs_Norm'))
            # print(train_list)
            # sys.exit()
            # train_list[0]["MFCCs_Norm"] = 1

            #Create the mean and std for input normalisation
            if rand_order[n]<num_norm_files:
                if MFCC_vals.shape[1] == 1:
                    MFCC_vals = MFCCs
                    Deltas_vals = Deltas
                else:
                    #Append the new data
                    MFCC_vals = np.concatenate((MFCC_vals, MFCCs), axis=1)
                    Deltas_vals = np.concatenate((Deltas_vals, Deltas), axis=1)


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
            for n in range(len(train_list)):
                train_list[n]["MFCCs_Norm"] = train_list[n].get('MFCCs')
                train_list[n]["Deltas_Norm"] = train_list[n].get('Deltas')
            Norm_vals = np.array([MFCC_mean, MFCC_std, Deltas_mean, Deltas_std])

        #Store for later use
        if not os.path.exists(data_path): os.makedirs(data_path)
        with open(data_path + '/CNN_Feat_Train_Norm_Vals.p', 'wb') as f:
            pickle.dump(Norm_vals, f)

        # Save Training data
        print("Saving Training data")
        with open(data_path + '/CNN_Feat_Train.p', 'wb') as f:
            pickle.dump(train_list, f)

    if os.path.exists(data_path + '/CNN_Feat_Test.p'):
        with open(data_path + '/CNN_Feat_Test.p', 'rb') as f:
            test_list  = pickle.load(f)
    else:
        #Testing Files
        print("Creating Test data")
        for f in range(len(test_files)):
            print("Reading file: ", f, " ", test_path+test_files[f])
            # fs, x = wavfile.read(test_path+test_files[f])
            x, fs = librosa.load(test_path+test_files[f], sr=None)
            MFCCs = librosa.feature.mfcc(x, sr=fs, n_mfcc=num_mfccs)
            Deltas = librosa.feature.delta(MFCCs,width=delta_width, order=1)
            L = MFCCs.shape[1]
            if L < min_len:
                min_len = L
            if L > max_len:
                max_len = L
                # print("Min Length = ", min_len)
            # F,T,X = signal.stft(x, fs, window='hann', nperseg=N, noverlap=N/2) #Returns length N/2+1
            # X_POW = np.square(abs(X))
            # Find the matching MOS
            for n in range(len(MeanOS)):
                if Names[n] in test_files[f]:
                    # print("Test_files", test_files[f], "Matlab name: ", MATLAB_NAME[n], "MeanOS", MeanOS[n])
                    break
            #Add to list of dictionaries
            test_list.append({
                "file": test_files[f],
                "MATLAB_loc": MATLAB_NAME[n],
                "MeanOS": MeanOS[n],
                "MedianOS": MedianOS[n],
                #"x": x,
                "fs": fs,
                "MFCCs": MFCCs,
                "Deltas": Deltas,
                "MFCCs_Norm": 0,
                "Deltas_Norm": 0,
                # "F": F,
                # "T": T,
                "L": MFCCs.shape[1],
                #"X": X,
                # "X_POW": X_POW })
                })


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
            for n in range(len(test_list)):
                test_list[n]["MFCCs_Norm"] = test_list[n].get('MFCCs')
                test_list[n]["Deltas_Norm"] = test_list[n].get('Deltas')
            Norm_vals = np.array([MFCC_mean, MFCC_std, Deltas_mean, Deltas_std])

        # Save Testing Data
        print("Saving Test data")
        if not os.path.exists(data_path): os.makedirs(data_path)
        with open(data_path + '/CNN_Feat_Test.p', 'wb') as f:
            pickle.dump(test_list, f)

    # Save out the train and test data
    print("Saving everything together")
    if not os.path.exists(data_path): os.makedirs(data_path)
    with open(data_path + '/CNN_Feat.p', 'wb') as f:
        pickle.dump((train_list, test_list, Norm_vals), f)


source_path = './data/audio/ref_train/'

#Create list of files in an arbitary order
source_files = [f for f in listdir(source_path) if isfile(join(source_path, f))]
source_list = []
if os.path.exists(data_path + '/CNN_Feat_Source.p'):
    with open(data_path + '/CNN_Feat_Source.p', 'rb') as f:
        source_list  = pickle.load(f)
else:
    #Source Files
    print("Creating Source data")
    for f in range(len(source_files)):
        print("Reading file: ", f, " ", source_path+source_files[f])
        x, fs = librosa.load(source_path+source_files[f], sr=None)
        MFCCs = librosa.feature.mfcc(x, sr=fs, n_mfcc=num_mfccs)
        Deltas = librosa.feature.delta(MFCCs,width=delta_width, order=1)
        L = MFCCs.shape[1]
        # print("Min Length = ", min_len)
        #Add to list of dictionaries
        source_list.append({
            "file": source_files[f],
            "MeanOS": float(5), #No testing was done, however the quality of scaling of these files is perfect by nature, therefore MOS=5
            "MedianOS": float(5),
            #"x": x,
            "fs": fs,
            "MFCCs": MFCCs,
            "Deltas": Deltas,
            "MFCCs_Norm": 0,
            "Deltas_Norm": 0,
            # "F": F,
            # "T": T,
            "L": MFCCs.shape[1],
            #"X": X,
            # "X_POW": X_POW })
            })


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
        for n in range(len(source_list)):
            source_list[n]["MFCCs_Norm"] = source_list[n].get('MFCCs')
            source_list[n]["Deltas_Norm"] = source_list[n].get('Deltas')
        Norm_vals = np.array([MFCC_mean, MFCC_std, Deltas_mean, Deltas_std])

    # Save Source Data
    print("Saving Source data")
    if not os.path.exists(data_path): os.makedirs(data_path)
    with open(data_path + '/CNN_Feat_Source.p', 'wb') as f:
        pickle.dump(source_list, f)


# #
# val_path = './data/audio/Validation/'
# #
# # #Create list of files in an arbitary order
# val_files = [f for f in listdir(val_path) if isfile(join(val_path, f))]
# #
# val_list = []
# if os.path.exists(data_path + '/CNN_Feat_Val.p'):
#     with open(data_path + '/CNN_Feat_Val.p', 'rb') as f:
#         val_list  = pickle.load(f)
# else:
#     #Testing Files
#     print("Creating Val data")
#     for f in range(len(val_files)):
#         print("Reading file: ", f, " ", val_path+val_files[f])
#         x, fs = librosa.load(val_path+val_files[f], sr=None)
#         MFCCs = librosa.feature.mfcc(x, sr=fs, n_mfcc=num_mfccs)
#         Deltas = librosa.feature.delta(MFCCs,width=delta_width, order=1)
#         L = MFCCs.shape[1]
#         # print("Min Length = ", min_len)
#         #Add to list of dictionaries
#         val_list.append({
#             "file": val_files[f],
#             "MeanOS": np.random.randint(1,5),
#             "MedianOS": np.random.randint(1,5),
#             #"x": x,
#             "fs": fs,
#             "MFCCs": MFCCs,
#             "Deltas": Deltas,
#             # "F": F,
#             # "T": T,
#             "L": MFCCs.shape[1],
#             #"X": X,
#             # "X_POW": X_POW })
#             })
#     # Save Testing Data
#     print("Saving Val data")
#     if not os.path.exists(data_path): os.makedirs(data_path)
#     with open(data_path + '/CNN_Feat_Val.p', 'wb') as f:
#         pickle.dump(val_list, f)


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
file1.write("Minimum Length: ")
file1.write(str(min_len))
file1.write("\n")
file1.write("Maximum Length: ")
file1.write(str(max_len))
file1.write("\n")

file1.close()
print("Complete")
print("Min Length = ", min_len)
print("Max Length = ", max_len)
