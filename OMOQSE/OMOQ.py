# OMOQ Classes and definitions
import torch
import numpy as np
import sys
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

def collate_simple(batch):
    # print(batch)

    Training_Features, MeanOS, L, fnum = zip(*batch)
    # print(len(Training_Features))
    # sys.exit()
    # Training_Features = [torch.FloatTensor(x) for x in Training_Features]
    Training_Features = torch.from_numpy(np.array(Training_Features)).float() #BxCxDxL
    # print(Training_Features.shape)
    # sys.exit()
    MeanOS = torch.Tensor(MeanOS)
    L = torch.Tensor(L)
    fnum = torch.Tensor(fnum)

    # data = torch.from_numpy(Training_Features).float()
    # target = torch.from_numpy(MeanOS).float()
    return Training_Features, MeanOS, L, fnum


def collate_fn_CNN_NChannels(batch):
    #Take 56 frames from a random location within the signal
    np.random.seed()
    # Min length found earlier
    min_len = 53 #MFCCs
    # min_len = 28 #MAG PHASE POWER
    # print("Batch: "+batch)
    data = np.zeros((len(batch), batch[0][0].shape[0], batch[0][0].shape[1], min_len))
    target = np.zeros((len(batch),1))
    for n in range(len(batch)):
        if batch[n][0].shape[2]-min_len > 0:
            start = np.random.randint(0, batch[n][0].shape[2]-min_len)
            data[n,:,:,:] = batch[n][0][:,:,start:start+min_len]
            target[n] = batch[n][1]
        elif batch[n][0].shape[2]-min_len == 0:
            start = 0
            data[n,:,:,:] = batch[n][0][:,:,start:start+min_len]
            target[n] = batch[n][1]
        else:
            start=0
            pad = np.abs(batch[n][0].shape[2]-min_len)
            data[n,:,:,:] = np.concatenate((np.zeros((batch[0][0].shape[0],batch[0][0].shape[1],pad)),batch[n][0][:,:,:]),axis=2)
            target[n] = batch[n][1]

    data = torch.from_numpy(data).float()
    target = torch.from_numpy(target).float()
    return data, target

def collate_fn_2Channels_frame_target(batch):
    np.random.seed()
    # Max and min length found earlier
    # max_len = 0
    min_len = 56
    # Run this section to get the min and max lengths.  Will be useful for padding
    # instead of truncating
    # print(batch[0][0].shape)
    # for n in range(len(batch)):
    #     # if  batch[n][0].shape[2] > max_len:
    #     #     max_len = batch[n][0].shape[2]
    #     #     print("Max Len Now: ", max_len)
    #     if batch[n][0].shape[2] < min_len:
    #         min_len = batch[n][0].shape[2]
    #         print("Min Len Now: ", min_len)

    # sys.exit()
    data = np.zeros((len(batch), batch[0][0].shape[0], batch[0][0].shape[1], min_len))
    # print("Data Shape: ", data.shape)
    target = np.zeros((len(batch),min_len))
    for n in range(len(batch)):
        #extract 30 frames from each file
        if batch[n][0].shape[2]-min_len > 0:
            start = np.random.randint(0, batch[n][0].shape[2]-min_len)
        else:
            start = 0
        # print("Start: ", start, "Batch: ", batch[n][0].shape)
        # print(batch[n][0][:][:,:,start:start+min_len])
        data[n,:,:,:] = batch[n][0][:,:,start:start+min_len]
        target[n,:] = batch[n][1]*np.ones((1,min_len))
        # sys.exit()
        # print("Target = ", target)

    # data = torch.unsqueeze(torch.from_numpy(data).float(),1)
    data = torch.from_numpy(data).float()
    target = torch.from_numpy(target).float()
    # print("Data Shape: ", data.shape)
    # print("Data: ", data)
    # print("Target: ", target)
    return data, target

    # lengths = [batch[t][0].shape[1] for t in batch] #Number of rows
    # print(lengths)
    # for n in range(len(batch)):


def pad_collate_fn(data):
    """
    collate function from Jaspreet
    """
    # print(data)
    # sys.exit()
    # sort data by sequence length
    data.sort(key=lambda x: x[2], reverse=True)

    Training_Features, MeanOS, L, fnum = zip(*data)

    Training_Features = [torch.FloatTensor(x) for x in Training_Features]
    MeanOS = torch.Tensor(MeanOS)
    L = torch.Tensor(L)
    fnum = torch.Tensor(fnum)
    # print("collate fn: " + str(Training_Features[0].shape)) # LxD
    # Pad features
    Training_Features = nn.utils.rnn.pad_sequence(Training_Features, batch_first=True, padding_value=0)



    return Training_Features, MeanOS, L, fnum ### also return feats_lengths and label_lengths if using packpadd


# Dataset class
class OMOQDatasetCNN(Dataset):
    """Create Dataset for OMOQ
    Min MFCC Length =  53
    Max MFCC Length =  2179
    Min POW Length =  28
    Max POW Length =  1091"""

    def __init__(self, data, transform=None):
        """
        Args:
            data (list of dictionaries): The saved dataset
                MFCC Delta and Delta Delta features
                {"file": train_files[rand_order[f]],
                "MATLAB_loc": MATLAB_NAME[nc],
                "MeanOS": MeanOS[nc],
                "MedianOS": MedianOS[nc],
                "MFCCs": MFCCs,
                "Deltas": Deltas,
                "D_Deltas": D_Deltas,
                "L": MFCCs.shape[1],
                "fnum": f}

                Magnitude, Phase and Power features
                {"file": train_files[rand_order[f]],
                "MATLAB_loc": MATLAB_NAME[nc],
                "MeanOS": MeanOS[nc],
                "MedianOS": MedianOS[nc],
                "fnum": f,
                "fs": fs,
                "X_POW": X_POW,
                "X_MAG": X_MAG,
                "X_PHASE": X_PHASE,
                "X_len": X_len}


            transform (callable, optional): Optional transforms on the samples

        """
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        filename = self.data[idx].get('file')
        MeanOS = (self.data[idx].get('MeanOS')-1)/4
        MedianOS = (self.data[idx].get('MedianOS')-1)/4
        try: #Training data
            MFCCs = self.data[idx].get('MFCCs')
            Deltas = self.data[idx].get('Deltas')
            # D_Deltas = self.data[idx].get('D_Deltas')
            # POW = self.data[idx].get('X_POW')
            # MAG = self.data[idx].get('X_MAG')
            # PHASE = self.data[idx].get('X_PHASE')
        except:
            #Setting to zeros to allow show when it's not using normalised input features
            MFCCs = np.zeros((1,self.data[idx].get('MFCCs').shape[0],self.data[idx].get('MFCCs').shape[1]))
            Deltas = np.zeros((1,self.data[idx].get('Deltas').shape[0],self.data[idx].get('Deltas').shape[1]))
            # D_Deltas = np.zeros((1,self.data[idx].get('D_Deltas').shape[0],self.data[idx].get('D_Deltas').shape[1]))
            # POW = np.zeros((1,self.data[idx].get('X_POW').shape[0],self.data[idx].get('X_POW').shape[1]))
            # MAG = np.zeros((1,self.data[idx].get('X_MAG').shape[0],self.data[idx].get('X_MAG').shape[1]))
            # PHASE = np.zeros((1,self.data[idx].get('X_PHASE').shape[0],self.data[idx].get('X_PHASE').shape[1]))
            print("Pre-Normalised data used")
        # # MFCC Pre init
        # Training_Features = np.zeros((1,MFCCs.shape[0],MFCCs.shape[1]))
        Training_Features = np.zeros((2,MFCCs.shape[0],MFCCs.shape[1]))
        # Training_Features = np.zeros((3,MFCCs.shape[0],MFCCs.shape[1]))
        # # Magnitude Phase Power Pre init
        # Training_Features = np.zeros((1,POW.shape[0],POW.shape[1]))
        # Training_Features = np.zeros((1,MAG.shape[0],MAG.shape[1]))
        # Training_Features = np.zeros((2,MAG.shape[0],MAG.shape[1]))
        # Training_Features = np.zeros((1,MFCCs.shape[0]-2,MFCCs.shape[1])) #Not sure what this one was.
        #Use these lines to change between MFCCs and other input features
        Training_Features[0,:,:] = MFCCs[:,:]
        Training_Features[1,:,:] = Deltas[:,:]
        # Training_Features[2,:,:] = D_Deltas[:,:]
        # Training_Features[0,:,:] = POW[:,:]
        # Training_Features[0,:,:] = MAG[:,:]
        # Training_Features[1,:,:] = PHASE[:,:]

        if self.transform:
            X_POW = self.transform(X_POW)

        return Training_Features, MeanOS

# Dataset class
class OMOQDatasetLSTM(Dataset):
    """Create Dataset for OMOQ
    Min MFCC Length =  53
    Max MFCC Length =  2179
    Min POW Length =  28
    Max POW Length =  1091"""

    def __init__(self, data, transform=None):
        """
        Args:
            data (list of dictionaries): The saved dataset
            MFCC Delta and Delta Delta features
            {"file": train_files[rand_order[f]],
            "MATLAB_loc": MATLAB_NAME[nc],
            "MeanOS": MeanOS[nc],
            "MedianOS": MedianOS[nc],
            "MFCCs": MFCCs,
            "Deltas": Deltas,
            "D_Deltas": D_Deltas,
            "L": MFCCs.shape[1],
            "fnum": f}

            Magnitude, Phase and Power features
            {"file": train_files[rand_order[f]],
            "MATLAB_loc": MATLAB_NAME[nc],
            "MeanOS": MeanOS[nc],
            "MedianOS": MedianOS[nc],
            "fnum": f,
            "fs": fs,
            "X_POW": X_POW,
            "X_MAG": X_MAG,
            "X_PHASE": X_PHASE,
            "X_len": X_len}
            transform (callable, optional): Optional transforms on the samples

        """
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # print(self.data[idx].get('X_POW').shape)
        # sys.exit()
        filename = self.data[idx].get('MATLAB_loc')
        MeanOS = (self.data[idx].get('MeanOS')-1)/4
        # print(MeanOS)
        MedianOS = (self.data[idx].get('MedianOS')-1)/4
        fnum = self.data[idx].get('fnum')
        # print(fnum)
        try: #Training data
            # MFCCs = self.data[idx].get('MFCCs')
            # Deltas = self.data[idx].get('Deltas')
            # D_Deltas = self.data[idx].get('D_Deltas')
            # L = self.data[idx].get('L')
            X_POW = self.data[idx].get('X_POW')
            L = self.data[idx].get('X_len')
        except:
            #Setting to zeros to allow show when it's not using normalised input features
            # MFCCs = np.zeros((1,self.data[idx].get('MFCCs').shape[0],self.data[idx].get('MFCCs').shape[1]))
            # Deltas = np.zeros((1,self.data[idx].get('Deltas').shape[0],self.data[idx].get('Deltas').shape[1]))
            # D_Deltas = np.zeros((1,self.data[idx].get('D_Deltas').shape[0],self.data[idx].get('D_Deltas').shape[1]))
            L = 0
            X_POW = np.zeros((1,self.data[idx].get('X_POW').shape[0],self.data[idx].get('X_POW').shape[1]))
            print("Pre-Normalised data used")
        # Training_Features = np.zeros((MFCCs.shape[0]*3,MFCCs.shape[1]))
        Training_Features = np.zeros((X_POW.shape[0],X_POW.shape[1]))
        # Training_Features = MFCCs[:,:].reshape(MFCCs.shape[1],MFCCs.shape[0])
        # print(Training_Features.shape)

        # Training_Features[0:128,:] = MFCCs[:,:]
        # Training_Features[128:256,:] = Deltas[:,:]
        # Training_Features[256:384,:] = D_Deltas[:,:]
        Training_Features[:,:] = X_POW[:,:]
        # print(Training_Features.shape)
        # sys.exit()
        Training_Features = np.transpose(Training_Features)
        # print(Training_Features.shape)
        # sys.exit()

        if self.transform:
            X_POW = self.transform(X_POW)

        return Training_Features, MeanOS, L, fnum

# Dataset class
class OMOQDatasetGRU(Dataset):
    """Create Dataset for OMOQ
    Min MFCC Length =  53
    Max MFCC Length =  2179
    Min POW Length =  28
    Max POW Length =  1091"""

    def __init__(self, data, transform=None):
        """
        Args:
            data (list of dictionaries): The saved dataset
            MFCC Delta and Delta Delta features
            {"file": train_files[rand_order[f]],
            "MATLAB_loc": MATLAB_NAME[nc],
            "MeanOS": MeanOS[nc],
            "MedianOS": MedianOS[nc],
            "MFCCs": MFCCs,
            "Deltas": Deltas,
            "D_Deltas": D_Deltas,
            "L": MFCCs.shape[1],
            "fnum": f}

            Magnitude, Phase and Power features
            {"file": train_files[rand_order[f]],
            "MATLAB_loc": MATLAB_NAME[nc],
            "MeanOS": MeanOS[nc],
            "MedianOS": MedianOS[nc],
            "fnum": f,
            "fs": fs,
            "X_POW": X_POW,
            "X_MAG": X_MAG,
            "X_PHASE": X_PHASE,
            "X_len": X_len}
            transform (callable, optional): Optional transforms on the samples

        """
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        filename = self.data[idx].get('MATLAB_loc')
        MeanOS = (self.data[idx].get('MeanOS')-1)/4
        # print(MeanOS)
        MedianOS = (self.data[idx].get('MedianOS')-1)/4
        fnum = self.data[idx].get('fnum')
        # print(fnum)
        try: #Training data
            MFCCs = self.data[idx].get('MFCCs')
            Deltas = self.data[idx].get('Deltas')
            # D_Deltas = self.data[idx].get('D_Deltas')
            L = self.data[idx].get('L')
            # X_MAG = self.data[idx].get('X_MAG')
            # X_PHASE = self.data[idx].get('X_PHASE')
            # X_POW = self.data[idx].get('X_POW')
            # L = self.data[idx].get('X_len')
        except:
            #Setting to zeros to allow show when it's not using normalised input features
            MFCCs = np.zeros((1,self.data[idx].get('MFCCs').shape[0],self.data[idx].get('MFCCs').shape[1]))
            Deltas = np.zeros((1,self.data[idx].get('Deltas').shape[0],self.data[idx].get('Deltas').shape[1]))
            # D_Deltas = np.zeros((1,self.data[idx].get('D_Deltas').shape[0],self.data[idx].get('D_Deltas').shape[1]))
            # L = 0
            # X_MAG = np.zeros((1,self.data[idx].get('X_MAG').shape[0],self.data[idx].get('X_MAG').shape[1]))
            # X_PHASE = np.zeros((1,self.data[idx].get('X_PHASE').shape[0],self.data[idx].get('X_PHASE').shape[1]))
            # X_POW = np.zeros((1,self.data[idx].get('X_POW').shape[0],self.data[idx].get('X_POW').shape[1]))
            L = 0
            print("Pre-Normalised data used")
        Training_Features = np.zeros((MFCCs.shape[0]*2,MFCCs.shape[1]))
        # Training_Features = np.zeros((X_MAG.shape[0]*1,X_MAG.shape[1]))
        # print("MFCCs shape: " + str(MFCCs.shape[0]))
        Training_Features[0:MFCCs.shape[0],:] = MFCCs[:,:]
        Training_Features[MFCCs.shape[0]:MFCCs.shape[0]*2,:] = Deltas[:,:]
        # Training_Features[MFCCs.shape[0]*2:MFCCs.shape[0]*3,:] = D_Deltas[:,:]
        # Training_Features[0:X_MAG.shape[0],:] = X_MAG[:,:]
        # Training_Features[X_MAG.shape[0]:X_MAG.shape[0]*2,:] = X_PHASE[:,:]
        # Training_Features[X_MAG.shape[0]*2:X_MAG.shape[0]*3,:] = X_POW[:,:]
        # print(Training_Features.shape)
        # sys.exit()
        Training_Features = np.transpose(Training_Features)
        # print(Training_Features.shape)
        # sys.exit()

        if self.transform:
            X_POW = self.transform(X_POW)

        return Training_Features, MeanOS, L, fnum

# Dataset class
class OMOQDatasetGRU_Trunc(Dataset):
    """Create Dataset for OMOQ
    Minimum MFCC Length: 47
    Maximum MFCC Length: 2046
    """

    def __init__(self, data, transform=None):
        """
        Args:
            data (list of dictionaries): The saved dataset
            MFCC Delta and Delta Delta features
            {"file": train_files[rand_order[f]],
            "MATLAB_loc": MATLAB_NAME[nc],
            "MeanOS": MeanOS[nc],
            "MedianOS": MedianOS[nc],
            "MFCCs": MFCCs,
            "Deltas": Deltas,
            "D_Deltas": D_Deltas,
            "L": MFCCs.shape[1],
            "fnum": f}

            transform (callable, optional): Optional transforms on the samples

        """
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        filename = self.data[idx].get('MATLAB_loc')
        MeanOS = (self.data[idx].get('MeanOS')-1)/4
        # print(MeanOS)
        MedianOS = (self.data[idx].get('MedianOS')-1)/4
        fnum = self.data[idx].get('fnum')
        # print(fnum)
        try: #Training data
            MFCCs = self.data[idx].get('MFCCs')
            Deltas = self.data[idx].get('Deltas')
            # Deltas_Short = self.data[idx].get('Deltas_Short')
            # D_Deltas = self.data[idx].get('D_Deltas')
            L = self.data[idx].get('L')

        except:
            #Setting to zeros to allow show when it's not using normalised input features
            MFCCs = np.zeros((1,self.data[idx].get('MFCCs').shape[0],self.data[idx].get('MFCCs').shape[1]))
            Deltas = np.zeros((1,self.data[idx].get('Deltas').shape[0],self.data[idx].get('Deltas').shape[1]))
            # Deltas_Short = np.zeros((1,self.data[idx].get('Deltas_Short').shape[0],self.data[idx].get('Deltas_Short').shape[1]))
            # D_Deltas = np.zeros((1,self.data[idx].get('D_Deltas').shape[0],self.data[idx].get('D_Deltas').shape[1]))
            L = 0
            print("Pre-Normalised data used")
        min_len = 47

        if L-min_len > 0:
            start = np.random.randint(0, L-min_len)
        elif L-min_len == 0:
            start = 0
        else:
            print('L-min_len < 0')
        Training_Features = np.zeros((MFCCs.shape[0]*2,min_len))
        Training_Features[0:MFCCs.shape[0],:] = MFCCs[:,start:start+min_len]
        Training_Features[MFCCs.shape[0]:MFCCs.shape[0]*2,:] = Deltas[:,start:start+min_len]
        #     Training_Features[MFCCs.shape[0]*2:MFCCs.shape[0]*3,:] = D_Deltas[:,start:start+min_len]
        L = min_len
        Training_Features = np.transpose(Training_Features)


        if self.transform:
            X_POW = self.transform(X_POW)

        return Training_Features, MeanOS, L, fnum

class OMOQDatasetGRU_Dup(Dataset):
    """Create Dataset for OMOQ
    Minimum MFCC Length: 47
    Maximum MFCC Length: 2046
    """

    def __init__(self, data, transform=None):
        """
        Args:
            data (list of dictionaries): The saved dataset
            MFCC Delta and Delta Delta features
            {"file": train_files[rand_order[f]],
            "MATLAB_loc": MATLAB_NAME[nc],
            "MeanOS": MeanOS[nc],
            "MedianOS": MedianOS[nc],
            "MFCCs": MFCCs,
            "Deltas": Deltas,
            "D_Deltas": D_Deltas,
            "L": MFCCs.shape[1],
            "fnum": f}

            transform (callable, optional): Optional transforms on the samples

        """
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        filename = self.data[idx].get('MATLAB_loc')
        MeanOS = (self.data[idx].get('MeanOS')-1)/4
        # print(MeanOS)
        MedianOS = (self.data[idx].get('MedianOS')-1)/4
        fnum = self.data[idx].get('fnum')
        # print(fnum)
        try: #Training data
            MFCCs = self.data[idx].get('MFCCs')
            Deltas = self.data[idx].get('Deltas')
            # Deltas_Short = self.data[idx].get('Deltas_Short')
            # D_Deltas = self.data[idx].get('D_Deltas')
            L = self.data[idx].get('L')

        except:
            #Setting to zeros to allow show when it's not using normalised input features
            MFCCs = np.zeros((1,self.data[idx].get('MFCCs').shape[0],self.data[idx].get('MFCCs').shape[1]))
            Deltas = np.zeros((1,self.data[idx].get('Deltas').shape[0],self.data[idx].get('Deltas').shape[1]))
            # Deltas_Short = np.zeros((1,self.data[idx].get('Deltas_Short').shape[0],self.data[idx].get('Deltas_Short').shape[1]))
            # D_Deltas = np.zeros((1,self.data[idx].get('D_Deltas').shape[0],self.data[idx].get('D_Deltas').shape[1]))
            L = 0
            print("Pre-Normalised data used")
        # max_len = 2046
        max_len = 1000
        Training_Features = np.zeros((MFCCs.shape[0]*2,max_len))
        in_p = 0
        while (in_p+L < max_len):
            #While more than L shorter than maximum length,
            #fill Training Features
            Training_Features[0:MFCCs.shape[0],in_p:in_p+L] = MFCCs[:,:]
            Training_Features[MFCCs.shape[0]:MFCCs.shape[0]*2,in_p:in_p+L] = Deltas[:,:]
            in_p = in_p+L
        diff = max_len-in_p
        Training_Features[0:MFCCs.shape[0],in_p:] = MFCCs[:,0:diff]
        Training_Features[MFCCs.shape[0]:MFCCs.shape[0]*2,in_p:] = Deltas[:,0:diff]
        #     Training_Features[MFCCs.shape[0]*2:MFCCs.shape[0]*3,:] = D_Deltas[:,start:start+min_len]
        L = max_len
        Training_Features = np.transpose(Training_Features)


        if self.transform:
            X_POW = self.transform(X_POW)

        return Training_Features, MeanOS, L, fnum

class OMOQDatasetCNN_Dup(Dataset):
    """Create Dataset for OMOQ
    Minimum MFCC Length: 47
    Maximum MFCC Length: 2046"""

    def __init__(self, data, transform=None):
        """
        Args:
            data (list of dictionaries): The saved dataset
                MFCC Delta and Delta Delta features
                {"file": train_files[rand_order[f]],
                "MATLAB_loc": MATLAB_NAME[nc],
                "MeanOS": MeanOS[nc],
                "MedianOS": MedianOS[nc],
                "MFCCs": MFCCs,
                "Deltas": Deltas,
                "Deltas_Short": Deltas_Short,
                "D_Deltas": D_Deltas,
                "L": MFCCs.shape[1],
                "fnum": f}

            transform (callable, optional): Optional transforms on the samples

        """
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        filename = self.data[idx].get('file')
        MeanOS = (self.data[idx].get('MeanOS')-1)/4
        MedianOS = (self.data[idx].get('MedianOS')-1)/4
        fnum = self.data[idx].get('fnum')
        try: #Training data
            MFCCs = self.data[idx].get('MFCCs')
            Deltas = self.data[idx].get('Deltas')
            Deltas_Short = self.data[idx].get('Deltas_Short')
            D_Deltas = self.data[idx].get('D_Deltas')
            L = self.data[idx].get('L')

        except:
            #Setting to zeros to allow show when it's not using normalised input features
            MFCCs = np.zeros((1,self.data[idx].get('MFCCs').shape[0],self.data[idx].get('MFCCs').shape[1]))
            Deltas = np.zeros((1,self.data[idx].get('Deltas').shape[0],self.data[idx].get('Deltas').shape[1]))
            Deltas_Short = np.zeros((1,self.data[idx].get('Deltas_Short').shape[0],self.data[idx].get('Deltas_Short').shape[1]))
            D_Deltas = np.zeros((1,self.data[idx].get('D_Deltas').shape[0],self.data[idx].get('D_Deltas').shape[1]))
            L=0
            print("Pre-Normalised data used")
        # # MFCC Pre init
        max_len = 500
        Training_Features = np.zeros((2,MFCCs.shape[0],max_len)) #BxDxL
        in_p = 0
        while (in_p+L < max_len):
            #While more than L shorter than maximum length,
            #fill Training Features
            Training_Features[0,:,in_p:in_p+L] = MFCCs[:,:]
            Training_Features[1,:,in_p:in_p+L] = Deltas[:,:]
            in_p = in_p+L
        diff = max_len-in_p
        Training_Features[0,:,in_p:] = MFCCs[:,0:diff]
        Training_Features[1,:,in_p:] = Deltas[:,0:diff]
        L = max_len
        # print(Training_Features.shape)
        # sys.exit()
        if self.transform:
            X_POW = self.transform(X_POW)

        return Training_Features, MeanOS, L, fnum

def conv2d_out_width(W, Fw, P, Sw):
    # (W)idth
    # (F)ilter width
    # (P)adding
    # (S)tride
    return (W-Fw+2*P)/Sw+1

def conv2d_out_height(H, Fh, P, Sh):
    # (W)idth
    # (F)ilter width
    # (P)adding
    # (S)tride
    return (H-Fh+2*P)/Sh+1
