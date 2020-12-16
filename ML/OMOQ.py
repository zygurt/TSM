# OMOQ Classes and definitions
import torch
import numpy as np
import sys

from torch.utils.data import Dataset, DataLoader


from prettytable import PrettyTable

def count_parameters(model):
    # from https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

# collate_fn function

def collate_fn_truncate(batch):
    np.random.seed()
    # Max and min length found earlier
    max_len = 1091
    min_len = 29
    data = np.zeros((len(batch), batch[0][0].shape[0], min_len))
    target = np.zeros((len(batch)))
    for n in range(len(batch)):
        #extract 29 frames from each file
        if batch[n][0].shape[1]-min_len > 0:
            start = np.random.randint(0, batch[n][0].shape[1]-min_len)
        else:
            start = 0
        data[n][:][:] = batch[n][0][:,start:start+min_len]
        target[n] = batch[n][1]

    data = torch.unsqueeze(torch.from_numpy(data).float(),1)
    target = torch.from_numpy(target).float()
    return data, target

def collate_fn_CNN_2Channels(batch):
    np.random.seed()
    # Min length found earlier
    min_len = 56
    data = np.zeros((len(batch), batch[0][0].shape[0], batch[0][0].shape[1], min_len))
    target = np.zeros((len(batch),1))
    for n in range(len(batch)):
        if batch[n][0].shape[2]-min_len > 0:
            start = np.random.randint(0, batch[n][0].shape[2]-min_len)
        else:
            start = 0
        data[n,:,:,:] = batch[n][0][:,:,start:start+min_len]
        target[n] = batch[n][1]
    data = torch.from_numpy(data).float()
    target = torch.from_numpy(target).float()
    return data, target

def collate_fn_CNN_2Channels_end(batch):
    np.random.seed()
    # Min length found earlier
    min_len = 56
    data = np.zeros((len(batch), batch[0][0].shape[0], batch[0][0].shape[1], min_len))
    target = np.zeros((len(batch),1))
    for n in range(len(batch)):
        if batch[n][0].shape[2]-min_len > 0:
            # start = np.random.randint(0, batch[n][0].shape[2]-min_len)
            start = batch[n][0].shape[2]-min_len
        else:
            start = 0
        data[n,:,:,:] = batch[n][0][:,:,start:start+min_len]
        target[n] = batch[n][1]
    data = torch.from_numpy(data).float()
    target = torch.from_numpy(target).float()
    return data, target

def collate_fn_CNN_2Channels_start(batch):
    np.random.seed()
    # Min length found earlier
    min_len = 56
    data = np.zeros((len(batch), batch[0][0].shape[0], batch[0][0].shape[1], min_len))
    target = np.zeros((len(batch),1))
    for n in range(len(batch)):
        # if batch[n][0].shape[2]-min_len > 0:
        #     # start = np.random.randint(0, batch[n][0].shape[2]-min_len)
        #     start = batch[n][0].shape[2]-min_len
        # else:
        #     start = 0
        # data[n,:,:,:] = batch[n][0][:,:,start:start+min_len]
        data[n,:,:,:] = batch[n][0][:,:,0:min_len]
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

def collate_fn_CNN_2Channels_PAD(batch):
    # Min and max length found earlier in feature creation
    min_len = 56
    max_len = 2179
    data = np.zeros((len(batch), batch[0][0].shape[0], batch[0][0].shape[1], max_len))
    target = np.zeros((len(batch),1))
    for n in range(len(batch)):
        # if batch[n][0].shape[2]-min_len > 0:
        #     start = np.random.randint(0, batch[n][0].shape[2]-min_len)
        # else:
        #     start = 0
        data[n,:,:,0:batch[n][0].shape[2]] = batch[n][0][:,:,:]
        target[n] = batch[n][1]
        # print(data[n,:,:,:])
        # sys.exit()
    data = torch.from_numpy(data).float()
    target = torch.from_numpy(target).float()
    return data, target


# Dataset class
class OMOQDataset(Dataset):
    """Create Dataset for OMOQ"""

    def __init__(self, data, transform=None):
        """
        Args:
            data (list of dictionaries): The saved dataset
            {   "file": train_files[f],
                "MeanOS": MeanOS[n],
                "MedianOS": MedianOS[n],
                #"x": x,
                "fs": fs,
                "F": F,
                "T": T,
                "L": len(T),
                #"X": X,
                "X_POW": X_POW }
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
        MeanOS = self.data[idx].get('MeanOS')
        MedianOS = self.data[idx].get('MedianOS')
        # L = self.data[idx].get('L')
        # X_POW = self.data[idx].get('X_POW')
        # sample = self.data[idx]
        try: #Training data
            MFCCs = self.data[idx].get('MFCCs_Norm')
            Deltas = self.data[idx].get('Deltas_Norm')
        except:
            #Setting to zeros to allow show when it's not using normalised input features
            # MFCCs = self.data[idx].get('MFCCs')
            MFCCs = np.zeros((1,self.data[idx].get('MFCCs').shape[0],self.data[idx].get('MFCCs').shape[1]))
            # Deltas = self.data[idx].get('Deltas')
            Deltas = np.zeros((1,self.data[idx].get('Deltas').shape[0],self.data[idx].get('Deltas').shape[1]))
            print("Pre-Normalised data used")
        Training_Features = np.zeros((2,MFCCs.shape[0],MFCCs.shape[1]))
        # Training_Features = np.zeros((1,MFCCs.shape[0]-2,MFCCs.shape[1]))
        Training_Features[0,:,:] = MFCCs[:,:]
        Training_Features[1,:,:] = Deltas[:,:]

        if self.transform:
            X_POW = self.transform(X_POW)

        return Training_Features, MeanOS


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
