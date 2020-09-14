# Initial testing with the OMOQ dataset
# Built based on https://adventuresinmachinelearning.com/pytorch-tutorial-deep-learning/
import scipy.io as sio
import numpy as np
import torch, sys, os, datetime, random
import matplotlib.pyplot as plt
import pickle
import OMOQ
# import librosa
from tqdm import tqdm

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset

print("CUDA Available: ", torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

evalbs = 1
dropout_per = 0.0 #Tried 0, 0.1, 0.25 0.4 0.5
eval_averaging = 16 #Number of times to sample signal in testing.

#Choose Evaluation Features
# load_folder = "./data/Features/MagPhasePow/"
# load_folder = "./data/Features/Eval_MFCC_Delta_Delta_NoNorm_Trim/"
load_folder = "./data/Features/Eval_MFCC_Lib/"
load_file = "Feat_Eval.p"
# Load the dataset
print('Loading Dataset')
with open(load_folder+load_file, 'rb') as f:
    eval_list = pickle.load(f)

# MODEL_PATH = './models/GRU/2020-08-31_15-07-55/OMOQ.pth' #Seed 6
# MODEL_PATH = './models/GRU/2020-08-30_01-54-09/OMOQ.pth' #Seed 28
MODEL_PATH = './models/GRU/2020-08-31_19-48-48/OMOQ.pth' #Lib 128 New features

# PATH = new_folder_name+'/OMOQ.pth'
fname = "BGRU_FT_2020-08-31_19-48-48"
new_folder_name = 'logs/Eval/'+str(datetime.datetime.now())[:19].replace(" ","_").replace(":","-")+fname
if not os.path.exists(new_folder_name): os.makedirs(new_folder_name) # create log directory.



# Setup the data for training
print('Setting up Evaluation data')
# Setup the data for testing
# print(eval_list[0])
# sys.exit()
eval_dataset = OMOQ.OMOQDatasetGRU(eval_list)
# testing_dataset = TensorDataset(torch.as_tensor(temp_data),torch.from_numpy(temp_mos))#,torch.from_numpy(temp_length))
eval_loader = DataLoader(eval_dataset, evalbs, shuffle=False, sampler=None, \
                                                    batch_sampler=None, num_workers=0, \
                                                    collate_fn=OMOQ.pad_collate_fn, \
                                                    pin_memory=False, drop_last=False, timeout=0, \
                                                    worker_init_fn=None, multiprocessing_context=None)


print('Defining the GRU Network')
class Net(nn.Module):
    def __init__(self, hidden_dim=256, input_size=256, numb_layers=2, batch=1, dirs=2, drop_per=0.1):
        super(Net, self).__init__()
        """
        input_size = 256 for MFCCs and Deltas
        input_size = 2050 for MagPhase
        input:
        out:
        """
        self.batch_size = batch
        self.hidden_dim = hidden_dim
        self.h0 = torch.randn(numb_layers*dirs,self.batch_size,self.hidden_dim).to(device) #num_layers * num_directions, batch, hidden_size
        self.gru1 = nn.GRU(input_size=input_size, hidden_size=self.hidden_dim, num_layers=numb_layers, batch_first=True,
                             bidirectional=(dirs==2), dropout=drop_per) # seq_len, batch, input_size
        self.conv1d1 = nn.Conv1d(in_channels=hidden_dim*dirs, out_channels=1, kernel_size=1, stride=1, padding=0, dilation=1,
                             groups=1, bias=True, padding_mode='zeros') #N (Batch),Cin (Channels/FeatDepth), Lin (Length)
        # try:
        #     #Some networks were trained with these unused definitions
        #     self.linear1 = nn.Linear(in_features=hidden_dim*2, out_features=256)
        #     self.linear2 = nn.Linear(in_features=256, out_features=128)
        #
        #     self.linear3 = nn.Linear(in_features=128, out_features=1)
        #     # if REGRESSION:
        #     self.LN1 = nn.LayerNorm(256)
        #     self.LN2 = nn.LayerNorm(128)
        # except:
        #     print('No linear weights in saved dictionary. This is normal.')
        # if REGRESSION:
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, seq_lens):
        #Recurrent Section
        x = nn.utils.rnn.pack_padded_sequence(x, seq_lens, batch_first=True) # Batch, Len, Dimension
        x, h = self.gru1(x,self.h0) # , batch, seq_len, input_size
        x, y = nn.utils.rnn.pad_packed_sequence(x, batch_first=True, padding_value=0)
        #Fully connected section
        x = x.permute(0,2,1)
        x = self.conv1d1(x) # Takes B x D x L
        x = x.permute(0,2,1)
        x = self.sigmoid(x)

        return x


#Create the network
model = Net(batch=evalbs)
model.load_state_dict(torch.load(MODEL_PATH))
model.to(device)
model.eval()


EVAL_OMOQ_vals = np.zeros((len(eval_list), 1))
EVAL_SMOQ_vals = np.zeros((len(eval_list), 1))
# EVAL_OMOQ = np.zeros((len(eval_list), 1))
# EVAL_SMOQ = np.zeros((len(eval_list), 1))
epoch=0 #Setting this, rather than removing indexing.
# e = list(enumerate(eval_list))
# print(e[0])
print('Evaluating')
for e_idx, (data, target, L, fnum) in enumerate(tqdm(eval_loader)):
    # print("Evaluating: "+str(e_idx)+"/"+str(len(eval_list)) +", File: "+str(eval_list[e_idx].get('file')).split('/')[-1])
    # print(data)
    eval_data = data.to(device)
    eval_L = L.to(device)
    lens = torch.Tensor.tolist(L.long())
    # test_target = target.to(device)
    eval_net_out = model(eval_data, eval_L)
    # Store Objective and Subjective values
    for b in range(0,L.shape[0]):
        EVAL_OMOQ_vals[fnum[b].long(),epoch] = np.array(torch.Tensor.item(torch.mean(eval_net_out[b,0:lens[b],:])))*4+1

#Average all of the OMOQ predictions (Copied from CNN which did random selection of signal sections)
# EVAL_OMOQ_vals[:,epoch] = np.mean(EVAL_OMOQ[:,epoch,:],1)


print("Saving log of Eval results\n")
if not os.path.exists('log'): os.makedirs('log') # create log directory.
with open(new_folder_name + "/Eval.csv", "a") as results: results.write("Filename, TSM, OMOS\n")
with open(new_folder_name + "/Eval.csv", "a") as results:
    for n in range(len(eval_list)):
        # print(test_list[n].get('file'))
        results.write(str(eval_list[n].get('file')).split('/')[-1]) #File name
        results.write(", ")
        results.write(str(eval_list[n].get('file')).split('_')[-2]) #TSM Ratio
        results.write(", ")
        results.write(str(EVAL_OMOQ_vals[n,0])+ "\n")









#Comment to put the code at a more readable position after saving
