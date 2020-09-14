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
load_folder = "./data/Features/Eval_MFCC_Delta_Delta_NoNorm_Trim/"
load_file = "Feat_Eval.p"
# Load the dataset
print('Loading Dataset')
with open(load_folder+load_file, 'rb') as f:
    eval_list = pickle.load(f)

MODEL_PATH = './models/Saved/CNN/2020-08-23_15-11-52/OMOQ.pth' #Seed 6
# MODEL_PATH = './models/saved/CNN/2020-08-23_17-29-40/OMOQ.pth' #Seed 24

# PATH = new_folder_name+'/OMOQ.pth'
fname = "MFCC_Delta_2020-08-23_15-11-52"
new_folder_name = 'logs/Eval/'+str(datetime.datetime.now())[:19].replace(" ","_").replace(":","-")+fname
if not os.path.exists(new_folder_name): os.makedirs(new_folder_name) # create log directory.



# Setup the data for training
print('Setting up Evaluation data')
# Setup the data for testing
# print(eval_list[0])
# sys.exit()
eval_dataset = OMOQ.OMOQDatasetCNN(eval_list)
# testing_dataset = TensorDataset(torch.as_tensor(temp_data),torch.from_numpy(temp_mos))#,torch.from_numpy(temp_length))
eval_loader = DataLoader(eval_dataset, evalbs, shuffle=False, sampler=None, \
                                                    batch_sampler=None, num_workers=0, \
                                                    collate_fn=OMOQ.collate_fn_CNN_NChannels, \
                                                    pin_memory=False, drop_last=False, timeout=0, \
                                                    worker_init_fn=None, multiprocessing_context=None)


print('Defining the Convolutional Network')
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #Input Layer
        self.layer1conv = nn.Conv2d(2, 16, kernel_size=(5,5), stride=(1,1), \
                                padding=0, dilation=1, groups=1, \
                                bias=True, padding_mode='zeros')#1 is 2 for MFCCs
        self.layer1Norm = nn.BatchNorm2d(16)
        self.layer1relu = nn.ReLU()
        self.layer1pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))

        self.layer2conv = nn.Conv2d(16, 32, kernel_size=(3,3), stride=(1,1), \
                                    padding=0, dilation=1, groups=1, \
                                    bias=True, padding_mode='zeros')
        self.layer2Norm = nn.BatchNorm2d(32)
        self.layer2relu = nn.ReLU()
        self.layer2pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))

        self.layer3conv = nn.Conv2d(32, 64, kernel_size=(3,3), stride=(1,1), \
                                    padding=0, dilation=1, groups=1, \
                                    bias=True, padding_mode='zeros')
        self.layer3Norm = nn.BatchNorm2d(64)
        self.layer3relu = nn.ReLU()

        self.layer4conv = nn.Conv2d(64, 32, kernel_size=(3,3), stride=(1,1), \
                                    padding=0, dilation=1, groups=1, \
                                    bias=True, padding_mode='zeros')
        self.layer4Norm = nn.BatchNorm2d(32)
        self.layer4relu = nn.ReLU()

        self.drop_out = nn.Dropout(p=dropout_per)
        #MFCCs
        self.fc1 = nn.Linear(5824, 128) #3584 for 80 MFCCs, 5824 for 128 MFCCs
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 1)
        #POW Spectrum
        # self.fc1 = nn.Linear(8000, 128)
        # self.fc2 = nn.Linear(128, 128)
        # self.fc3 = nn.Linear(128, 128)
        # self.fc4 = nn.Linear(128, 1)


    def forward(self, x, train):
        # Layer norm goes after layer output before activation function
        # Additions are residual connections
        # print(x.shape)
        out = self.layer1conv(x)
        out = self.layer1relu(out)
        out = self.layer1Norm(out)
        out = self.layer1pool(out)
        # print(out.shape)
        out = self.layer2conv(out)
        out = self.layer2relu(out)
        out = self.layer2Norm(out)
        out = self.layer2pool(out)
        # print(out.shape)
        out = self.layer3conv(out)
        out = self.layer3relu(out)
        out = self.layer3Norm(out)
        # print(out.shape)
        out = self.layer4conv(out)
        out = self.layer4relu(out)
        out = self.layer4Norm(out)
        # print(out.shape)
        out = out.reshape(out.size(0), -1)
        if train:
            out = self.drop_out(out)

        LN1 = nn.LayerNorm(self.fc1(out).size()[1:]).to(device)
        out = F.relu(LN1(self.fc1(out)))
        LN2 = nn.LayerNorm(self.fc2(out).size()[1:]).to(device)
        out = torch.add(F.relu(LN2(self.fc2(out))),out)
        LN3 = nn.LayerNorm(self.fc3(out).size()[1:]).to(device)
        out = torch.add(F.relu(LN3(self.fc3(out))),out)
        out = torch.sigmoid(self.fc4(out))

        return out


#Create the network
model = Net()
model.load_state_dict(torch.load(MODEL_PATH))
model.to(device)
model.eval()


EVAL_OMOQ_vals = np.zeros((len(eval_list), 1))
EVAL_SMOQ_vals = np.zeros((len(eval_list), 1))
EVAL_OMOQ = np.zeros((len(eval_list), 1, eval_averaging))
EVAL_SMOQ = np.zeros((len(eval_list), 1, eval_averaging))
epoch=0 #Setting this, rather than removing indexing.
# e = list(enumerate(eval_list))
# print(e[0])
for n in range(0,eval_averaging):
    for e_idx, (data, target) in enumerate(eval_loader):
        print("Pass: " + str(n) +", Evaluating: "+str(e_idx)+"/"+str(len(eval_list)) +", File: "+str(eval_list[e_idx].get('file')).split('/')[-1])
        # print(data)
        eval_data = data.to(device)
        # test_target = target.to(device)
        eval_net_out = model(eval_data, 0)
        # Store Objective and Subjective values

        EVAL_OMOQ[e_idx*evalbs:e_idx*evalbs+evalbs,epoch,n] = (np.array(torch.Tensor.tolist(eval_net_out.view(evalbs)))*4+1).flatten()

#Average all of the OMOQ predictions
EVAL_OMOQ_vals[:,epoch] = np.mean(EVAL_OMOQ[:,epoch,:],1)


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
