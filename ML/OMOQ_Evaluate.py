# Initial testing with the OMOQ dataset
# Built based on https://adventuresinmachinelearning.com/pytorch-tutorial-deep-learning/
import scipy.io as sio

import numpy as np
import torch, sys, os, datetime, smtplib, ssl, pickle, random, csv
import matplotlib.pyplot as plt
from scipy import stats

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


testing_batch_size = 1
training_target = 0
feat_start = 5
csv_out_load_file = 'data/Features/MOVs_Eval_All_To_Test.csv' #This is only for file names and methods
load_file = 'data/Features/MOVs_Eval_All_To_Test.mat'

MODEL_PATH = './models/FCN/2020-04-16_20-08-31RMSE_TO_TEST_SOURCE_ADAMW/Models/OMOQ.pth'
Norm_Val_file = './models/FCN/2020-04-16_20-08-31RMSE_TO_TEST_SOURCE_ADAMW/Norm_Vals.p'

features = sio.loadmat(load_file)
fname = '_TO_TEST_INCL_Source_MeanOS'
chosen_features = np.arange(0,features['MOVs'].shape[1])   #New Phasiness Features
dropout_per=0
#
# # Make deterministic
# torch.manual_seed(my_seed)
# torch.cuda.manual_seed(my_seed)
# np.random.seed(my_seed)
# random.seed(my_seed)
# torch.backends.cudnn.deterministic=True
new_folder_name = 'log/Eval/'+str(datetime.datetime.now())[:19].replace(" ","_").replace(":","-")+fname
if not os.path.exists(new_folder_name): os.makedirs(new_folder_name) # create log directory.
# if not os.path.exists(new_folder_name+'/Models'): os.makedirs(new_folder_name+'/Models')
# PATH = new_folder_name+'/Models/OMOQ'


#Restrict features to chosen ones
features['MOVs'] = features['MOVs'][:,chosen_features]
num_files = features['MOVs'].shape[0]
# Normalise the input features

#Load mean, std, min and max values
with open(Norm_Val_file, 'rb') as f:
    temp_mean, temp_std, temp_min, temp_max = pickle.load(f)

# Normalise to zero mean and unity standard deviation
features_norm = (features['MOVs'][:,feat_start:]-temp_mean)/temp_std
# Rescale to training data 0-1
features_norm = (features_norm-temp_min)/(temp_max-temp_min)

#Limit to 0-1
features_norm[np.greater(features_norm,1)] = 1
features_norm[np.less(features_norm,0)] = 0

# Scale the labels to 0-1
Test_Targets = features['MOVs'][:,training_target]
Test_Targets = (Test_Targets-1)/4

# Create the input and target features.
Test_Input_MOVs = torch.from_numpy(features_norm[:,:]).float()  # All Test Set
Test_Target_MOVs = torch.from_numpy(Test_Targets).float()  # All Test Set

print('Defining the Fully Connected Network')
class Net(nn.Module):
    def __init__(self, nodes=128):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(np.ma.size(chosen_features)-feat_start, nodes)
        self.fc2 = nn.Linear(nodes, nodes)
        self.fc3 = nn.Linear(nodes, nodes)
        self.fc4 = nn.Linear(nodes, 1)
        self.LN1 = nn.LayerNorm(nodes)
        self.LN2 = nn.LayerNorm(nodes)
        self.LN3 = nn.LayerNorm(nodes)

    def forward(self, x, train):
        # Relu Activation
        DR = nn.Dropout(dropout_per, inplace=False)
        if train:
            x = DR(F.relu(self.LN1(self.fc1(x))))
            x = torch.add(DR(F.relu(self.LN2(self.fc2(x)))),x)
            x = torch.add(DR(F.relu(self.LN3(self.fc3(x)))),x)
        else: #Testing
            x = F.relu(self.LN1(self.fc1(x)))
            x = torch.add(F.relu(self.LN2(self.fc2(x))),x)
            x = torch.add(F.relu(self.LN3(self.fc3(x))),x)
        x = torch.sigmoid(self.fc4(x))
        return x

# PEAQ Original Network
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.fc1 = nn.Linear(np.ma.size(chosen_features)-4, 3)
#         self.fc4 = nn.Linear(3, 1)
#
#     def forward(self, x, train):
#         # Layer norm goes after layer output before activation function
#         # Additions are residual connections
#         # LN1 = nn.LayerNorm(self.fc1(x).size()[1:])
#         if train:
#             # x = DR(F.relu(LN1(self.fc1(x))))
#             x = torch.sigmoid(self.fc1(x))
#         else:
#             # x = F.relu(LN1(self.fc1(x)))
#             x = torch.sigmoid(self.fc1(x))
#         x = torch.sigmoid(self.fc4(x))
#         return x

#Create the network
net = Net()

# # Setup the optimizer
# print('Setup the optimizer')
# # optimizer = optim.Adam(net.parameters(), lr=learning_rate)
# criterion = nn.L1Loss() # Mean Absolute Error Loss
# loss_type = 'MAE'
# # criterion = nn.MSELoss() # Mean Squared Error Loss
# # loss_type = 'MSE'


# Setup the data for training
print('Setting up training data')
# Setup the data for testing
testing_dataset = TensorDataset(Test_Input_MOVs,Test_Target_MOVs)
test_loader = DataLoader(testing_dataset, testing_batch_size, shuffle=False, sampler=None, \
                                                batch_sampler=None, num_workers=0, collate_fn=None, \
                                                pin_memory=False, drop_last=False, timeout=0, \
                                                worker_init_fn=None, multiprocessing_context=None)



# Create vector to store loss values after each epoch
TEST_OMOQ_vals = np.zeros((num_files,1))
TEST_SMOQ_vals = np.zeros((num_files,1))


print('Evaluate the Features')
# Load the pretrained network
net.load_state_dict(torch.load(MODEL_PATH))
net.eval()
idx = 0
for data, target in test_loader:
    test_net_out = net(data, 0)
    # Store Objective and Subjective values
    TEST_OMOQ_vals[idx,0] = torch.Tensor.item(test_net_out.view(testing_batch_size))*4+1
    TEST_SMOQ_vals[idx,0] = torch.Tensor.item(target)*4+1
    idx+=1
    print('File: ',idx,' Evaluated')


# Write out the predicted values at the best epoch for the testing set.

with open(csv_out_load_file, 'r', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    flist = list(reader)

with open(new_folder_name + "/Eval.csv", "a") as results: results.write("Test File, Ref File, Method, TSM, OMOS\n")
with open(new_folder_name + "/Eval.csv", "a") as results:
    for n in range(num_files):
        results.write(flist[n]['test_file'])
        results.write(", ")
        results.write(flist[n]['ref_file'])
        results.write(", ")
        results.write(flist[n]['method'])
        results.write(", ")
        results.write(str(features['MOVs'][n,feat_start-1]))
        results.write(", ")
        results.write(str(TEST_OMOQ_vals[n,0])+ "\n")


print('Evaluation Complete')
