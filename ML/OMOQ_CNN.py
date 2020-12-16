# Initial testing with the OMOQ dataset
# Built based on https://adventuresinmachinelearning.com/pytorch-tutorial-deep-learning/
import scipy.io as sio
import numpy as np
import torch, sys, os, datetime
import matplotlib.pyplot as plt
import pickle
import OMOQ
# import librosa

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset

print("CUDA Available: ", torch.cuda.is_available())
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

verb = 0
learning_rate = 1e-4 #Have tried 1e-2 to 1e-6
epochs = 200
log_interval = 50
training_batch_size = 120  #factor of number of training files (Tried 1 to 5280)
testing_batch_size = 1
source_batch_size = 1
val_batch_size = 1
dropout_per = 0.1 #Tried 0, 0.1, 0.25 0.4 0.5
best_pcc = 0.5
best_mean_pcc = 0.5
test_averaging = 16

# Create folder for results
new_folder_name = 'plots/CNN/'+str(datetime.datetime.now())[:19].replace(" ","_").replace(":","-")
if not os.path.exists(new_folder_name): os.makedirs(new_folder_name) # create log directory.

PATH = new_folder_name+'/OMOQ.pth'
Mean_PATH = new_folder_name+'/MeanPCC_OMOQ.pth'
load_file = "CNN_Freq_Norm/CNN_Feat.p"
# Load the dataset
print('Loading Dataset')
with open("./data/"+load_file, 'rb') as f:
    train_list, test_list, Norm_vals= pickle.load(f)


# training_mean = np.mean(features['MOVs'][:,4:],0)
# training_std = np.std(features['MOVs'][:,4:],0)
# features_norm = (features['MOVs'][:,4:]-training_mean)/training_std


# Setup the data for training
print('Setting up testing data')
# Setup the data for testing
testing_dataset = OMOQ.OMOQDataset(test_list)
# testing_dataset = TensorDataset(torch.as_tensor(temp_data),torch.from_numpy(temp_mos))#,torch.from_numpy(temp_length))
test_loader = DataLoader(testing_dataset, testing_batch_size, shuffle=False, sampler=None, \
                                                    batch_sampler=None, num_workers=0, \
                                                    collate_fn=OMOQ.collate_fn_CNN_2Channels, \
                                                    pin_memory=False, drop_last=False, timeout=0, \
                                                    worker_init_fn=None, multiprocessing_context=None)
# # Test the DataLoader
# for batch_idx, (data, target) in enumerate(test_loader):
#     print(batch_idx)
#     print("Data = ", data,"Target = ", target)
#     sys.exit()


print('Setting up training data')
training_dataset = OMOQ.OMOQDataset(train_list)
train_loader = DataLoader(training_dataset, training_batch_size, shuffle=False, sampler=None, \
                                                    batch_sampler=None, num_workers=0, \
                                                    collate_fn=OMOQ.collate_fn_CNN_2Channels, \
                                                    pin_memory=False, drop_last=False, timeout=0, \
                                                    worker_init_fn=None, multiprocessing_context=None)

# print("Training Loader: ", train_loader)
# sys.exit()

print('Defining the Convolutional Network')
class Net(nn.Module):
    def __init__(self, KernSize=3):
        super(Net, self).__init__()
        #Input Layer
        self.layer1conv = nn.Conv2d(2, 16, kernel_size=(KernSize,KernSize), stride=(1,1), \
                                padding=0, dilation=1, groups=1, \
                                bias=True, padding_mode='zeros')
        self.layer1Norm = nn.BatchNorm2d(16)
        self.layer1relu = nn.ReLU()
        self.layer1pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))

        self.layer2conv = nn.Conv2d(16, 32, kernel_size=(KernSize,KernSize), stride=(1,1), \
                                    padding=0, dilation=1, groups=1, \
                                    bias=True, padding_mode='zeros')
        self.layer2Norm = nn.BatchNorm2d(32)
        self.layer2relu = nn.ReLU()
        self.layer2pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))

        self.layer3conv = nn.Conv2d(32, 64, kernel_size=(KernSize,KernSize), stride=(1,1), \
                                    padding=0, dilation=1, groups=1, \
                                    bias=True, padding_mode='zeros')
        self.layer3Norm = nn.BatchNorm2d(64)
        self.layer3relu = nn.ReLU()

        self.layer4conv = nn.Conv2d(64, 32, kernel_size=(KernSize,KernSize), stride=(1,1), \
                                    padding=0, dilation=1, groups=1, \
                                    bias=True, padding_mode='zeros')
        self.layer4Norm = nn.BatchNorm2d(32)
        self.layer4relu = nn.ReLU()

        self.drop_out = nn.Dropout(p=dropout_per)
        self.fc1 = nn.Linear(3584, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 1)


    def forward(self, x, train):
        # Layer norm goes after layer output before activation function
        # Additions are residual connections
        # print(x.size())

        out = self.layer1conv(x)
        out = self.layer1relu(out)
        out = self.layer1Norm(out)
        out = self.layer1pool(out)

        out = self.layer2conv(out)
        out = self.layer2relu(out)
        out = self.layer2Norm(out)
        out = self.layer2pool(out)

        out = self.layer3conv(out)
        out = self.layer3relu(out)
        out = self.layer3Norm(out)

        out = self.layer4conv(out)
        out = self.layer4relu(out)
        out = self.layer4Norm(out)

        out = out.reshape(out.size(0), -1)
        if train:
            out = self.drop_out(out)


        LN1 = nn.LayerNorm(self.fc1(out).size()[1:]).to(device)
        out = F.relu(LN1(self.fc1(out)))

        LN2 = nn.LayerNorm(self.fc2(out).size()[1:]).to(device)
        out = torch.add(F.relu(LN2(self.fc2(out))),out)
        # out = F.relu(LN2(self.fc2(out)))

        LN3 = nn.LayerNorm(self.fc3(out).size()[1:]).to(device)
        out = torch.add(F.relu(LN3(self.fc3(out))),out)
        # out = F.relu(LN3(self.fc3(out)))

        # LN1 = nn.LayerNorm(self.fc1(out).size()[1:]).to(device)
        # out = F.relu(LN1(self.fc1(out)))
        # LN2 = nn.LayerNorm(self.fc2(out).size()[1:]).to(device)
        # out = F.relu(LN2(self.fc2(out)))
        # out = F.relu(self.fc2(out))
        out = self.fc4(out)
        # print(out.size())
        return out


#Create the network
model = Net()
OMOQ.count_parameters(model)
sys.exit()
model.to(device)
# Setup the optimizer
print('Setup the optimizer')
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# criterion = nn.L1Loss() # Mean Absolute Error Loss
# loss_type = 'MAE'
criterion = nn.MSELoss() # Mean Squared Error Loss
loss_type = 'MSE'


#Create vector to store loss values after each epoch
loss_vals = np.zeros(epochs)
test_loss_vals = np.zeros(epochs)
pcc_vals = np.zeros(epochs)
mean_pcc_vals = np.zeros(epochs)
epoch_vals = np.arange(0,epochs,1)

#
#

# test_loss = np.zeros((test_averaging,1))
OMOQ_vals = np.zeros((len(test_list), test_averaging, epochs))
SMOQ_vals = np.zeros((len(test_list), test_averaging, epochs))



print('Training the Network')
# Train the network
for epoch in range(epochs):

    for batch_idx, (data, target) in enumerate(train_loader):
        # print("Training Data Shape : ", data.shape)

        # plt.figure(figsize=(10, 4))
        # plt.imshow(data[0,0,:,:])
        # # plt.colorbar()
        # plt.title('MFCC')
        # plt.tight_layout()
        # plt.colorbar()
        # plt.savefig("Training_MFCCs_Norm_imshow.png",dpi=300,format='png')
        # sys.exit()



        # Transfer data and target to device
        train_data = data.to(device)
        train_target = target.to(device)
        optimizer.zero_grad()
        # print("Tensor Data: ", torch.from_numpy(data))
        # print("Tensor Target: ", torch.from_numpy(target))
        model.train()
        net_out = model(train_data, 1) # 1 to denote training
        # print("Net Out: ", net_out)#.view(training_batch_size))
        # print("Target: ", train_target)
        # loss = criterion(net_out.view(training_batch_size), train_target)

        loss = criterion(net_out, train_target)
        loss.backward()
        optimizer.step()


        # if batch_idx % log_interval == 0:
        #     print('Epoch: {} \tLoss: {:.6f}'.format(epoch, torch.Tensor.item(loss.data)),end='')
            # sys.exit()

    #Save the loss into an array for plotting
    print('Epoch: {} \tLoss: {:.6f}'.format(epoch, torch.Tensor.item(loss.data)),end='')
    loss_vals[epoch] = torch.Tensor.item(loss.data)
    # Test the network after each epoch

    for a in range(test_averaging):
        test_loss = 0
        idx = 0
        for data, target in test_loader:
            # print("Target data shape: ", data.shape)
            # plt.figure(figsize=(10, 4))
            # plt.imshow(data[0,0,:,:])
            # # plt.colorbar()
            # plt.title('MFCC')
            # plt.tight_layout()
            # plt.colorbar()
            # plt.savefig("Testing_MFCCs_imshow2.png",dpi=300,format='png')
            # sys.exit()

            test_data = data.to(device)
            test_target = target.to(device)
            model.eval()
            test_net_out = model(test_data, 0)
            # Store Objective and Subjective values
            OMOQ_vals[idx,a,epoch] = torch.Tensor.item(test_net_out.view(testing_batch_size))
            SMOQ_vals[idx,a,epoch] = torch.Tensor.item(test_target)
            # OMOQ_vals[idx,epoch] = np.mean(torch.Tensor.tolist(net_out))
            # SMOQ_vals[idx,epoch] = np.mean(torch.Tensor.tolist(test_target))
            idx += 1
            # sum up batch loss
            # test_loss += criterion(net_out.view(testing_batch_size), test_target).data
            test_loss += criterion(test_net_out, test_target).data
        #Average the batch loss
        test_loss /= len(test_loader.dataset)
        test_loss_vals[epoch] = torch.Tensor.item(test_loss)
        print('\t Test loss: {:.6f}'.format(test_loss),end='')

    pcc = np.corrcoef(OMOQ_vals[:,-1,epoch],SMOQ_vals[:,-1,epoch])
    # print("\n mean OMOQ_vals\n", np.mean(OMOQ_vals[:,:,epoch],axis=1))
    mean_pcc = np.corrcoef(np.mean(OMOQ_vals[:,:,epoch],axis=1),np.mean(SMOQ_vals[:,:,epoch],axis=1))
    pcc_vals[epoch] = pcc[0,1]
    mean_pcc_vals[epoch] = mean_pcc[0,1]
    print('\tPCC: {:.6f}\tMeanPCC: {:.6f}'.format(pcc[0,1],mean_pcc[0,1]))
    if pcc_vals[epoch] > best_pcc:
        print("Saving Model\n")
        best_pcc = pcc_vals[epoch]
        torch.save(model.state_dict(), PATH)
    if mean_pcc_vals[epoch] > best_mean_pcc:
        print("Saving Mean Model\n")
        best_mean_pcc = mean_pcc_vals[epoch]
        torch.save(model.state_dict(), Mean_PATH)


best_epoch = np.argmax(pcc_vals)
best_mean_pcc_epoch = np.argmax(mean_pcc_vals)
print("Saving log of Test results\n")
if not os.path.exists('log'): os.makedirs('log') # create log directory.
with open(new_folder_name + "/Test.csv", "a") as results: results.write("Filename, TSM, File MeanOS, SMOS, OMOS\n")
with open(new_folder_name + "/Test.csv", "a") as results:
    for n in range(len(test_list)):
        # print(test_list[n].get('file'))
        results.write(str(test_list[n].get('MATLAB_loc')))
        results.write(", ")
        results.write(str(test_list[n].get('file')).split('_')[-2])
        results.write(", ")
        results.write(str(test_list[n].get('MeanOS')))
        results.write(", ")
        results.write(str(SMOQ_vals[n,-1,best_epoch]))
        results.write(", ")
        results.write(str(OMOQ_vals[n,-1,best_epoch])+ "\n")



# torch.load(PATH)


# #Compare to synthetic testing of val files.
# load_val_file = "OMOQ_CNN_MFCC_Deltas_Val.p"
# print('Loading Val Dataset')
# with open("./data/"+load_val_file, 'rb') as f:
#     val_list = pickle.load(f)
#
# # print(val_list)
# print('Setting up Validation data')
# val_dataset = OMOQ.OMOQDataset(val_list)
# val_loader = DataLoader(val_dataset, val_batch_size, shuffle=False, sampler=None, \
#                                                     batch_sampler=None, num_workers=0, collate_fn=OMOQ.collate_fn_CNN_2Channels, \
#                                                     pin_memory=False, drop_last=False, timeout=0, \
#                                                     worker_init_fn=None, multiprocessing_context=None)
#
#
# val_OMOQ_vals = np.zeros((len(val_list),1))
# val_SMOQ_vals = np.zeros((len(val_list),1))
# val_loss = 0
# idx = 0
# for data, target in val_loader:
#     val_data = data.to(device)
#     val_target = target.to(device)
#     net_out = model(val_data, 0)
#     # Store Objective and Subjective values
#     val_OMOQ_vals[idx,0] = torch.Tensor.item(net_out.view(testing_batch_size))
#     # rng = np.random.default_rng()
#     val_SMOQ_vals[idx,0] = torch.Tensor.item(val_target)#-rng.random()*np.finfo(np.float32).eps
#     # OMOQ_vals[idx,epoch] = np.mean(torch.Tensor.tolist(net_out))
#     # SMOQ_vals[idx,epoch] = np.mean(torch.Tensor.tolist(test_target))
#     idx += 1
#     # sum up batch loss
#     # test_loss += criterion(net_out.view(testing_batch_size), test_target).data
#     val_loss += criterion(net_out, val_target).data
# #Average the batch loss
# val_loss /= len(val_loader.dataset)
# val_loss_vals = torch.Tensor.item(val_loss)
# # print('\t Val loss: {:.6f}'.format(val_loss))
# # print("Obective: ", val_OMOQ_vals, "Subjective", val_SMOQ_vals)
# # val_pcc = np.corrcoef(val_SMOQ_vals,val_OMOQ_vals)
# # val_pcc_vals = val_pcc[0,1]
# # print('\tPCC: {:.6f}'.format(val_pcc[0,1]))
#
# if not os.path.exists('log'): os.makedirs('log') # create log directory.
# with open("log/Val.csv", "a") as results: results.write("Filename, TSM, File MeanOS, SMOS, OMOS\n")
# with open("log/Val.csv", "a") as results:
#     for n in range(len(val_list)):
#         # print(val_list[n].get('file'))
#         results.write(str(val_list[n].get('file')))
#         results.write(", ")
#         results.write(str(val_list[n].get('file')).split('_')[-2])
#         results.write(", ")
#         results.write(str(val_list[n].get('MeanOS')))
#         results.write(", ")
#         results.write(str(val_SMOQ_vals[n,0]))
#         results.write(", ")
#         results.write(str(val_OMOQ_vals[n,0])+ "\n")
# 		# results.write("%s, %3.2f\n" % (val_list[n].get('name'), val_OMOQ_vals[n,0]))


best_epoch = np.argmax(pcc_vals)
print("Saving Network Variables\n")
file1 = open(new_folder_name + "/Network.txt","a")
file1.write("\n")
file1.write("Feature File: ")
file1.write(load_file)
file1.write("\n")
file1.write("Normalisation = BatchNorm2d\n")
file1.write("4 CNN layers: 16, 32, 64, 32\n")
file1.write("Kernels: 3x3")
file1.write("3 FCN layers: 3584, 128, 128, 128, 1\n") # Was 10240 for 3 CNN layers
file1.write("FCN Residual Connections\n")
file1.write("Test Averaging = ")
file1.write(str(test_averaging))
file1.write("\n")
file1.write("Training target = MeanOS")
file1.write("\n")
file1.write("Training Batch Size = ")
file1.write(str(training_batch_size))
file1.write("\n")
file1.write("Learning Rate = ")
file1.write(str(learning_rate))
file1.write("\n")
file1.write("Epochs = ")
file1.write(str(epochs))
file1.write("\n")
file1.write("Loss Type = ")
file1.write(loss_type)
file1.write("\n")
file1.write("Adam Optimisation\n")
file1.write("Dropout = ")
file1.write(str(dropout_per))
file1.write("\n")
# file1.write("Residual Connections = TRUE")
# file1.write("\n")
file1.write("Final Training Loss = ")
file1.write(str(loss_vals[-1]))
file1.write("\n")
file1.write("Final Testing Loss = ")
file1.write(str(test_loss_vals[-1]))
file1.write("\n")
file1.write("Final Pearson Correlation Coefficient = ")
file1.write(str(pcc_vals[-1]))
file1.write("\n")
file1.write("Best Epoch = ")
file1.write(str(best_epoch))
file1.write("\n")
file1.write("Best Pearson Correlation Coefficient = ")
file1.write(str(np.max(pcc_vals)))
file1.write("\n")
file1.write("Training Loss at best PCC = ")
file1.write(str(loss_vals[best_epoch]))
file1.write("\n")
file1.write("Testing Loss at best PCC = ")
file1.write(str(test_loss_vals[best_epoch]))
file1.write("\n")

file1.write("Best Mean PCC Epoch = ")
file1.write(str(best_mean_pcc_epoch))
file1.write("\n")
file1.write("Best Mean PCC = ")
file1.write(str(np.max(mean_pcc_vals)))

# file1.write("Source Pearson Correlation Coefficient at best PCC = ")
# file1.write(str(source_pcc_vals))
# file1.write("\n")
# file1.write("Source Loss at best PCC = ")
# file1.write(str(source_loss))
# file1.write("\n")
file1.close()

print("Plotting figures\n")
#Plot all the things
plt.figure()
plt.plot(epoch_vals,loss_vals)
plt.plot(epoch_vals,test_loss_vals)
plt.plot(epoch_vals,pcc_vals)
plt.plot(epoch_vals,mean_pcc_vals)
plt.xlabel('Epoch')
plt.ylabel('MAE Loss')
plt.legend(['Training Loss','Testing Loss', 'PCC', 'Mean PCC'], loc='best')
save_name = new_folder_name + "/Loss_" + load_file[:-2].replace("/","-") + ".png"
plt.savefig(save_name,dpi=300,format='png')
# plt.show()


line_x = [1, 5]
line_y = line_x
plt.figure()
plt.hist2d(SMOQ_vals[:,-1,best_epoch],OMOQ_vals[:,-1,best_epoch],bins=40,range = [[0, 6],[0, 6]])
cb = plt.colorbar()
cb.set_label('Count')
plt.plot(line_x,line_y,'r--', linewidth=2)
plt.xlabel('Subjective MeanOS')
plt.ylabel('Objective MOQ')
plt.title('Confusion Matrix at Best PCC Epoch')
save_name = new_folder_name + "/Subjective_vs_Objective_Best_" + load_file[:-2].replace("/","-") + ".png"
plt.savefig(save_name,dpi=300,format='png')
# plt.show()

line_x = [1, 5]
line_y = line_x
plt.figure()
plt.hist2d(SMOQ_vals[:,-1,best_mean_pcc_epoch],OMOQ_vals[:,-1,best_mean_pcc_epoch],bins=40,range = [[0, 6],[0, 6]])
cb = plt.colorbar()
cb.set_label('Count')
plt.plot(line_x,line_y,'r--', linewidth=2)
plt.xlabel('Subjective MeanOS')
plt.ylabel('Objective MOQ')
plt.title('Confusion Matrix at Best Mean PCC Epoch')
save_name = new_folder_name + "/Subjective_vs_Objective_MeanPCC_" + load_file[:-2].replace("/","-") + ".png"
plt.savefig(save_name,dpi=300,format='png')




print("Best PCC: ", best_pcc)



#Comment to put the code at a more readable position after saving
