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
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

verb = 0
learning_rate = 1e-4 #Have tried -2 to -6
epochs = 100
log_interval = 50
trainbs = 132 #training_batch_size #factor of number of training files (Tried 1 to 5280)
valbs = 264
testbs = 120
source_batch_size = 1
val_batch_size = 1
dropout_per = 0.1 #Tried 0, 0.1, 0.25 0.4 0.5
best_pcc = 0.2
best_dist = 5
best_mean_pcc = 0.2
test_averaging = 8 #Number of times to sample signal in testing.
# load_folder = "./data/Features/MagPhasePow/"
load_folder = "./data/Features/MFCC_Delta_Delta_NoNorm_Trim_Source/"
# fname = 'CNN_Pow_NoNorm_Kernel5333'
fname = 'CNN_MFCC_Delta_DDelta_NoNorm_Kernel3333_dummy'
CSV_NAME = fname+".csv"

seed_start = 0
seed_stop = 30
seeds = range(seed_start, seed_stop)
best_seed = 0;
best_pcc_overall = 0;
best_loss_overall = 5;

for seed_loop in seeds:
    my_seed = seed_loop
    PCC_CSV = open("models/"+CSV_NAME,"a")
    PCC_CSV.write("\nFolder,BatchSize,Target,TrainLoss,ValLoss,TestLoss,TrainPCC,ValPCC,TestPCC,BestEpoch,LRate,BestTestPCC,Seed,MeanLoss,MeanPCC,DiffLoss,DiffPCC,Dist\n")
    PCC_CSV.close()
    # Make deterministic
    torch.manual_seed(my_seed)
    torch.cuda.manual_seed(my_seed)
    np.random.seed(my_seed)
    random.seed(my_seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark = False
    print("Seed loop: ", seed_loop+1, "/", seed_stop)
    # Create folder for results
    new_folder_name = 'models/CNN/'+str(datetime.datetime.now())[:19].replace(" ","_").replace(":","-")
    if not os.path.exists(new_folder_name): os.makedirs(new_folder_name) # create log directory.

    PATH = new_folder_name+'/OMOQ.pth'
    Mean_PATH = new_folder_name+'/MeanPCC_OMOQ.pth'

    load_file = "Feat.p"


    PCC_CSV = open("models/"+CSV_NAME,"a")
    PCC_CSV.write(new_folder_name)
    PCC_CSV.write(",")
    PCC_CSV.write(str(trainbs))
    PCC_CSV.write(",")
    PCC_CSV.write("Mean")
    PCC_CSV.write(",")
    PCC_CSV.close()

    # Load the dataset
    print('Loading Dataset')
    with open(load_folder+load_file, 'rb') as f:
        train_list, val_list, test_list, Norm_vals= pickle.load(f)

    # print(train_list)
    # print(test_list)
    # print(Norm_vals)
    # sys.exit()
    # training_mean = np.mean(features['MOVs'][:,4:],0)
    # training_std = np.std(features['MOVs'][:,4:],0)
    # features_norm = (features['MOVs'][:,4:]-training_mean)/training_std

    print('Setting up training data')
    training_dataset = OMOQ.OMOQDatasetCNN(train_list)
    train_loader = DataLoader(training_dataset, trainbs, shuffle=False, sampler=None, \
                                                        batch_sampler=None, num_workers=0, \
                                                        collate_fn=OMOQ.collate_fn_CNN_NChannels, \
                                                        pin_memory=False, drop_last=False, timeout=0, \
                                                        worker_init_fn=None, multiprocessing_context=None)

    print('Setting up validation data')
    validation_dataset = OMOQ.OMOQDatasetCNN(val_list)
    validation_loader = DataLoader(validation_dataset, valbs, shuffle=False, sampler=None, \
                                                        batch_sampler=None, num_workers=0, \
                                                        collate_fn=OMOQ.collate_fn_CNN_NChannels, \
                                                        pin_memory=False, drop_last=False, timeout=0, \
                                                        worker_init_fn=None, multiprocessing_context=None)

    # print("Training Loader: ", train_loader)
    # sys.exit()

    # Setup the data for training
    print('Setting up testing data')
    # Setup the data for testing
    testing_dataset = OMOQ.OMOQDatasetCNN(test_list)
    # testing_dataset = TensorDataset(torch.as_tensor(temp_data),torch.from_numpy(temp_mos))#,torch.from_numpy(temp_length))
    test_loader = DataLoader(testing_dataset, testbs, shuffle=False, sampler=None, \
                                                        batch_sampler=None, num_workers=0, \
                                                        collate_fn=OMOQ.collate_fn_CNN_NChannels, \
                                                        pin_memory=False, drop_last=False, timeout=0, \
                                                        worker_init_fn=None, multiprocessing_context=None)
    # # Test the DataLoader
    # for batch_idx, (data, target) in enumerate(test_loader):
    #     print(batch_idx)
    #     print("Data = ", data,"Target = ", target)
    #     sys.exit()




    print('Defining the Convolutional Network')
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            #Input Layer
            self.layer1conv = nn.Conv2d(3, 16, kernel_size=(5,5), stride=(1,1), \
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

            # print(out.shape)
            LN1 = nn.LayerNorm(self.fc1(out).size()[1:]).to(device)
            out = F.relu(LN1(self.fc1(out)))
            # print(out.shape)
            LN2 = nn.LayerNorm(self.fc2(out).size()[1:]).to(device)
            out = torch.add(F.relu(LN2(self.fc2(out))),out)
            # out = F.relu(LN2(self.fc2(out)))
            # print(out.shape)
            LN3 = nn.LayerNorm(self.fc3(out).size()[1:]).to(device)
            out = torch.add(F.relu(LN3(self.fc3(out))),out)
            # out = F.relu(LN3(self.fc3(out)))
            # print(out.shape)
            # LN1 = nn.LayerNorm(self.fc1(out).size()[1:]).to(device)
            # out = F.relu(LN1(self.fc1(out)))
            # LN2 = nn.LayerNorm(self.fc2(out).size()[1:]).to(device)
            # out = F.relu(LN2(self.fc2(out)))
            # out = F.relu(self.fc2(out))
            out = torch.sigmoid(self.fc4(out))
            # print(out.size())
            # print(out.shape)
            # sys.exit()
            return out


    #Create the network
    model = Net()
    model.to(device)
    # Setup the optimizer
    print('Setup the optimizer')
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    # criterion = nn.L1Loss() # Mean Absolute Error Loss
    # loss_type = 'MAE'
    criterion = nn.MSELoss() # Mean Squared Error Loss
    loss_type = 'RMSE'

    #Create vector to store loss values after each epoch
    train_loss_vals = np.zeros(epochs)
    val_loss_vals = np.zeros(epochs)
    test_loss_vals = np.zeros(epochs)
    train_pcc_vals = np.zeros(epochs)
    val_pcc_vals = np.zeros(epochs)
    test_pcc_vals = np.zeros(epochs)
    mean_pcc_vals = np.zeros(epochs)
    loss_dist = np.zeros(epochs)
    pcc_dist = np.zeros(epochs)
    dist = np.zeros(epochs)
    mean_loss = np.zeros(epochs)
    mean_pcc = np.zeros(epochs)
    diff_loss = np.zeros(epochs)
    diff_pcc = np.zeros(epochs)
    epoch_vals = np.arange(0,epochs,1)





    test_loss = np.zeros((test_averaging,1))
    TRAIN_OMOQ_vals = np.zeros((len(train_list), epochs))
    TRAIN_SMOQ_vals = np.zeros((len(train_list), epochs))
    VAL_OMOQ_vals = np.zeros((len(val_list), epochs))
    VAL_SMOQ_vals = np.zeros((len(val_list), epochs))
    TEST_OMOQ_vals = np.zeros((len(test_list), epochs))
    TEST_SMOQ_vals = np.zeros((len(test_list), epochs))
    TEST_OMOQ = np.zeros((len(test_list), epochs, test_averaging))
    TEST_SMOQ = np.zeros((len(test_list), epochs, test_averaging))

    print('Training the Network')
    # Train the network
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
            # print("Training Data Shape : ", data.shape)

            # Transfer data and target to device
            train_data = data.to(device)
            train_target = target.to(device)
            optimizer.zero_grad()
            # print("Tensor Data: ", torch.from_numpy(data))
            # print("Tensor Target: ", torch.from_numpy(target))
            model = model.train()
            net_out = model(train_data, 1) # 1 to denote training
            # print(net_out.data)
            # sys.exit()
            # print("Net Out: ", net_out)#.view(trainbs))
            # print("Target: ", train_target)
            # loss = criterion(net_out.view(trainbs), train_target)
            loss = torch.sqrt(criterion(net_out, train_target))
            loss.backward()
            optimizer.step()
            # print(TRAIN_OMOQ_vals[batch_idx*trainbs:batch_idx*trainbs+trainbs,epoch].shape)
            # print((np.array(torch.Tensor.tolist(net_out.view(trainbs)))*4+1).flatten().shape)
            # sys.exit()

            TRAIN_OMOQ_vals[batch_idx*trainbs:batch_idx*trainbs+trainbs,epoch] = (np.array(torch.Tensor.tolist(net_out.view(trainbs)))*4+1).flatten()
            TRAIN_SMOQ_vals[batch_idx*trainbs:batch_idx*trainbs+trainbs,epoch] = (np.array(torch.Tensor.tolist(train_target))*4+1).flatten()


            # if batch_idx % log_interval == 0:
            #     print('Epoch: {} \tLoss: {:.6f}'.format(epoch, torch.Tensor.item(loss.data)),end='')
                # sys.exit()

        train_pcc_vals[epoch] = np.corrcoef(TRAIN_OMOQ_vals[:,epoch],TRAIN_SMOQ_vals[:,epoch])[0][1]
        train_loss_vals[epoch] = np.sqrt(np.square(np.subtract(TRAIN_OMOQ_vals[:,epoch], TRAIN_SMOQ_vals[:,epoch])).mean()) #RMSE
        #Save the loss into an array for plotting
        print('Epoch: {} \tTrain RMSE: {:.6f} \tTrain PCC: {:.6f}'.format(epoch, train_loss_vals[epoch], train_pcc_vals[epoch],end=''))
        # train_loss_vals[epoch] = torch.Tensor.item(loss.data)
        # Test the network after each epoch

        # Validate the network after each epoch
        for v_idx, (val_data, val_target) in enumerate(validation_loader):

            val_data = val_data.to(device)
            # print(test_target)
            val_target = val_target.to(device)
            model = model.eval()
            val_net_out = model(val_data, 0)
            # Store Objective and Subjective values
            VAL_OMOQ_vals[v_idx*valbs:v_idx*valbs+valbs,epoch] = (np.array(torch.Tensor.tolist(val_net_out.view(valbs)))*4+1).flatten()
            # print((np.array(torch.Tensor.tolist(test_target))*4+1).flatten())
            VAL_SMOQ_vals[v_idx*valbs:v_idx*valbs+valbs,epoch] = (np.array(torch.Tensor.tolist(val_target))*4+1).flatten()

        val_pcc_vals[epoch] = np.corrcoef(VAL_OMOQ_vals[:,epoch],VAL_SMOQ_vals[:,epoch])[0][1]
        val_loss_vals[epoch] = np.sqrt(np.square(np.subtract(VAL_OMOQ_vals[:,epoch], VAL_SMOQ_vals[:,epoch])).mean()) #RMSE
        print('\t\t Val RMSE: {:.6f}\tVal PCC: {:.6f}'.format(val_loss_vals[epoch], val_pcc_vals[epoch],end=''))


        for n in range(0,test_averaging):
            for t_idx, (data, target) in enumerate(test_loader):

                test_data = data.to(device)
                test_target = target.to(device)
                model = model.eval()
                test_net_out = model(test_data, 0)
                # Store Objective and Subjective values

                # OMOQ_vals[idx,epoch] = np.mean(torch.Tensor.tolist(net_out))
                # SMOQ_vals[idx,epoch] = np.mean(torch.Tensor.tolist(test_target))
                TEST_OMOQ[t_idx*testbs:t_idx*testbs+testbs,epoch,n] = (np.array(torch.Tensor.tolist(test_net_out.view(testbs)))*4+1).flatten()
                # print(TEST_OMOQ[:,epoch,n])
                TEST_SMOQ[t_idx*testbs:t_idx*testbs+testbs,epoch,n] = (np.array(torch.Tensor.tolist(test_target))*4+1).flatten()
                # sum up loss across test averaging
                # test_loss += criterion(net_out.view(testbs), test_target).data
                test_loss = criterion(test_net_out, test_target).data
                # print(test_loss)
                # sys.exit()
        #Average all of the OMOQ predictions
        TEST_OMOQ_vals[:,epoch] = np.mean(TEST_OMOQ[:,epoch,:],1)
        TEST_SMOQ_vals[:,epoch] = TEST_SMOQ[:,epoch,0]
        #Average the batch loss
        # test_loss /= len(test_loader.dataset)
        # test_loss_vals[epoch] = torch.Tensor.item(test_loss)

        # print('\n')
        # print(OMOQ_vals[:,-1,epoch])
        # print('\n')
        # print(SMOQ_vals[:,-1,epoch])
        # print(np.corrcoef(TEST_OMOQ_vals[:,epoch],TEST_SMOQ_vals[:,epoch])[0][1])
        test_pcc_vals[epoch] = np.corrcoef(TEST_OMOQ_vals[:,epoch],TEST_SMOQ_vals[:,epoch])[0][1]
        test_loss_vals[epoch] = np.sqrt(np.square(np.subtract(TEST_OMOQ_vals[:,epoch], TEST_SMOQ_vals[:,epoch])).mean()) #RMSE
        # print('\n')
        # print(pcc)
        # sys.exit()
        # print("\n mean TEST_OMOQ_vals\n", np.mean(TEST_OMOQ_vals[:,:,epoch],axis=1))
        # mean_pcc = np.corrcoef(np.mean(TEST_OMOQ_vals[:,:,epoch],axis=1),np.mean(TEST_SMOQ_vals[:,:,epoch],axis=1))
        # train_pcc_vals[epoch] = test_pcc[0,1]
        # mean_pcc_vals[epoch] = mean_pcc[0,1]
        print('\t\t Test RMSE: {:.6f}\tTest PCC: {:.6f}'.format(test_loss_vals[epoch], test_pcc_vals[epoch],end=''))

        # sys.exit()

        #Distance Calculation

        #Distance Calculation
        mean_loss[epoch] = np.mean((train_loss_vals[epoch],val_loss_vals[epoch],test_loss_vals[epoch]))
        mean_pcc[epoch] = np.mean((train_pcc_vals[epoch],val_pcc_vals[epoch],test_pcc_vals[epoch]))
        diff_loss[epoch] = max((train_loss_vals[epoch],val_loss_vals[epoch],test_loss_vals[epoch]))-min((train_loss_vals[epoch],val_loss_vals[epoch],test_loss_vals[epoch]))
        diff_pcc[epoch] = max((train_pcc_vals[epoch],val_pcc_vals[epoch],test_pcc_vals[epoch]))-min((train_pcc_vals[epoch],val_pcc_vals[epoch],test_pcc_vals[epoch]))
        loss_dist[epoch] = np.sqrt((np.square(mean_loss[epoch])+np.square(diff_loss[epoch])))
        pcc_dist[epoch] = np.sqrt((np.square((1-mean_pcc[epoch]))+np.square(diff_pcc[epoch]))) #1-mean used to have smaller=better for all.
        dist[epoch] = np.sqrt((np.square(loss_dist[epoch])+np.square(pcc_dist[epoch])))
        print('\tDistance: {:.6f}'.format(dist[epoch]))
        if dist[epoch] < best_dist:
            print("Saving Model\n")
            best_dist = dist[epoch]
            torch.save(model.state_dict(), PATH)


        # if test_pcc_vals[epoch] > best_pcc:
        #     print("Saving Model\n")
        #     best_pcc = test_pcc_vals[epoch]
        #     torch.save(model.state_dict(), PATH)
        # if mean_pcc_vals[epoch] > best_mean_pcc:
        #     print("Saving Mean Model\n")
        #     best_mean_pcc = mean_pcc_vals[epoch]
        #     torch.save(model.state_dict(), Mean_PATH)

    best_epoch = np.argmin(dist)
    # best_mean_pcc_epoch = np.argmax(mean_pcc_vals)
    #Write out to the csv
    PCC_CSV = open("models/"+CSV_NAME,"a")
    PCC_CSV.write(str(train_loss_vals[best_epoch]))
    PCC_CSV.write(",")
    PCC_CSV.write(str(val_loss_vals[best_epoch]))
    PCC_CSV.write(",")
    PCC_CSV.write(str(test_loss_vals[best_epoch]))
    PCC_CSV.write(",")
    PCC_CSV.write(str(train_pcc_vals[best_epoch]))
    PCC_CSV.write(",")
    PCC_CSV.write(str(val_pcc_vals[best_epoch]))
    PCC_CSV.write(",")
    PCC_CSV.write(str(test_pcc_vals[best_epoch]))
    PCC_CSV.write(",")
    PCC_CSV.write(str(best_epoch))
    PCC_CSV.write(",")
    PCC_CSV.write(str(learning_rate))
    PCC_CSV.write(",")
    PCC_CSV.write(str(np.max(test_pcc_vals)))
    PCC_CSV.write(",")
    PCC_CSV.write(str(my_seed))
    PCC_CSV.write(",")
    PCC_CSV.write(str(mean_loss[best_epoch]))
    PCC_CSV.write(",")
    PCC_CSV.write(str(mean_pcc[best_epoch]))
    PCC_CSV.write(",")
    PCC_CSV.write(str(diff_loss[best_epoch]))
    PCC_CSV.write(",")
    PCC_CSV.write(str(diff_pcc[best_epoch]))
    PCC_CSV.write(",")
    PCC_CSV.write(str(dist[best_epoch])+"\n")
    PCC_CSV.close()


    print("Saving log of Test results\n")
    if not os.path.exists('log'): os.makedirs('log') # create log directory.
    with open(new_folder_name + "/Test.csv", "a") as results: results.write("Filename, TSM, FileMeanOS, SMOS, OMOS\n")
    with open(new_folder_name + "/Test.csv", "a") as results:
        for n in range(len(test_list)):
            # print(test_list[n].get('file'))
            results.write(str(test_list[n].get('file')))
            results.write(", ")
            results.write(str(test_list[n].get('file')).split('_')[-2])
            results.write(", ")
            results.write(str(test_list[n].get('MeanOS')))
            results.write(", ")
            results.write(str(TEST_SMOQ_vals[n,best_epoch]))
            results.write(", ")
            results.write(str(TEST_OMOQ_vals[n,best_epoch])+ "\n")
    print("Saving log of Val results\n")
    if not os.path.exists('log'): os.makedirs('log') # create log directory.
    with open(new_folder_name + "/Val.csv", "a") as results: results.write("Filename, TSM, FileMeanOS, SMOS, OMOS\n")
    with open(new_folder_name + "/Val.csv", "a") as results:
        for n in range(len(val_list)):
            # print(test_list[n].get('file'))
            results.write(str(val_list[n].get('file')))
            results.write(", ")
            results.write(str(val_list[n].get('file')).split('_')[-2])
            results.write(", ")
            results.write(str(val_list[n].get('MeanOS')))
            results.write(", ")
            results.write(str(VAL_SMOQ_vals[n,best_epoch]))
            results.write(", ")
            results.write(str(VAL_OMOQ_vals[n,best_epoch])+ "\n")
    print("Saving log of Train results\n")
    if not os.path.exists('log'): os.makedirs('log') # create log directory.
    with open(new_folder_name + "/Train.csv", "a") as results: results.write("Filename, TSM, FileMeanOS, SMOS, OMOS\n")
    with open(new_folder_name + "/Train.csv", "a") as results:
        for n in range(len(train_list)):
            # print(test_list[n].get('file'))
            results.write(str(train_list[n].get('file')))
            results.write(", ")
            results.write(str(train_list[n].get('file')).split('_')[-2])
            results.write(", ")
            results.write(str(train_list[n].get('MeanOS')))
            results.write(", ")
            results.write(str(TRAIN_SMOQ_vals[n,best_epoch]))
            results.write(", ")
            results.write(str(TRAIN_OMOQ_vals[n,best_epoch])+ "\n")



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
    #     val_OMOQ_vals[idx,0] = torch.Tensor.item(net_out.view(testbs))
    #     # rng = np.random.default_rng()
    #     val_SMOQ_vals[idx,0] = torch.Tensor.item(val_target)#-rng.random()*np.finfo(np.float32).eps
    #     # TEST_OMOQ_vals[idx,epoch] = np.mean(torch.Tensor.tolist(net_out))
    #     # TEST_SMOQ_vals[idx,epoch] = np.mean(torch.Tensor.tolist(test_target))
    #     idx += 1
    #     # sum up batch loss
    #     # test_loss += criterion(net_out.view(testbs), test_target).data
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


    # best_epoch = np.argmax(dist)
    # if test_pcc_vals[best_epoch]>best_pcc_overall:
    #     best_pcc_overall = test_pcc_vals[best_epoch]
    #     best_seed = my_seed

    print("Saving Network Variables\n")
    file1 = open(new_folder_name + "/Network.txt","a")
    file1.write("\n")
    file1.write("My Seed = ")
    file1.write(str(my_seed))
    file1.write("\n")
    file1.write("Feature File: ")
    file1.write(load_folder+load_file)
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
    file1.write(str(trainbs))
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
    file1.write("AdamW Optimisation\n")
    file1.write("Dropout = ")
    file1.write(str(dropout_per))
    file1.write("\n")
    # file1.write("Residual Connections = TRUE")
    # file1.write("\n")
    file1.write("Final Training Loss = ")
    file1.write(str(train_loss_vals[-1]))
    file1.write("\n")
    file1.write("Final Testing Loss = ")
    file1.write(str(test_loss_vals[-1]))
    file1.write("\n")
    file1.write("Final Training Pearson Correlation Coefficient = ")
    file1.write(str(train_pcc_vals[-1]))
    file1.write("\n")
    file1.write("Final Testing Pearson Correlation Coefficient = ")
    file1.write(str(test_pcc_vals[-1]))
    file1.write("\n")
    file1.write("Best Epoch = ")
    file1.write(str(best_epoch))
    file1.write("\n")
    file1.write("Best Training Pearson Correlation Coefficient = ")
    file1.write(str(np.max(train_pcc_vals)))
    file1.write("\n")
    file1.write("Best Testing Pearson Correlation Coefficient = ")
    file1.write(str(np.max(test_pcc_vals)))
    file1.write("\n")
    file1.write("Training Loss at best PCC = ")
    file1.write(str(train_loss_vals[best_epoch]))
    file1.write("\n")
    file1.write("Testing Loss at best PCC = ")
    file1.write(str(test_loss_vals[best_epoch]))
    file1.write("\n")

    file1.write("Best PCC Epoch = ")
    file1.write(str(best_epoch))
    file1.write("\n")
    # file1.write("Best Mean PCC = ")
    # file1.write(str(np.max(mean_pcc_vals)))

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
    line_x = [best_epoch, best_epoch]
    line_y = [0, 1]
    plt.plot(epoch_vals,train_loss_vals)
    plt.plot(epoch_vals,val_loss_vals)
    plt.plot(epoch_vals,test_loss_vals)
    plt.plot(epoch_vals,train_pcc_vals)
    plt.plot(epoch_vals,val_pcc_vals)
    plt.plot(epoch_vals,test_pcc_vals)
    plt.plot(line_x,line_y,'r--', linewidth=2)
    # plt.plot(epoch_vals,dist)
    plt.xlabel('Epoch')
    plt.ylabel('RMSE Loss')
    plt.legend(['Training RMSE','Validation RMSE','Testing RMSE', 'Training PCC', 'Validation PCC', 'Testing PCC','Best Epoch'], loc='best')
    save_name = new_folder_name + "/Loss_" + load_file[:-2].replace("/","-") + ".png"
    plt.savefig(save_name,dpi=300,format='png')
    # plt.show()


    line_x = [1, 5]
    line_y = line_x
    plt.figure()
    plt.hist2d(TEST_SMOQ_vals[:,best_epoch],TEST_OMOQ_vals[:,best_epoch],bins=40,range = [[0, 6],[0, 6]])
    cb = plt.colorbar()
    cb.set_label('Count')
    plt.plot(line_x,line_y,'r--', linewidth=2)
    plt.xlabel('Subjective MeanOS')
    plt.ylabel('Objective MOQ')
    plt.title('Testing Confusion Matrix at Best Distance Epoch')
    save_name = new_folder_name + "/Subjective_vs_Objective_Best_Testing_" + load_file[:-2].replace("/","-") + ".png"
    plt.savefig(save_name,dpi=300,format='png')
    # plt.show()

    line_x = [1, 5]
    line_y = line_x
    plt.figure()
    plt.hist2d(VAL_SMOQ_vals[:,best_epoch],VAL_OMOQ_vals[:,best_epoch],bins=40,range = [[0, 6],[0, 6]])
    cb = plt.colorbar()
    cb.set_label('Count')
    plt.plot(line_x,line_y,'r--', linewidth=2)
    plt.xlabel('Subjective MeanOS')
    plt.ylabel('Objective MOQ')
    plt.title('Validation Confusion Matrix at Best Distance Epoch')
    save_name = new_folder_name + "/Subjective_vs_Objective_Best_Validation_" + load_file[:-2].replace("/","-") + ".png"
    plt.savefig(save_name,dpi=300,format='png')
    # plt.show()

    line_x = [1, 5]
    line_y = line_x
    plt.figure()
    plt.hist2d(TRAIN_SMOQ_vals[:,best_epoch],TRAIN_OMOQ_vals[:,best_epoch],bins=40,range = [[0, 6],[0, 6]])
    cb = plt.colorbar()
    cb.set_label('Count')
    plt.plot(line_x,line_y,'r--', linewidth=2)
    plt.xlabel('Subjective MeanOS')
    plt.ylabel('Objective MOQ')
    plt.title('Training Confusion Matrix at Best Distance Epoch')
    save_name = new_folder_name + "/Subjective_vs_Objective_Best_Training_" + load_file[:-2].replace("/","-") + ".png"
    plt.savefig(save_name,dpi=300,format='png')
    # plt.show()

    line_x = [1, 5]
    line_y = line_x
    plt.figure()
    plt.hist2d(TEST_SMOQ_vals[:,-1],TEST_OMOQ_vals[:,-1],bins=40,range = [[0, 6],[0, 6]])
    cb = plt.colorbar()
    cb.set_label('Count')
    plt.plot(line_x,line_y,'r--', linewidth=2)
    plt.xlabel('Subjective MeanOS')
    plt.ylabel('Objective MOQ')
    plt.title('Testing Confusion Matrix at Final Epoch')
    save_name = new_folder_name + "/Subjective_vs_Objective_Final_Testing_" + load_file[:-2].replace("/","-") + ".png"
    plt.savefig(save_name,dpi=300,format='png')
    # plt.show()

    line_x = [1, 5]
    line_y = line_x
    plt.figure()
    plt.hist2d(VAL_SMOQ_vals[:,-1],VAL_OMOQ_vals[:,-1],bins=40,range = [[0, 6],[0, 6]])
    cb = plt.colorbar()
    cb.set_label('Count')
    plt.plot(line_x,line_y,'r--', linewidth=2)
    plt.xlabel('Subjective MeanOS')
    plt.ylabel('Objective MOQ')
    plt.title('Validation Confusion Matrix at Final Epoch')
    save_name = new_folder_name + "/Subjective_vs_Objective_Final_Validation_" + load_file[:-2].replace("/","-") + ".png"
    plt.savefig(save_name,dpi=300,format='png')
    # plt.show()

    line_x = [1, 5]
    line_y = line_x
    plt.figure()
    plt.hist2d(TRAIN_SMOQ_vals[:,-1],TRAIN_OMOQ_vals[:,-1],bins=40,range = [[0, 6],[0, 6]])
    cb = plt.colorbar()
    cb.set_label('Count')
    plt.plot(line_x,line_y,'r--', linewidth=2)
    plt.xlabel('Subjective MeanOS')
    plt.ylabel('Objective MOQ')
    plt.title('Training Confusion Matrix at Final Epoch')
    save_name = new_folder_name + "/Subjective_vs_Objective_Final_Training_" + load_file[:-2].replace("/","-") + ".png"
    plt.savefig(save_name,dpi=300,format='png')
    # plt.show()



# print("Best PCC: ", best_pcc)



#Comment to put the code at a more readable position after saving
