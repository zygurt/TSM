# Initial testing with the OMOQ dataset
# Built based on https://adventuresinmachinelearning.com/pytorch-tutorial-deep-learning/
import scipy.io as sio

import numpy as np
import torch, sys, os, datetime, smtplib, ssl, pickle, random
import matplotlib.pyplot as plt
from scipy import stats

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

plt.close('all')

verb = 0
val_split = 0.1
learning_rate = 0.0001
epochs = 800
log_interval = 880
training_batch_size = int((5280+88)*(1-val_split))  # This should be a factor of number of training files
val_batch_size = (5280+88)-training_batch_size
testing_batch_size = 240
dropout_per = 0
training_target = 0
early_stop_thresh = -5e-5
stopping_range = 60
averaging_range = 80
start_early_stopping = 150
feature_start = 5
load_file = 'data/Features/MOVs_Final_To_Test_Source_20200416.mat'
features = sio.loadmat(load_file)

fname = 'TO_TEST_SOURCE'
CSV_NAME = fname+".csv"

# OMOV = {'MeanOS', 'MedianOS', ...
#     'MeanOS_RAW', 'MedianOS_RAW', ...
#     'TSM', ...
#     'WinModDiff1B', 'AvgModDiff1B', 'AvgModDiff2B', ...
#     'RmsNoiseLoudB', ...
#     'BandwidthRefB', 'BandwidthTestB', 'BandwidthTestB_new', ...
#     'TotalNMRB', ...
#     'RelDistFramesB', ...
#     'MFPDB', 'ADBB', ...
#     'EHSB', ...
#     'RmsModDiffA', 'RmsNoiseLoudAsymA', 'AvgLinDistA', 'SegmentalNMRB', ...
#     'DM', 'SER', ...
#     'peak_delta', 'transient_ratio', 'hpsep_transient_ratio', ...
#     'MPhNW', 'SPhNW', ...
#     'MPhMW', 'SPhMW', ...
#     'SS_MAD','SS_MD'};

#Choose certain features
chosen_features = np.arange(0,features['MOVs'].shape[1]) # All the features

print(features['OMOV'][0,chosen_features])

# # Uncomment to concatenate log10 features
# # features['MOVs'] = features['MOVs'][:,chosen_features]
# # features['OMOV'] = features['OMOV'][:,chosen_features]
# # print(features['OMOV'][0,:])
# # # # Concatenate log10 features with original features
# for n in np.arange(feature_start,features['MOVs'].shape[1]):
#     # print('n: ',n)
#     # print(features['MOVs'].shape)
#     # print(features['MOVs'][:,n:n+1].shape)
#     if np.min(features['MOVs'][:,n])>0:
#         # Make log and concat
#         print('Making log: ',features['OMOV'][0,n])
#         # print(features['MOVs'].shape)
#         # print(features['MOVs'][:,n].shape)
#         # print(10*np.log10(features['MOVs'][:,n]).shape)
#         features['MOVs'] = np.concatenate((features['MOVs'], 10*np.log10(features['MOVs'][:,n:n+1])),axis=1)


#Reset the chosen features
chosen_features = np.arange(0,features['MOVs'].shape[1])


# Normalise the input features
# Normalise to zero mean and unity standard deviation
temp_mean = np.mean(features['MOVs'][0:(5280+88),feature_start:],0)
temp_std = np.std(features['MOVs'][0:(5280+88),feature_start:],0)
features_norm = (features['MOVs'][:,feature_start:]-temp_mean)/temp_std
# Rescale to 0-1
temp_min = np.amin(features_norm,0)
temp_max = np.amax(features_norm,0)
features_norm = (features_norm-temp_min)/(temp_max-temp_min)

# run_range = [0, 1, 2, 3]
run_range = [0, 2] #Which targets are being trained to?


seed_start = 0
seed_stop = 100
seeds = range(seed_start, seed_stop)

# seeds = [49]

for seed_loop in seeds:
    my_seed = seed_loop
    PCC_CSV = open("models/"+CSV_NAME,"a")
    PCC_CSV.write("\nFolder,BatchSize,Target,TrainLoss,ValLoss,TestLoss,TrainPCC,ValPCC,TestPCC,BestEpoch,LRate,BestTestPCC,Seed,Activation,BestDistEpoch,BDETrainLoss,BDEValLoss,BDETestLoss,BDETrainPCC,BDEValPCC,BDETestPCC,BestDist\n")
    PCC_CSV.close()
    for run_loop in run_range:
        # Make deterministic
        torch.manual_seed(my_seed)
        torch.cuda.manual_seed(my_seed)
        np.random.seed(my_seed)
        random.seed(my_seed)
        torch.backends.cudnn.deterministic=True
        print("Seed loop: ", seed_loop+1, "/", 100, "RUN LOOP: ",run_loop+1, "/", len(run_range))
        # training_target = run_loop % 2 #Alternate between Mean and Median
        training_target = run_loop #Alternate between Mean and Median

        new_folder_name = 'models/FCN/'+str(datetime.datetime.now())[:19].replace(" ","_").replace(":","-")+fname
        PCC_CSV = open("models/"+CSV_NAME,"a")
        PCC_CSV.write(new_folder_name)
        PCC_CSV.write(",")
        PCC_CSV.write(str(training_batch_size))
        PCC_CSV.write(",")
        PCC_CSV.close()
        if not os.path.exists(new_folder_name): os.makedirs(new_folder_name) # create log directory.
        if not os.path.exists(new_folder_name+'/Models'): os.makedirs(new_folder_name+'/Models')
        PATH = new_folder_name+'/Models/OMOQ'




        print("Saving Normalising Variables")
        with open(new_folder_name + '/Norm_Vals.p', 'wb') as f:
        # with open('data/Norm_Vals.p', 'wb') as f:
            pickle.dump((temp_mean, temp_std, temp_min, temp_max), f)
        # sys.exit()
        # Scale the lables to 0-1
        Train_Targets = features['MOVs'][0:(5280+88),training_target]
        Test_Targets = features['MOVs'][(5280+88):,training_target]
        Train_Targets = (Train_Targets-1)/4
        Test_Targets = (Test_Targets-1)/4

        # Create the input and target features.
        Train_Input_MOVs = torch.from_numpy(features_norm[0:(5280+88),:]).float()  #Convert to float so final type is torch.FloatTensor
        Train_Target_MOVs = torch.from_numpy(Train_Targets).float()  #Convert to float so final type is torch.FloatTensor
        Test_Input_MOVs = torch.from_numpy(features_norm[(5280+88):,:]).float()  # All Test Set
        Test_Target_MOVs = torch.from_numpy(Test_Targets).float()  # All Test Set

        print('Defining the Fully Connected Network')
        class Net(nn.Module):
            def __init__(self, nodes=128):
                super(Net, self).__init__()
                self.fc1 = nn.Linear(np.ma.size(chosen_features)-feature_start, nodes)
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
        ## PEAQ Basic Network
        # class Net(nn.Module):
        #     def __init__(self):
        #         super(Net, self).__init__()
        #         self.fc1 = nn.Linear(np.ma.size(chosen_features)-feature_start, 3)
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

        # Setup the optimizer
        print('Setup the optimizer')
        optimizer = optim.AdamW(net.parameters(), lr=learning_rate)
        # criterion = nn.L1Loss() # Mean Absolute Error Loss
        # loss_type = 'MAE'
        criterion = nn.MSELoss() # Mean Squared Error Loss
        # loss_type = 'MSE'
        loss_type = 'RMSE'

        # Setup the data for training
        print('Setting up training data')
        training_dataset = TensorDataset(Train_Input_MOVs,Train_Target_MOVs)
        # train_set, val_set = torch.utils.data.random_split(training_dataset, [int((5280+88)*(1-val_split)), int((5280+88)*val_split)])
        train_set, val_set = torch.utils.data.random_split(training_dataset, [training_batch_size, val_batch_size])

        train_loader = DataLoader(train_set, training_batch_size, shuffle=True, sampler=None, \
                                                        batch_sampler=None, num_workers=0, collate_fn=None, \
                                                        pin_memory=False, drop_last=False, timeout=0, \
                                                        worker_init_fn=None, multiprocessing_context=None)

        val_loader = DataLoader(val_set, val_batch_size, shuffle=False, sampler=None, \
                                                        batch_sampler=None, num_workers=0, collate_fn=None, \
                                                        pin_memory=False, drop_last=False, timeout=0, \
                                                        worker_init_fn=None, multiprocessing_context=None)


        # Setup the data for testing
        testing_dataset = TensorDataset(Test_Input_MOVs,Test_Target_MOVs)
        test_loader = DataLoader(testing_dataset, testing_batch_size, shuffle=False, sampler=None, \
                                                        batch_sampler=None, num_workers=0, collate_fn=None, \
                                                        pin_memory=False, drop_last=False, timeout=0, \
                                                        worker_init_fn=None, multiprocessing_context=None)



        # Create vector to store loss values after each epoch
        train_loss_vals = np.zeros(epochs)
        val_loss_vals = np.zeros(epochs)
        test_loss_vals = np.zeros(epochs)
        train_pcc_vals = np.zeros(epochs)
        val_pcc_vals = np.zeros(epochs)
        test_pcc_vals = np.zeros(epochs)
        rmse_dist = np.zeros(epochs)
        pcc_dist = np.zeros(epochs)
        overall_dist = np.zeros(epochs)
        val_loss_vel = np.zeros(epochs)-1  #-1 so the gradient continues to increase
        # tr_rvalue = np.zeros(epochs)
        # v_rvalue = np.zeros(epochs)
        # te_rvalue = np.zeros(epochs)
        # tr_intercept = np.zeros(epochs)
        # v_intercept = np.zeros(epochs)
        # te_intercept = np.zeros(epochs)
        # tr_slope = np.zeros(epochs)
        # v_slope = np.zeros(epochs)
        # te_slope = np.zeros(epochs)
        epoch_vals = np.arange(0,epochs,1)
        TRAIN_OMOQ_vals = np.zeros((training_batch_size,epochs))
        TRAIN_SMOQ_vals = np.zeros((training_batch_size,epochs))
        VAL_OMOQ_vals = np.zeros((val_batch_size,epochs))
        VAL_SMOQ_vals = np.zeros((val_batch_size,epochs))
        TEST_OMOQ_vals = np.zeros((testing_batch_size,epochs))
        TEST_SMOQ_vals = np.zeros((testing_batch_size,epochs))


        print('Training the Network')
        best_val_loss = 1
        best_overall_dist = 10
        best_epoch = 0
        # best_pcc_mad = 1
        down_count = 0
        early_stop = 0
        stopped_epoch = 0
        net = net.train()
        # Train the network
        for epoch in range(epochs):
            #Train the Network
            saved = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                train_net_out = net(data, 1)
                loss = torch.sqrt(criterion(train_net_out.view(training_batch_size), target))
                loss.backward()
                optimizer.step()

        #Save the loss into an array for plotting
            # train_loss_vals[epoch] = torch.Tensor.item(loss.data)
        # Test the network after each epoch
            val_loss = 0
            TRAIN_OMOQ_vals[:,epoch] = np.add(np.multiply(torch.Tensor.tolist(train_net_out.view(training_batch_size)),4),1)
            TRAIN_SMOQ_vals[:,epoch] = np.add(np.multiply(torch.Tensor.tolist(target),4),1)
            train_pcc = np.corrcoef(TRAIN_OMOQ_vals[:,epoch],TRAIN_SMOQ_vals[:,epoch])
            train_pcc_vals[epoch] = train_pcc[0,1]
            # train_loss_vals[epoch] = loss
            train_loss_vals[epoch] = np.sqrt(np.square(np.subtract(TRAIN_OMOQ_vals[:,epoch], TRAIN_SMOQ_vals[:,epoch])).mean()) #RMSE
            # tr_slope[epoch], tr_intercept[epoch], tr_rvalue[epoch], tr_pvalue, stderr = stats.linregress(TRAIN_OMOQ_vals[:,epoch],TRAIN_SMOQ_vals[:,epoch])

            net = net.eval()
            for data, target in val_loader:
                val_net_out = net(data, 0)
                # Store Objective and Subjective values
                VAL_OMOQ_vals[:,epoch] = np.add(np.multiply(torch.Tensor.tolist(val_net_out.view(val_batch_size)),4),1)
                VAL_SMOQ_vals[:,epoch] = np.add(np.multiply(torch.Tensor.tolist(target),4),1)
                val_loss = torch.sqrt(criterion(val_net_out.view(val_batch_size), target))

            # val_loss_vals[epoch] = torch.Tensor.item(val_loss.data)
            val_pcc = np.corrcoef(VAL_OMOQ_vals[:,epoch],VAL_SMOQ_vals[:,epoch])
            val_pcc_vals[epoch] = val_pcc[0,1]
            # val_loss_vals[epoch] = val_loss
            val_loss_vals[epoch] = np.sqrt(np.square(np.subtract(VAL_OMOQ_vals[:,epoch], VAL_SMOQ_vals[:,epoch])).mean()) #RMSE
            # v_slope[epoch], v_intercept[epoch], v_rvalue[epoch], v_pvalue, stderr = stats.linregress(VAL_OMOQ_vals[:,epoch],VAL_SMOQ_vals[:,epoch])

            #Find the PCC and values for the Test set AT THE END NORMALLY
            for data, target in test_loader:
                test_net_out = net(data, 0)
                # Store Objective and Subjective values
                TEST_OMOQ_vals[:,epoch] = np.add(np.multiply(torch.Tensor.tolist(test_net_out.view(testing_batch_size)),4),1)
                TEST_SMOQ_vals[:,epoch] = np.add(np.multiply(torch.Tensor.tolist(target),4),1)
                # sum up batch loss
                test_loss = torch.sqrt(criterion(test_net_out.view(testing_batch_size), target))
            #Average the batch loss
            # test_loss_vals[epoch] = torch.Tensor.item(test_loss.data)
            test_pcc = np.corrcoef(TEST_OMOQ_vals[:,epoch],TEST_SMOQ_vals[:,epoch])
            test_pcc_vals[epoch] = test_pcc[0,1]
            # test_loss_vals[epoch] = test_loss
            test_loss_vals[epoch] = np.sqrt(np.square(np.subtract(TEST_OMOQ_vals[:,epoch], TEST_SMOQ_vals[:,epoch])).mean()) #RMSE
            # te_slope[epoch], te_intercept[epoch], te_rvalue[epoch], te_pvalue, stderr = stats.linregress(VAL_OMOQ_vals[:,epoch],VAL_SMOQ_vals[:,epoch])
            rmse_dist_mean = np.mean((train_loss_vals[epoch],val_loss_vals[epoch],test_loss_vals[epoch]))
            rmse_dist_spread = max((train_loss_vals[epoch],val_loss_vals[epoch],test_loss_vals[epoch]))-min((train_loss_vals[epoch],val_loss_vals[epoch],test_loss_vals[epoch]))
            rmse_dist[epoch] = np.sqrt((np.square(rmse_dist_mean)+np.square(rmse_dist_spread)))

            pcc_dist_mean = np.mean((train_pcc_vals[epoch],val_pcc_vals[epoch],test_pcc_vals[epoch]))
            pcc_dist_spread = max((train_pcc_vals[epoch],val_pcc_vals[epoch],test_pcc_vals[epoch]))-min((train_pcc_vals[epoch],val_pcc_vals[epoch],test_pcc_vals[epoch]))
            pcc_dist[epoch] = np.sqrt((np.square((1-pcc_dist_mean))+np.square(pcc_dist_spread)))

            overall_dist[epoch] = np.sqrt((np.square(rmse_dist[epoch])+np.square(pcc_dist[epoch])))


            print('Epoch: {} \tTrain RMSE: {:.6f}'.format(epoch, train_loss_vals[epoch]),end='')
            print('\tVal RMSE: {:.6f}'.format(val_loss_vals[epoch]),end='')
            print('\tTest RMSE: {:.6f}'.format(test_loss_vals[epoch]),end='')
            print('\tTrain PCC: {:.6f}'.format(train_pcc[0,1]),end='')
            print('\tVal PCC: {:.6f}'.format(val_pcc[0,1]),end='')
            print('\tTest PCC: {:.6f}'.format(test_pcc[0,1]),end='')
            print('\tDist: {:.6f}'.format(overall_dist[epoch]))
            #If saving all epochs, uncomment the next line
            # torch.save(net.state_dict(), PATH + '_' + str(epoch) + '.pth')
            if val_loss < best_val_loss:
                print("Saving Val Model\n")
                best_epoch = epoch
                best_val_loss = val_loss
                # torch.save(net.state_dict(), PATH + '_' + str(epoch) + '.pth')
                # saved = 1
            if overall_dist[epoch] < best_overall_dist and saved == 0:
                print("Saving Dist Model\n")
                # best_epoch = epoch
                best_overall_dist = overall_dist[epoch]
                # torch.save(net.state_dict(), PATH + '_' + str(epoch) + '.pth')
                torch.save(net.state_dict(), PATH + '.pth')
            if epoch > start_early_stopping and stopped_epoch == 0 :
                val_loss_vel[epoch] = np.mean(np.subtract(val_loss_vals[epoch-stopping_range:epoch],val_loss_vals[epoch-stopping_range-1:epoch-1]))
                grad = np.mean(val_loss_vel[epoch-averaging_range:epoch])
                print("Gradient Stopping value: ",grad)
                if grad > early_stop_thresh and np.isnan(grad) == 0:
                    print("Early Stopping")
                    stopped_epoch = best_epoch
                    # early_stop = 1
                    # break

        if stopped_epoch == 0:
            stopped_epoch = best_epoch
        else:
            best_epoch = stopped_epoch
        best_dist_epoch = np.argmin(overall_dist)
        # Saving values out if early stop condition isn't met

        PCC_CSV = open("models/"+CSV_NAME,"a")
        PCC_CSV.write(features['OMOV'][0][training_target][0])
        PCC_CSV.write(",")
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
        PCC_CSV.write("Relu")
        PCC_CSV.write(",")
        PCC_CSV.write(str(best_dist_epoch))
        PCC_CSV.write(",")
        PCC_CSV.write(str(train_loss_vals[best_dist_epoch]))
        PCC_CSV.write(",")
        PCC_CSV.write(str(val_loss_vals[best_dist_epoch]))
        PCC_CSV.write(",")
        PCC_CSV.write(str(test_loss_vals[best_dist_epoch]))
        PCC_CSV.write(",")
        PCC_CSV.write(str(train_pcc_vals[best_dist_epoch]))
        PCC_CSV.write(",")
        PCC_CSV.write(str(val_pcc_vals[best_dist_epoch]))
        PCC_CSV.write(",")
        PCC_CSV.write(str(test_pcc_vals[best_dist_epoch]))
        PCC_CSV.write(",")
        PCC_CSV.write(str(overall_dist[best_dist_epoch])+"\n")
        PCC_CSV.close()

        # Write out the predicted values at the best epoch for the training set.
        with open(new_folder_name + "/Train.csv", "a") as results: results.write("Training Target, SMOS, OMOS\n")
        with open(new_folder_name + "/Train.csv", "a") as results:
            for n in range(training_batch_size):
                results.write(features['OMOV'][0][training_target][0])
                results.write(", ")
                results.write(str(TRAIN_SMOQ_vals[n,best_dist_epoch]))
                results.write(", ")
                results.write(str(TRAIN_OMOQ_vals[n,best_dist_epoch])+ "\n")

        # Write out the predicted values at the best epoch for the training set.
        with open(new_folder_name + "/Val.csv", "a") as results: results.write("Training Target, SMOS, OMOS\n")
        with open(new_folder_name + "/Val.csv", "a") as results:
            for n in range(val_batch_size):
                results.write(features['OMOV'][0][training_target][0])
                results.write(", ")
                results.write(str(VAL_SMOQ_vals[n,best_dist_epoch]))
                results.write(", ")
                results.write(str(VAL_OMOQ_vals[n,best_dist_epoch])+ "\n")

        # Write out the predicted values at the best epoch for the testing set.
        with open(new_folder_name + "/Test.csv", "a") as results: results.write("Training Target, SMOS, OMOS\n")
        with open(new_folder_name + "/Test.csv", "a") as results:
            for n in range(testing_batch_size):
                results.write(features['OMOV'][0][training_target][0])
                results.write(", ")
                results.write(str(TEST_SMOQ_vals[n,best_dist_epoch]))
                results.write(", ")
                results.write(str(TEST_OMOQ_vals[n,best_dist_epoch])+ "\n")

        with open(new_folder_name + "/Dist.csv", "a") as results: results.write("Loss tr, Loss val, Loss te, PCC tr, PCC val, PCC te, r dist, pcc dist, overall dist, loss_grad\n")
        with open(new_folder_name + "/Dist.csv", "a") as results:
            for n in range(epochs):
                results.write(str(train_loss_vals[n]))
                results.write(", ")
                results.write(str(val_loss_vals[n]))
                results.write(", ")
                results.write(str(test_loss_vals[n]))
                results.write(", ")
                results.write(str(train_pcc_vals[n]))
                results.write(", ")
                results.write(str(val_pcc_vals[n]))
                results.write(", ")
                results.write(str(test_pcc_vals[n]))
                results.write(", ")
                results.write(str(rmse_dist[n]))
                results.write(", ")
                results.write(str(pcc_dist[n]))
                results.write(", ")
                results.write(str(overall_dist[n]))
                results.write(", ")
                results.write(str(val_loss_vel[n])+ "\n")



        # best_epoch = np.argmax(test_pcc_vals)

        file1 = open(new_folder_name + "/Network.txt","a")
        file1.write("\n")
        file1.write("FCN with input normalisation to training data\n")
        file1.write("\n")
        file1.write("My Seed = ")
        file1.write(str(my_seed))
        file1.write("\n")
        file1.write("Feature File = ")
        file1.write(load_file)
        file1.write("\n")
        file1.write("Enable Early Stopping Threshold = ")
        file1.write(str(start_early_stopping))
        file1.write("\n")
        file1.write("Early Stopping Threshold = ")
        file1.write(str(early_stop_thresh))
        file1.write("\n")
        file1.write("Batch Size = ")
        file1.write(str(training_batch_size))
        file1.write("\n")
        file1.write("Which Features = \n")
        file1.write(str(chosen_features))
        file1.write("\n")
        file1.write("Training target = ")
        file1.write(features['OMOV'][0][training_target][0])
        file1.write("\n")
        file1.write("Epochs = ")
        file1.write(str(epochs))
        file1.write("\n")
        file1.write("Training Loss Type = ")
        file1.write(loss_type)
        file1.write("\n")
        file1.write("Reporting Loss Type = RMSE\n")
        file1.write("Dropout = ")
        file1.write(str(dropout_per))
        file1.write("\n")
        file1.write("Residual Connections = TRUE")
        file1.write("\n")
        file1.write("Activation Function: Relu")
        file1.write("\n")
        file1.write("Final Training Loss (RMSE) = ")
        file1.write(str(train_loss_vals[-1]))
        file1.write("\n")
        file1.write("Final Validation Loss (RMSE) = ")
        file1.write(str(val_loss_vals[-1]))
        file1.write("\n")
        file1.write("Final Testing Loss (RMSE) = ")
        file1.write(str(test_loss_vals[-1]))
        file1.write("\n")

        file1.write("Final Training PCC = ")
        file1.write(str(train_pcc_vals[-1]))
        file1.write("\n")
        file1.write("Final Validation PCC = ")
        file1.write(str(val_pcc_vals[-1]))
        file1.write("\n")
        file1.write("Final Testing PCC = ")
        file1.write(str(test_pcc_vals[-1]))
        file1.write("\n")

        file1.write("Best Epoch = ")
        file1.write(str(best_epoch))
        file1.write("\n")
        file1.write("Maximum Training PCC = ")
        file1.write(str(np.max(train_pcc_vals)))
        file1.write("\n")
        file1.write("Maximum Validation PCC = ")
        file1.write(str(np.max(val_pcc_vals)))
        file1.write("\n")
        file1.write("Maximum Testing PCC = ")
        file1.write(str(np.max(test_pcc_vals)))
        file1.write("\n")

        file1.write("Training Loss (RMSE) at best epoch = ")
        file1.write(str(train_loss_vals[best_epoch]))
        file1.write("\n")
        file1.write("Validation Loss (RMSE) at best epoch = ")
        file1.write(str(val_loss_vals[best_epoch]))
        file1.write("\n")
        file1.write("Testing Loss at best epoch = ")
        file1.write(str(test_loss_vals[best_epoch]))
        file1.write("\n")

        file1.write("Training PCC at best epoch = ")
        file1.write(str(train_pcc_vals[best_epoch]))
        file1.write("\n")
        file1.write("Validation PCC at best epoch = ")
        file1.write(str(val_pcc_vals[best_epoch]))
        file1.write("\n")
        file1.write("Testing PCC at best epoch = ")
        file1.write(str(test_pcc_vals[best_epoch]))
        file1.write("\n")
        file1.write("Best Distance Epoch = ")
        file1.write(str(best_dist_epoch))
        file1.write("\n")
        file1.close()

        # Plot all the things
        print("Plotting Loss and PCC across Epochs")
        plt.figure()
        line_x = [best_dist_epoch, best_dist_epoch]
        line_y = [0, 1]
        plt.plot(epoch_vals,train_loss_vals)
        plt.plot(epoch_vals,val_loss_vals)
        plt.plot(epoch_vals,test_loss_vals)
        plt.plot(epoch_vals,train_pcc_vals)
        plt.plot(epoch_vals,val_pcc_vals)
        plt.plot(epoch_vals,test_pcc_vals)
        # plt.plot(epoch_vals,tr_intercept)
        # plt.plot(epoch_vals,v_intercept)
        # plt.plot(epoch_vals,te_intercept)
        # plt.plot(epoch_vals,tr_rvalue)
        # plt.plot(epoch_vals,v_rvalue)
        # plt.plot(epoch_vals,te_rvalue)
        plt.plot(line_x,line_y,'r--', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('RMSE Loss and PCC')
        # plt.legend(['Training Loss','Validation Loss','Testing Loss', 'Training PCC', 'Validation PCC', 'Testing PCC', 'Training Intercept', 'Validation Intercept', 'Testing Intercept', 'Training R^2', 'Validation R^2', 'Testing R^2'], loc='best')
        plt.legend(['Training Loss','Validation Loss','Testing Loss', 'Training PCC', 'Validation PCC', 'Testing PCC', 'Best Epoch'], loc='best')
        plt.savefig(new_folder_name + '/Loss_FCN.png',dpi=300,format='png')
        plt.savefig(new_folder_name + '/Loss_FCN.eps',dpi=300,format='eps')


        print("Plotting Loss and PCC across Epochs")
        plt.figure()
        line_x = [best_dist_epoch, best_dist_epoch]
        line_y = [np.min((np.min(val_loss_vals),np.min(test_loss_vals))), np.max((np.max(val_loss_vals),np.max(test_loss_vals)))]
        plt.plot(epoch_vals,val_loss_vals)
        plt.plot(epoch_vals,test_loss_vals)
        plt.plot(line_x,line_y,'r--', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('RMSE Loss and PCC')
        plt.legend(['Validation Loss','Testing Loss', 'Best Epoch'], loc='best')
        plt.savefig(new_folder_name + '/Val_and_Test_Loss_FCN.png',dpi=300,format='png')
        plt.savefig(new_folder_name + '/Val_and_Test_Loss_FCN.eps',dpi=300,format='eps')


        print("Plotting Training Set Confusion Matrix at Best Epoch")
        line_x = [1, 5]
        line_y = line_x
        plt.figure()
        plt.hist2d(TRAIN_SMOQ_vals[:,best_epoch],TRAIN_OMOQ_vals[:,best_epoch], bins=40, range = [[0, 6],[0, 6]])
        cb = plt.colorbar()
        cb.set_label('Count')
        plt.plot(line_x,line_y,'r--', linewidth=2)
        # plt.plot([0, 1], [te_intercept[best_epoch], te_intercept[best_epoch] + te_slope[best_epoch]*1], 'g:')
        plt.xlabel('Subjective Target')
        plt.ylabel('Objective Estimate')
        plt.title('Training Set Confusion Matrix at Best Epoch')
        plt.savefig(new_folder_name + '/Subj_vs_Obj_Train.png',dpi=300,format='png')
        plt.savefig(new_folder_name + '/Subj_vs_Obj_Train.eps',dpi=300,format='eps')

        print("Plotting Validation Set Confusion Matrix at Best Epoch")
        line_x = [1, 5]
        line_y = line_x
        plt.figure()
        plt.hist2d(VAL_SMOQ_vals[:,best_epoch],VAL_OMOQ_vals[:,best_epoch], bins=40, range = [[0, 6],[0, 6]])
        cb = plt.colorbar()
        cb.set_label('Count')
        plt.plot(line_x,line_y,'r--', linewidth=2)
        # plt.plot([0, 1], [te_intercept[best_epoch], te_intercept[best_epoch] + te_slope[best_epoch]*1], 'g:')
        plt.xlabel('Subjective Target')
        plt.ylabel('Objective Estimate')
        plt.title('Validation Set Confusion Matrix at Best Epoch')
        plt.savefig(new_folder_name + '/Subj_vs_Obj_Val.png',dpi=300,format='png')
        plt.savefig(new_folder_name + '/Subj_vs_Obj_Val.eps',dpi=300,format='eps')

        print("Plotting Testing Set Confusion Matrix at Best Epoch")
        line_x = [1, 5]
        line_y = line_x
        plt.figure()
        plt.hist2d(TEST_SMOQ_vals[:,best_epoch],TEST_OMOQ_vals[:,best_epoch], bins=40, range = [[0, 6],[0, 6]])
        cb = plt.colorbar()
        cb.set_label('Count')
        plt.plot(line_x,line_y,'r--', linewidth=2)
        # plt.plot([0, 1], [te_intercept[best_epoch], te_intercept[best_epoch] + te_slope[best_epoch]*1], 'g:')
        plt.xlabel('Subjective Target')
        plt.ylabel('Objective Estimate')
        plt.title('Testing Set Confusion Matrix at Best Epoch')
        plt.savefig(new_folder_name + '/Subj_vs_Obj_Test.png',dpi=300,format='png')
        plt.savefig(new_folder_name + '/Subj_vs_Obj_Test.eps',dpi=300,format='eps')

        print("Plotting Training Set Confusion Matrix at Best Dist Epoch")
        line_x = [1, 5]
        line_y = line_x
        plt.figure()
        plt.hist2d(TRAIN_SMOQ_vals[:,best_dist_epoch],TRAIN_OMOQ_vals[:,best_dist_epoch], bins=40, range = [[0, 6],[0, 6]])
        cb = plt.colorbar()
        cb.set_label('Count')
        plt.plot(line_x,line_y,'r--', linewidth=2)
        # plt.plot([0, 1], [te_intercept[best_epoch], te_intercept[best_epoch] + te_slope[best_epoch]*1], 'g:')
        plt.xlabel('Subjective Target')
        plt.ylabel('Objective Estimate')
        plt.title('Training Set Confusion Matrix at Best Dist Epoch')
        plt.savefig(new_folder_name + '/Subj_vs_Obj_Train_Best_Dist.png',dpi=300,format='png')
        plt.savefig(new_folder_name + '/Subj_vs_Obj_Train_Best_Dist.eps',dpi=300,format='eps')

        print("Plotting Validation Set Confusion Matrix at Best Dist Epoch")
        line_x = [1, 5]
        line_y = line_x
        plt.figure()
        plt.hist2d(VAL_SMOQ_vals[:,best_dist_epoch],VAL_OMOQ_vals[:,best_dist_epoch], bins=40, range = [[0, 6],[0, 6]])
        cb = plt.colorbar()
        cb.set_label('Count')
        plt.plot(line_x,line_y,'r--', linewidth=2)
        # plt.plot([0, 1], [te_intercept[best_epoch], te_intercept[best_epoch] + te_slope[best_epoch]*1], 'g:')
        plt.xlabel('Subjective Target')
        plt.ylabel('Objective Estimate')
        plt.title('Validation Set Confusion Matrix at Best Dist Epoch')
        plt.savefig(new_folder_name + '/Subj_vs_Obj_Val_Best_Dist.png',dpi=300,format='png')
        plt.savefig(new_folder_name + '/Subj_vs_Obj_Val_Best_Dist.eps',dpi=300,format='eps')

        print("Plotting Testing Set Confusion Matrix at Best Dist Epoch")
        line_x = [1, 5]
        line_y = line_x
        plt.figure()
        plt.hist2d(TEST_SMOQ_vals[:,best_dist_epoch],TEST_OMOQ_vals[:,best_dist_epoch], bins=40, range = [[0, 6],[0, 6]])
        cb = plt.colorbar()
        cb.set_label('Count')
        plt.plot(line_x,line_y,'r--', linewidth=2)
        # plt.plot([0, 1], [te_intercept[best_epoch], te_intercept[best_epoch] + te_slope[best_epoch]*1], 'g:')
        plt.xlabel('Subjective Target')
        plt.ylabel('Objective Estimate')
        plt.title('Testing Set Confusion Matrix at Best Dist Epoch')
        plt.savefig(new_folder_name + '/Subj_vs_Obj_Test_Best_Dist.png',dpi=300,format='png')
        plt.savefig(new_folder_name + '/Subj_vs_Obj_Test_Best_Dist.eps',dpi=300,format='eps')

        plt.close('all')

        #Save Variables
        print("Saving Variables")
        with open(new_folder_name + '/FCN_Vals.p', 'wb') as f:
            pickle.dump((train_loss_vals, val_loss_vals, test_loss_vals, train_pcc_vals, val_pcc_vals, test_pcc_vals, overall_dist), f)


#If you want to automatically send an email, uncomment and set the appropriate variables
#If batching processing, this only sends the best distance of the last run.
print("Sending Email")
port = 465  # For SSL
smtp_server = ""  # Enter the smtp address of your server
sender_email = ""  # Enter your address
receiver_email = ""  # Enter receiver address
# password = input("Type your password and press enter: ")
password = ""
message = """\
From: <email address goes here>

    Subject: FCN Processing Complete

    To: <email address goes here>

    The {} features processing has finished.
    With a best overall distance of {}""".format(fname,best_overall_dist)
context = ssl.create_default_context()
with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
    server.login(sender_email, password)
    server.sendmail(sender_email, receiver_email, message)
print("Email Sent")
