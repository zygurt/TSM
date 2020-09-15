# Initial testing with the OMOQ dataset
# Objective Measure of Quality, GRU RNN and Frame Target
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

PADDING_VALUE=0
verb = 0
learning_rate = 1e-4 #Have tried -2     to -6

num_dir = 2 # use 2 for Bidirectional
num_layers = 2
hidden_size = 256
input_size = 128*2 #double the standard feature depth
epochs = 40
log_interval = 50
trainbs = 48 #training_batch_size #factor of number of training files (Tried 1 to 5280) 48 is gcd(4782,582,240)
valbs = 48
testbs = 48
val_batch_size = 1
dropout_per = 0.1 #Best is 0.1 so far, 0.5 is too much

test_averaging = 16 #Number of times to sample signal in testing.
load_folder = "./data/Features/MFCC_Delta_Delta_NoNorm_Trim_Source/"
# load_folder = "./data/Features/MFCC_Lib/"
# fname = 'GRU_MFCCs_Delta_FT_ReRun_' + str(hidden_size)
# load_folder = "./data/Features/MFCC_256/"
# fname = 'GRU_' +str(num_layers) + '_MFCC_Lib_128_FT_2conv1d_median_' + str(hidden_size)
fname = 'GRU_' +str(num_layers) + '_MFCC_Lib_128_FT_Testing_' + str(hidden_size)
if (num_dir == 2):
    fname = 'B'+fname
    print("fname: " + fname)
else:
    print("fname: " + fname)
CSV_NAME = fname+".csv"

seed_start = 27
seed_stop = 30
seeds = range(seed_start, seed_stop)
best_seed = 0;
best_pcc_overall = 0;
best_loss_overall = 5;

for seed_loop in seeds:
    my_seed = seed_loop
    best_pcc = 0.2
    best_dist = 5
    best_mean_pcc = 0.2


    # Make deterministic
    torch.manual_seed(my_seed)
    torch.cuda.manual_seed(my_seed)
    np.random.seed(my_seed)
    random.seed(my_seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark = False
    print("Seed loop: ", seed_loop+1, "/", seed_stop)
    # Create folder for results
    new_folder_name = 'models/GRU/'+str(datetime.datetime.now())[:19].replace(" ","_").replace(":","-")
    if not os.path.exists(new_folder_name): os.makedirs(new_folder_name) # create log directory.

    PATH = new_folder_name+'/OMOQ.pth'
    Mean_PATH = new_folder_name+'/MeanPCC_OMOQ.pth'

    load_file = "Feat.p"




    # Load the dataset
    print('Loading Dataset')
    with open(load_folder+load_file, 'rb') as f:
        train_list, val_list, test_list, Norm_vals= pickle.load(f)

    # with open(load_folder+'Feat_Source.p', 'rb') as f:
    #     source_list = pickle.load(f)

    # print(len(train_list))
    # print(len(val_list))
    # print(len(test_list))
    # print(Norm_vals)
    # sys.exit()
    # training_mean = np.mean(features['MOVs'][:,4:],0)
    # training_std = np.std(features['MOVs'][:,4:],0)
    # features_norm = (features['MOVs'][:,4:]-training_mean)/training_std

    print('Setting up training data')
    training_dataset = OMOQ.OMOQDatasetGRU(train_list)
    train_loader = DataLoader(training_dataset, trainbs, shuffle=True, sampler=None, \
                                                        batch_sampler=None, num_workers=0, \
                                                        collate_fn=OMOQ.pad_collate_fn, \
                                                        pin_memory=False, drop_last=False, timeout=0, \
                                                        worker_init_fn=None, multiprocessing_context=None)
    print('Setting up validation data')
    validation_dataset = OMOQ.OMOQDatasetGRU(val_list)
    validation_loader = DataLoader(validation_dataset, valbs, shuffle=False, sampler=None, \
                                                        batch_sampler=None, num_workers=0, \
                                                        collate_fn=OMOQ.pad_collate_fn, \
                                                        pin_memory=False, drop_last=False, timeout=0, \
                                                        worker_init_fn=None, multiprocessing_context=None)

    # print("Training Loader: ", train_loader)
    # sys.exit()

    # Setup the data for training
    print('Setting up testing data')
    # Setup the data for testing
    testing_dataset = OMOQ.OMOQDatasetGRU(test_list)
    # testing_dataset = TensorDataset(torch.as_tensor(temp_data),torch.from_numpy(temp_mos))#,torch.from_numpy(temp_length))
    test_loader = DataLoader(testing_dataset, testbs, shuffle=False, sampler=None, \
                                                        batch_sampler=None, num_workers=0, \
                                                        collate_fn=OMOQ.pad_collate_fn, \
                                                        pin_memory=False, drop_last=False, timeout=0, \
                                                        worker_init_fn=None, multiprocessing_context=None)

    # print('Setting up Source data')
    # # Setup the data for testing
    # source_dataset = OMOQ.OMOQDatasetLSTM(source_list)
    # # testing_dataset = TensorDataset(torch.as_tensor(temp_data),torch.from_numpy(temp_mos))#,torch.from_numpy(temp_length))
    # source_loader = DataLoader(source_dataset, 1, shuffle=False, sampler=None, \
    #                                                     batch_sampler=None, num_workers=0, \
    #                                                     collate_fn=OMOQ.pad_collate_fn, \
    #                                                     pin_memory=False, drop_last=False, timeout=0, \
    #                                                     worker_init_fn=None, multiprocessing_context=None)
    # # Test the DataLoader
    # for batch_idx, (data, target) in enumerate(test_loader):
    #     print(batch_idx)
    #     print("Data = ", data,"Target = ", target)
    #     sys.exit()




    print('Defining the GRU Network')
    class Net(nn.Module):
        def __init__(self, hidden_dim=256, input_size=256, numb_layers=2, batch=trainbs, dirs=1, drop_per=0.1):
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
            # self.conv1d1 = nn.Conv1d(in_channels=hidden_dim*dirs, out_channels=hidden_dim, kernel_size=1, stride=1, padding=0, dilation=1,
            #                      groups=1, bias=True, padding_mode='zeros') #N (Batch),Cin (Channels/FeatDepth), Lin (Length)
            # self.conv1d2 = nn.Conv1d(in_channels=hidden_dim, out_channels=1, kernel_size=1, stride=1, padding=0, dilation=1,
            #                      groups=1, bias=True, padding_mode='zeros') #N (Batch),Cin (Channels/FeatDepth), Lin (Length)

            self.conv1d1 = nn.Conv1d(in_channels=hidden_dim*dirs, out_channels=1, kernel_size=1, stride=1, padding=0, dilation=1,
                                 groups=1, bias=True, padding_mode='zeros') #N (Batch),Cin (Channels/FeatDepth), Lin (Length)

            # if REGRESSION:
            self.sigmoid = nn.Sigmoid()

        def forward(self, x, seq_lens):
            x = nn.utils.rnn.pack_padded_sequence(x, seq_lens, batch_first=True) # Batch, Len, Dimension
            # print("Packed: " + str(x.shape))
            x, h = self.gru1(x,self.h0) # , batch, seq_len, input_size
            x, y = nn.utils.rnn.pad_packed_sequence(x, batch_first=True, padding_value=PADDING_VALUE)
            # print("Unpacked: " + str(x.shape)) # BLD
            # sys.exit()

            # print(x.data.shape) #
            x = x.permute(0,2,1)
            # print("Before conv1d: " +str(x.shape))
            x = self.conv1d1(x) # Takes B x D x L
            # x = self.conv1d2(x)
            # print("After conv1d: " +str(x.shape)) # B D L
            x = x.permute(0,2,1) #BLD
            x = self.sigmoid(x)

            return x


    #Create the network
    model = Net(numb_layers = num_layers, hidden_dim = hidden_size, input_size=input_size, dirs=num_dir, drop_per = dropout_per)
    model.to(device)
    # Setup the optimizer
    print('Setup the optimizer')
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # criterion = nn.L1Loss() # Mean Absolute Error Loss
    # loss_type = 'MAE'
    criterion = nn.MSELoss() # Mean Squared Error Loss
    loss_type = 'MSE'

    #Create vector to store loss values after each epoch
    train_loss_vals = np.zeros(epochs)
    val_loss_vals = np.zeros(epochs)
    test_loss_vals = np.zeros(epochs)
    source_loss_vals = np.zeros(epochs)
    train_pcc_vals = np.zeros(epochs)
    val_pcc_vals = np.zeros(epochs)
    test_pcc_vals = np.zeros(epochs)
    source_pcc_vals = np.zeros(epochs)
    mean_pcc_vals = np.zeros(epochs)
    loss_dist = np.zeros(epochs)
    pcc_dist = np.zeros(epochs)
    dist = np.zeros(epochs)
    mean_loss = np.zeros(epochs)
    mean_pcc = np.zeros(epochs)
    diff_loss = np.zeros(epochs)
    diff_pcc = np.zeros(epochs)

    train_loss_vals_median = np.zeros(epochs)
    val_loss_vals_median = np.zeros(epochs)
    test_loss_vals_median = np.zeros(epochs)
    source_loss_vals_median = np.zeros(epochs)
    train_pcc_vals_median = np.zeros(epochs)
    val_pcc_vals_median = np.zeros(epochs)
    test_pcc_vals_median = np.zeros(epochs)
    source_pcc_vals_median = np.zeros(epochs)
    mean_pcc_vals_median = np.zeros(epochs)
    loss_dist_median = np.zeros(epochs)
    pcc_dist_median = np.zeros(epochs)
    dist_median = np.zeros(epochs)
    mean_loss_median = np.zeros(epochs)
    mean_pcc_median = np.zeros(epochs)
    diff_loss_median = np.zeros(epochs)
    diff_pcc_median = np.zeros(epochs)

    train_loss_vals_min = np.zeros(epochs)
    val_loss_vals_min = np.zeros(epochs)
    test_loss_vals_min = np.zeros(epochs)
    source_loss_vals_min = np.zeros(epochs)
    train_pcc_vals_min = np.zeros(epochs)
    val_pcc_vals_min = np.zeros(epochs)
    test_pcc_vals_min = np.zeros(epochs)
    source_pcc_vals_min = np.zeros(epochs)
    mean_pcc_vals_min = np.zeros(epochs)
    loss_dist_min = np.zeros(epochs)
    pcc_dist_min = np.zeros(epochs)
    dist_min = np.zeros(epochs)
    mean_loss_min = np.zeros(epochs)
    mean_pcc_min = np.zeros(epochs)
    diff_loss_min = np.zeros(epochs)
    diff_pcc_min = np.zeros(epochs)

    train_loss_vals_max = np.zeros(epochs)
    val_loss_vals_max = np.zeros(epochs)
    test_loss_vals_max = np.zeros(epochs)
    source_loss_vals_max = np.zeros(epochs)
    train_pcc_vals_max = np.zeros(epochs)
    val_pcc_vals_max = np.zeros(epochs)
    test_pcc_vals_max = np.zeros(epochs)
    source_pcc_vals_max = np.zeros(epochs)
    mean_pcc_vals_max = np.zeros(epochs)
    loss_dist_max = np.zeros(epochs)
    pcc_dist_max = np.zeros(epochs)
    dist_max = np.zeros(epochs)
    mean_loss_max = np.zeros(epochs)
    mean_pcc_max = np.zeros(epochs)
    diff_loss_max = np.zeros(epochs)
    diff_pcc_max = np.zeros(epochs)

    epoch_vals = np.arange(0,epochs,1)

    test_loss = np.zeros((test_averaging,1))
    TRAIN_OMOQ_vals = np.zeros((len(train_list), epochs))
    TRAIN_SMOQ_vals = np.zeros((len(train_list), epochs))
    VAL_OMOQ_vals = np.zeros((len(val_list), epochs))
    VAL_SMOQ_vals = np.zeros((len(val_list), epochs))
    TEST_OMOQ_vals = np.zeros((len(test_list), epochs))
    TEST_SMOQ_vals = np.zeros((len(test_list), epochs))

    TRAIN_OMOQ_vals_median = np.zeros((len(train_list), epochs))
    TRAIN_SMOQ_vals_median = np.zeros((len(train_list), epochs))
    VAL_OMOQ_vals_median = np.zeros((len(val_list), epochs))
    VAL_SMOQ_vals_median = np.zeros((len(val_list), epochs))
    TEST_OMOQ_vals_median = np.zeros((len(test_list), epochs))
    TEST_SMOQ_vals_median = np.zeros((len(test_list), epochs))

    TRAIN_OMOQ_vals_min = np.zeros((len(train_list), epochs))
    TRAIN_SMOQ_vals_min = np.zeros((len(train_list), epochs))
    VAL_OMOQ_vals_min = np.zeros((len(val_list), epochs))
    VAL_SMOQ_vals_min = np.zeros((len(val_list), epochs))
    TEST_OMOQ_vals_min = np.zeros((len(test_list), epochs))
    TEST_SMOQ_vals_min = np.zeros((len(test_list), epochs))

    TRAIN_OMOQ_vals_max = np.zeros((len(train_list), epochs))
    TRAIN_SMOQ_vals_max = np.zeros((len(train_list), epochs))
    VAL_OMOQ_vals_max = np.zeros((len(val_list), epochs))
    VAL_SMOQ_vals_max = np.zeros((len(val_list), epochs))
    TEST_OMOQ_vals_max = np.zeros((len(test_list), epochs))
    TEST_SMOQ_vals_max = np.zeros((len(test_list), epochs))
    # SOURCE_OMOQ_vals = np.zeros((len(source_list), epochs))
    # SOURCE_SMOQ_vals = np.zeros((len(source_list), epochs))
    TEST_OMOQ = np.zeros((len(test_list), epochs))
    TEST_SMOQ = np.zeros((len(test_list), epochs))

    print('Training the Network')
    # Train the network
    for epoch in range(epochs):
        for batch_idx, (data, target, L, fnum) in enumerate(tqdm(train_loader)):
            # print("Epoch: "+ str(epoch) +", Batch Index: " + str(batch_idx) + "/"+str(int(5280/trainbs)))
            # Transfer data and target to device
            # print(fnum)

            train_data = data.to(device)
            train_L = L.to(device)
            lens = torch.Tensor.tolist(L.long())
            optimizer.zero_grad()
            model = model.train()
            net_out = model(train_data, train_L)
            #Create Frame Targets
            frame_target = np.zeros((net_out.data.shape[0], net_out.data.shape[1], net_out.data.shape[2])) #batch len dim
            for i in range(net_out.data.shape[0]):
                frame_target[i,0:lens[i],:] = np.array([[target[i]]]*lens[i])#.reshape(1,lens[i])

            FT = torch.Tensor(frame_target).to(device)

            loss = criterion(net_out[FT[:,:,0]!=0.0],FT[FT[:,:,0]!=0.0]) # #
            # print(loss)
            loss.backward()
            optimizer.step()

            #Save the output and target values
            # print(torch.mean(net_out.data,1))
            # print(net_out.data.shape)
            # print(train_target.shape)
            # print(torch.mean(net_out[b,0:lens[i],:]))
            # sys.exit()
            # print(lens)

            for b in range(0,L.shape[0]):
                # print(lens[b])

                # print((np.array(torch.Tensor.item(torch.mean(net_out.data[b,:,0:L[b].long().to(device)])))*4+1))
                # print(np.array(torch.Tensor.item(torch.mean(net_out[b,0:lens[i],:])))*4+1)
                # print(np.array(torch.Tensor.item(FT[b,0,0]))*4+1)
                TRAIN_OMOQ_vals[fnum[b].long(),epoch] = np.array(torch.Tensor.item(torch.mean(net_out[b,0:lens[b],:])))*4+1
                TRAIN_OMOQ_vals_median[fnum[b].long(),epoch] = np.array(torch.Tensor.item(torch.median(net_out[b,0:lens[b],:])))*4+1
                TRAIN_OMOQ_vals_min[fnum[b].long(),epoch] = np.array(torch.Tensor.item(torch.min(net_out[b,0:lens[b],:])))*4+1
                TRAIN_OMOQ_vals_max[fnum[b].long(),epoch] = np.array(torch.Tensor.item(torch.max(net_out[b,0:lens[b],:])))*4+1
                TRAIN_SMOQ_vals[fnum[b].long(),epoch] = (np.array(torch.Tensor.item(FT[b,0,0]))*4+1)
            # sys.exit()
            # print(TRAIN_OMOQ_vals)
            # print(TRAIN_SMOQ_vals)
            # sys.exit()

        train_pcc_vals[epoch] = np.corrcoef(TRAIN_OMOQ_vals[:,epoch],TRAIN_SMOQ_vals[:,epoch])[0][1]
        train_loss_vals[epoch] = np.sqrt(np.square(np.subtract(TRAIN_OMOQ_vals[:,epoch], TRAIN_SMOQ_vals[:,epoch])).mean()) #RMSE
        train_pcc_vals_median[epoch] = np.corrcoef(TRAIN_OMOQ_vals_median[:,epoch],TRAIN_SMOQ_vals[:,epoch])[0][1]
        train_loss_vals_median[epoch] = np.sqrt(np.square(np.subtract(TRAIN_OMOQ_vals_median[:,epoch], TRAIN_SMOQ_vals[:,epoch])).mean()) #RMSE
        train_pcc_vals_min[epoch] = np.corrcoef(TRAIN_OMOQ_vals_min[:,epoch],TRAIN_SMOQ_vals[:,epoch])[0][1]
        train_loss_vals_min[epoch] = np.sqrt(np.square(np.subtract(TRAIN_OMOQ_vals_min[:,epoch], TRAIN_SMOQ_vals[:,epoch])).mean()) #RMSE
        train_pcc_vals_max[epoch] = np.corrcoef(TRAIN_OMOQ_vals_max[:,epoch],TRAIN_SMOQ_vals[:,epoch])[0][1]
        train_loss_vals_max[epoch] = np.sqrt(np.square(np.subtract(TRAIN_OMOQ_vals_max[:,epoch], TRAIN_SMOQ_vals[:,epoch])).mean()) #RMSE
        #Save the loss into an array for plotting
        print('Epoch: {} \tTrain RMSE: {:.6f} \tTrain PCC: {:.6f}'.format(epoch, train_loss_vals[epoch], train_pcc_vals[epoch],end=''))
        print('\t \tMedian RMSE: {:.6f} \tMedian PCC: {:.6f}'.format( train_loss_vals_median[epoch], train_pcc_vals_median[epoch],end=''))
        print('\t \tMin RMSE: {:.6f} \tMin PCC: {:.6f}'.format( train_loss_vals_min[epoch], train_pcc_vals_min[epoch],end=''))
        print('\t \tMax RMSE: {:.6f} \tMax PCC: {:.6f}'.format( train_loss_vals_max[epoch], train_pcc_vals_max[epoch],end=''))
        # sys.exit()
        # Validate the network after each epoch
        for v_idx, (val_data, val_target, val_L, fnum) in enumerate(validation_loader):

            val_data = val_data.to(device)
            val_L = val_L.to(device)
            val_lens = torch.Tensor.tolist(val_L.long())
            val_target = val_target.to(device)
            model = model.eval()
            val_net_out = model(val_data, val_L)
            # print(val_net_out.data.shape)
            # sys.exit()
            # Store Objective and Subjective values
            for b in range(0,val_L.shape[0]):
                # print((np.array(torch.Tensor.item(torch.mean(net_out.data[b,:,0:L[b].long().to(device)])))*4+1))
                VAL_OMOQ_vals[fnum[b].long(),epoch] = (np.array(torch.Tensor.item(torch.mean(val_net_out[b,0:val_lens[b],:])))*4+1)
                VAL_OMOQ_vals_median[fnum[b].long(),epoch] = np.array(torch.Tensor.item(torch.median(val_net_out[b,0:lens[b],:])))*4+1
                VAL_OMOQ_vals_min[fnum[b].long(),epoch] = np.array(torch.Tensor.item(torch.min(val_net_out[b,0:lens[b],:])))*4+1
                VAL_OMOQ_vals_max[fnum[b].long(),epoch] = np.array(torch.Tensor.item(torch.max(val_net_out[b,0:lens[b],:])))*4+1
                VAL_SMOQ_vals[fnum[b].long(),epoch] = (np.array(torch.Tensor.item(val_target[b]))*4+1)
            # VAL_OMOQ_vals[fnum.long(),epoch] = (np.array(torch.Tensor.tolist(torch.mean(val_net_out.data,1)))*4+1).flatten()
            # print((np.array(torch.Tensor.tolist(val_target))*4+1).flatten())
            # VAL_SMOQ_vals[fnum.long(),epoch] = (np.array(torch.Tensor.tolist(val_target))*4+1).flatten()

        val_pcc_vals[epoch] = np.corrcoef(VAL_OMOQ_vals[:,epoch],VAL_SMOQ_vals[:,epoch])[0][1]
        val_loss_vals[epoch] = np.sqrt(np.square(np.subtract(VAL_OMOQ_vals[:,epoch], VAL_SMOQ_vals[:,epoch])).mean()) #RMSE
        print('\t\t Val RMSE: {:.6f}\tVal PCC: {:.6f}'.format(val_loss_vals[epoch], val_pcc_vals[epoch],end=''))
        val_pcc_vals_median[epoch] = np.corrcoef(VAL_OMOQ_vals_median[:,epoch],VAL_SMOQ_vals[:,epoch])[0][1]
        val_loss_vals_median[epoch] = np.sqrt(np.square(np.subtract(VAL_OMOQ_vals_median[:,epoch], VAL_SMOQ_vals[:,epoch])).mean()) #RMSE
        val_pcc_vals_min[epoch] = np.corrcoef(VAL_OMOQ_vals_min[:,epoch],VAL_SMOQ_vals[:,epoch])[0][1]
        val_loss_vals_min[epoch] = np.sqrt(np.square(np.subtract(VAL_OMOQ_vals_min[:,epoch], VAL_SMOQ_vals[:,epoch])).mean()) #RMSE
        val_pcc_vals_max[epoch] = np.corrcoef(VAL_OMOQ_vals_max[:,epoch],VAL_SMOQ_vals[:,epoch])[0][1]
        val_loss_vals_max[epoch] = np.sqrt(np.square(np.subtract(VAL_OMOQ_vals_max[:,epoch], VAL_SMOQ_vals[:,epoch])).mean()) #RMSE
        print('\t \tMedian RMSE: {:.6f} \tMedian PCC: {:.6f}'.format( val_loss_vals_median[epoch], val_pcc_vals_median[epoch],end=''))
        print('\t \tMin RMSE: {:.6f} \tMin PCC: {:.6f}'.format( val_loss_vals_min[epoch], val_pcc_vals_min[epoch],end=''))
        print('\t \tMax RMSE: {:.6f} \tMax PCC: {:.6f}'.format( val_loss_vals_max[epoch], val_pcc_vals_max[epoch],end=''))
        # sys.exit()
        # Test the network after each epoch
        for b_idx, (test_data, test_target, test_L, fnum) in enumerate(test_loader):

            test_data = test_data.to(device)
            # print(test_target)
            test_target = test_target.to(device)
            test_lens = torch.Tensor.tolist(test_L.long())
            test_L = test_L.to(device)
            model = model.eval()
            test_net_out = model(test_data, test_L)
            # Store Objective and Subjective values
            for b in range(0,test_L.shape[0]):
                # if b == 0 and b_idx == 0:
                    # print(test_net_out.data[b,:,:])
                    # print((np.array(torch.Tensor.item(torch.mean(test_net_out.data[b,:,0:test_L[b].long()])))*4+1).flatten())
                TEST_OMOQ_vals[fnum[b].long(),epoch] = (np.array(torch.Tensor.item(torch.mean(test_net_out[b,0:test_lens[b],:])))*4+1)
                TEST_OMOQ_vals_median[fnum[b].long(),epoch] = np.array(torch.Tensor.item(torch.median(test_net_out[b,0:lens[b],:])))*4+1
                TEST_OMOQ_vals_min[fnum[b].long(),epoch] = np.array(torch.Tensor.item(torch.min(test_net_out[b,0:lens[b],:])))*4+1
                TEST_OMOQ_vals_max[fnum[b].long(),epoch] = np.array(torch.Tensor.item(torch.max(test_net_out[b,0:lens[b],:])))*4+1
                TEST_SMOQ_vals[fnum[b].long(),epoch] = (np.array(torch.Tensor.item(test_target[b]))*4+1)
            # TEST_OMOQ_vals[fnum.long(),epoch] = (np.array(torch.Tensor.tolist(torch.mean(test_net_out.data,1)))*4+1).flatten()
            # print((np.array(torch.Tensor.tolist(test_target))*4+1).flatten())
            # TEST_SMOQ_vals[fnum.long(),epoch] = (np.array(torch.Tensor.tolist(test_target))*4+1).flatten()
        # print(TEST_OMOQ_vals[0:8,epoch])
        test_pcc_vals[epoch] = np.corrcoef(TEST_OMOQ_vals[:,epoch],TEST_SMOQ_vals[:,epoch])[0][1]
        test_loss_vals[epoch] = np.sqrt(np.square(np.subtract(TEST_OMOQ_vals[:,epoch], TEST_SMOQ_vals[:,epoch])).mean()) #RMSE
        print('\t\t Test RMSE: {:.6f}\tTest PCC: {:.6f}'.format(test_loss_vals[epoch], test_pcc_vals[epoch],end=''))
        test_pcc_vals_median[epoch] = np.corrcoef(TEST_OMOQ_vals_median[:,epoch],TEST_SMOQ_vals[:,epoch])[0][1]
        test_loss_vals_median[epoch] = np.sqrt(np.square(np.subtract(TEST_OMOQ_vals_median[:,epoch], TEST_SMOQ_vals[:,epoch])).mean()) #RMSE
        test_pcc_vals_min[epoch] = np.corrcoef(TEST_OMOQ_vals_min[:,epoch],TEST_SMOQ_vals[:,epoch])[0][1]
        test_loss_vals_min[epoch] = np.sqrt(np.square(np.subtract(TEST_OMOQ_vals_min[:,epoch], TEST_SMOQ_vals[:,epoch])).mean()) #RMSE
        test_pcc_vals_max[epoch] = np.corrcoef(TEST_OMOQ_vals_max[:,epoch],TEST_SMOQ_vals[:,epoch])[0][1]
        test_loss_vals_max[epoch] = np.sqrt(np.square(np.subtract(TEST_OMOQ_vals_max[:,epoch], TEST_SMOQ_vals[:,epoch])).mean()) #RMSE
        print('\t \tMedian RMSE: {:.6f} \tMedian PCC: {:.6f}'.format( test_loss_vals_median[epoch], test_pcc_vals_median[epoch],end=''))
        print('\t \tMin RMSE: {:.6f} \tMin PCC: {:.6f}'.format( test_loss_vals_min[epoch], test_pcc_vals_min[epoch],end=''))
        print('\t \tMax RMSE: {:.6f} \tMax PCC: {:.6f}'.format( test_loss_vals_max[epoch], test_pcc_vals_max[epoch],end=''))
        # sys.exit()
        # # Test the network after each epoch for source files
        # for b_idx, (source_data, source_target, source_L, fnum) in enumerate(source_loader):
        #
        #     source_data = source_data.to(device)
        #     # print(test_target)
        #     source_target = source_target.to(device)
        #     source_L = source_L.to(device)
        #     model = model.eval()
        #     source_net_out = model(source_data, source_L)
        #     # Store Objective and Subjective values
        #     SOURCE_OMOQ_vals[fnum.long(),epoch] = (np.array(torch.Tensor.tolist(torch.mean(source_net_out.data,1)))*4+1).flatten()
        #     # print((np.array(torch.Tensor.tolist(test_target))*4+1).flatten())
        #     SOURCE_SMOQ_vals[fnum.long(),epoch] = (np.array(torch.Tensor.tolist(source_target))*4+1).flatten()
        #
        # # source_pcc_vals[epoch] = np.corrcoef(SOURCE_OMOQ_vals[:,epoch],SOURCE_SMOQ_vals[:,epoch])[0][1]
        # source_loss_vals[epoch] = np.sqrt(np.square(np.subtract(SOURCE_OMOQ_vals[:,epoch], SOURCE_SMOQ_vals[:,epoch])).mean()) #RMSE
        # print('\t\t Source RMSE: {:.6f}\tSource PCC: {:.6f}'.format(source_loss_vals[epoch], source_pcc_vals[epoch],end=''))


        # exit()
        #Distance Calculation
        mean_loss[epoch] = np.mean((train_loss_vals[epoch],val_loss_vals[epoch],test_loss_vals[epoch]))
        mean_pcc[epoch] = np.mean((train_pcc_vals[epoch],val_pcc_vals[epoch],test_pcc_vals[epoch]))
        diff_loss[epoch] = max((train_loss_vals[epoch],val_loss_vals[epoch],test_loss_vals[epoch]))-min((train_loss_vals[epoch],val_loss_vals[epoch],test_loss_vals[epoch]))
        diff_pcc[epoch] = max((train_pcc_vals[epoch],val_pcc_vals[epoch],test_pcc_vals[epoch]))-min((train_pcc_vals[epoch],val_pcc_vals[epoch],test_pcc_vals[epoch]))
        loss_dist[epoch] = np.sqrt((np.square(mean_loss[epoch])+np.square(diff_loss[epoch])))
        pcc_dist[epoch] = np.sqrt((np.square((1-mean_pcc[epoch]))+np.square(diff_pcc[epoch]))) #1-mean used to have smaller=better for all.
        dist[epoch] = np.sqrt((np.square(loss_dist[epoch])+np.square(pcc_dist[epoch])))
        print('\tDistance: {:.6f}'.format(dist[epoch]))

        mean_loss_median[epoch] = np.mean((train_loss_vals_median[epoch],val_loss_vals_median[epoch],test_loss_vals_median[epoch]))
        mean_pcc_median[epoch] = np.mean((train_pcc_vals_median[epoch],val_pcc_vals_median[epoch],test_pcc_vals_median[epoch]))
        diff_loss_median[epoch] = max((train_loss_vals_median[epoch],val_loss_vals_median[epoch],test_loss_vals_median[epoch]))-min((train_loss_vals_median[epoch],val_loss_vals_median[epoch],test_loss_vals_median[epoch]))
        diff_pcc_median[epoch] = max((train_pcc_vals_median[epoch],val_pcc_vals_median[epoch],test_pcc_vals_median[epoch]))-min((train_pcc_vals_median[epoch],val_pcc_vals_median[epoch],test_pcc_vals_median[epoch]))
        loss_dist_median[epoch] = np.sqrt((np.square(mean_loss_median[epoch])+np.square(diff_loss_median[epoch])))
        pcc_dist_median[epoch] = np.sqrt((np.square((1-mean_pcc_median[epoch]))+np.square(diff_pcc_median[epoch]))) #1-mean used to have smaller=better for all.
        dist_median[epoch] = np.sqrt((np.square(loss_dist_median[epoch])+np.square(pcc_dist_median[epoch])))
        print('\tDistance_median: {:.6f}'.format(dist_median[epoch]))

        mean_loss_min[epoch] = np.mean((train_loss_vals_min[epoch],val_loss_vals_min[epoch],test_loss_vals_min[epoch]))
        mean_pcc_min[epoch] = np.mean((train_pcc_vals_min[epoch],val_pcc_vals_min[epoch],test_pcc_vals_min[epoch]))
        diff_loss_min[epoch] = max((train_loss_vals_min[epoch],val_loss_vals_min[epoch],test_loss_vals_min[epoch]))-min((train_loss_vals_min[epoch],val_loss_vals_min[epoch],test_loss_vals_min[epoch]))
        diff_pcc_min[epoch] = max((train_pcc_vals_min[epoch],val_pcc_vals_min[epoch],test_pcc_vals_min[epoch]))-min((train_pcc_vals_min[epoch],val_pcc_vals_min[epoch],test_pcc_vals_min[epoch]))
        loss_dist_min[epoch] = np.sqrt((np.square(mean_loss_min[epoch])+np.square(diff_loss_min[epoch])))
        pcc_dist_min[epoch] = np.sqrt((np.square((1-mean_pcc_min[epoch]))+np.square(diff_pcc_min[epoch]))) #1-mean used to have smaller=better for all.
        dist_min[epoch] = np.sqrt((np.square(loss_dist_min[epoch])+np.square(pcc_dist_min[epoch])))
        print('\tDistance_min: {:.6f}'.format(dist_min[epoch]))

        mean_loss_max[epoch] = np.mean((train_loss_vals_max[epoch],val_loss_vals_max[epoch],test_loss_vals_max[epoch]))
        mean_pcc_max[epoch] = np.mean((train_pcc_vals_max[epoch],val_pcc_vals_max[epoch],test_pcc_vals_max[epoch]))
        diff_loss_max[epoch] = max((train_loss_vals_max[epoch],val_loss_vals_max[epoch],test_loss_vals_max[epoch]))-min((train_loss_vals_max[epoch],val_loss_vals_max[epoch],test_loss_vals_max[epoch]))
        diff_pcc_max[epoch] = max((train_pcc_vals_max[epoch],val_pcc_vals_max[epoch],test_pcc_vals_max[epoch]))-min((train_pcc_vals_max[epoch],val_pcc_vals_max[epoch],test_pcc_vals_max[epoch]))
        loss_dist_max[epoch] = np.sqrt((np.square(mean_loss_max[epoch])+np.square(diff_loss_max[epoch])))
        pcc_dist_max[epoch] = np.sqrt((np.square((1-mean_pcc_max[epoch]))+np.square(diff_pcc_max[epoch]))) #1-mean used to have smaller=better for all.
        dist_max[epoch] = np.sqrt((np.square(loss_dist_max[epoch])+np.square(pcc_dist_max[epoch])))
        print('\tDistance_max: {:.6f}'.format(dist_max[epoch]))

        # sys.exit()

        if dist[epoch] < best_dist:
            print("Saving Model\n\n")
            best_dist = dist[epoch]
            torch.save(model.state_dict(), PATH)


        # if test_pcc_vals[epoch] > best_pcc:
        #     print("Saving Model\n")
        #     best_pcc = test_pcc_vals[epoch]
        #     torch.save(model.state_dict(), PATH)


    best_epoch = np.argmin(dist)
    PCC_CSV = open("models/"+CSV_NAME,"a")
    PCC_CSV.write("\nFolder,BatchSize,Target,TrainLoss,ValLoss,TestLoss,TrainPCC,ValPCC,TestPCC,BestEpoch,LRate,BestTestPCC,Seed,MeanLoss,MeanPCC,DiffLoss,DiffPCC,Dist\n")
    PCC_CSV.write(new_folder_name)
    PCC_CSV.write(",")
    PCC_CSV.write(str(trainbs))
    PCC_CSV.write(",")
    PCC_CSV.write("Mean")
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
    with open(new_folder_name + "/Test.csv", "a") as results: results.write("Filename, TSM, FileMeanOS, SMOS")
    with open(new_folder_name + "/Test.csv", "a") as results:
        for k in range(0,epochs):
            results.write(", OMOS"+str(k))
        results.write("\n")
        for n in range(len(test_list)):
            #print(test_list[n].get('file'))
            results.write(str(test_list[n].get('MATLAB_loc')))
            results.write(", ")
            results.write(str(test_list[n].get('file')).split('_')[-2])
            results.write(", ")
            results.write(str(test_list[n].get('MeanOS')))
            results.write(", ")
            results.write(str(TEST_SMOQ_vals[n,best_epoch]))
            results.write(", ")
            for k in range(0,epochs):
                results.write(str(TEST_OMOQ_vals[n,k])+ ", ")
            results.write("\n")

    print("Saving log of Validation results\n")
    if not os.path.exists('log'): os.makedirs('log') # create log directory.
    with open(new_folder_name + "/Val.csv", "a") as results: results.write("Filename, TSM, FileMeanOS, SMOS")
    with open(new_folder_name + "/Val.csv", "a") as results:
        for k in range(0,epochs):
            results.write(", OMOS"+str(k))
        results.write("\n")
        for n in range(len(val_list)):
            #print(test_list[n].get('file'))
            results.write(str(val_list[n].get('MATLAB_loc')))
            results.write(", ")
            results.write(str(val_list[n].get('file')).split('_')[-2])
            results.write(", ")
            results.write(str(val_list[n].get('MeanOS')))
            results.write(", ")
            results.write(str(VAL_SMOQ_vals[n,best_epoch]))
            results.write(", ")
            for k in range(0,epochs):
                results.write(str(VAL_OMOQ_vals[n,k])+ ", ")
            results.write("\n")

    print("Saving log of Train results\n")
    if not os.path.exists('log'): os.makedirs('log') # create log directory.
    with open(new_folder_name + "/Train.csv", "a") as results: results.write("Filename, TSM, FileMeanOS, SMOS")
    with open(new_folder_name + "/Train.csv", "a") as results:
        for k in range(0,epochs):
            results.write(", OMOS"+str(k))
        results.write("\n")
        for n in range(len(train_list)):
            # print(test_list[n].get('file'))
            results.write(str(train_list[n].get('MATLAB_loc')))
            results.write(", ")
            results.write(str(train_list[n].get('file')).split('_')[-2])
            results.write(", ")
            results.write(str(train_list[n].get('MeanOS')))
            results.write(", ")
            results.write(str(TRAIN_SMOQ_vals[n,best_epoch]))
            results.write(", ")
            for k in range(0,epochs):
                results.write(str(TRAIN_OMOQ_vals[n,k])+ ", ")
            results.write("\n")

    # print("Saving log of Source results\n")
    # if not os.path.exists('log'): os.makedirs('log') # create log directory.
    # with open(new_folder_name + "/Source.csv", "a") as results: results.write("Filename, TSM, FileMeanOS, SMOS")
    # with open(new_folder_name + "/Source.csv", "a") as results:
    #     for k in range(0,epochs):
    #         results.write(", OMOS"+str(k))
    #     results.write("\n")
    #     for n in range(len(source_list)):
    #         #print(test_list[n].get('file'))
    #         results.write(str(source_list[n].get('file')))
    #         results.write(", ")
    #         results.write('100')
    #         results.write(", ")
    #         results.write(str(source_list[n].get('MeanOS')))
    #         results.write(", ")
    #         results.write(str(SOURCE_SMOQ_vals[n,best_epoch]))
    #         results.write(", ")
    #         for k in range(0,epochs):
    #             results.write(str(SOURCE_OMOQ_vals[n,k])+ ", ")
    #         results.write("\n")

    # torch.load(PATH)


    # #Compare to synthetic testing of val files.
    # load_val_file = "OMOQ_MFCC_Deltas_Val.p"
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
    # file1.write("Normalisation = BatchNorm2d\n")
    # file1.write("4 CNN layers: 16, 32, 64, 32\n")
    # file1.write("Kernels: 3x3")
    # file1.write("3 FCN layers: 3584, 128, 128, 128, 1\n") # Was 10240 for 3 CNN layers
    # file1.write("FCN Residual Connections\n")
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
    file1.write("RNN Hidden Size = ")
    file1.write(str(hidden_size))
    file1.write("\n")
    file1.write("Number GRU layers = ")
    file1.write(str(num_layers))
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

    file1.write("BiDirectional: ")
    file1.write(str(num_dir==2))
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
    plt.ylabel('PCC and RMSE Loss')
    plt.legend(['Training RMSE','Validation RMSE','Testing RMSE', 'Training PCC', 'Validation PCC', 'Testing PCC','Best Epoch'], loc='best')
    save_name = new_folder_name + "/Loss_mean_" + load_file[:-2].replace("/","-") + ".png"
    plt.savefig(save_name,dpi=300,format='png')
    # plt.show()

    #Plot all the things
    plt.figure()
    line_x = [np.argmin(dist_median), np.argmin(dist_median)]
    line_y = [0, 1]
    plt.plot(epoch_vals,train_loss_vals_median)
    plt.plot(epoch_vals,val_loss_vals_median)
    plt.plot(epoch_vals,test_loss_vals_median)
    plt.plot(epoch_vals,train_pcc_vals_median)
    plt.plot(epoch_vals,val_pcc_vals_median)
    plt.plot(epoch_vals,test_pcc_vals_median)
    plt.plot(line_x,line_y,'r--', linewidth=2)
    # plt.plot(epoch_vals,dist)
    plt.xlabel('Epoch')
    plt.ylabel('PCC and RMSE Loss')
    plt.legend(['Training RMSE','Validation RMSE','Testing RMSE', 'Training PCC', 'Validation PCC', 'Testing PCC','Best Epoch'], loc='best')
    save_name = new_folder_name + "/Loss_median_" + load_file[:-2].replace("/","-") + ".png"
    plt.savefig(save_name,dpi=300,format='png')
    # plt.show()

    #Plot all the things
    plt.figure()
    line_x = [np.argmin(dist_min), np.argmin(dist_min)]
    line_y = [0, 1]
    plt.plot(epoch_vals,train_loss_vals_min)
    plt.plot(epoch_vals,val_loss_vals_min)
    plt.plot(epoch_vals,test_loss_vals_min)
    plt.plot(epoch_vals,train_pcc_vals_min)
    plt.plot(epoch_vals,val_pcc_vals_min)
    plt.plot(epoch_vals,test_pcc_vals_min)
    plt.plot(line_x,line_y,'r--', linewidth=2)
    # plt.plot(epoch_vals,dist)
    plt.xlabel('Epoch')
    plt.ylabel('PCC and RMSE Loss')
    plt.legend(['Training RMSE','Validation RMSE','Testing RMSE', 'Training PCC', 'Validation PCC', 'Testing PCC','Best Epoch'], loc='best')
    save_name = new_folder_name + "/Loss_min_" + load_file[:-2].replace("/","-") + ".png"
    plt.savefig(save_name,dpi=300,format='png')
    # plt.show()

    #Plot all the things
    plt.figure()
    line_x = [np.argmin(dist_max), np.argmin(dist_max)]
    line_y = [0, 1]
    plt.plot(epoch_vals,train_loss_vals_max)
    plt.plot(epoch_vals,val_loss_vals_max)
    plt.plot(epoch_vals,test_loss_vals_max)
    plt.plot(epoch_vals,train_pcc_vals_max)
    plt.plot(epoch_vals,val_pcc_vals_max)
    plt.plot(epoch_vals,test_pcc_vals_max)
    plt.plot(line_x,line_y,'r--', linewidth=2)
    # plt.plot(epoch_vals,dist)
    plt.xlabel('Epoch')
    plt.ylabel('PCC and RMSE Loss')
    plt.legend(['Training RMSE','Validation RMSE','Testing RMSE', 'Training PCC', 'Validation PCC', 'Testing PCC','Best Epoch'], loc='best')
    save_name = new_folder_name + "/Loss_max_" + load_file[:-2].replace("/","-") + ".png"
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

    # line_x = [1, 5]
    # line_y = line_x
    # plt.figure()
    # plt.hist2d(TEST_SMOQ_vals[:,-1,best_epoch],TEST_OMOQ_vals[:,-1,best_epoch],bins=40,range = [[0, 6],[0, 6]])
    # cb = plt.colorbar()
    # cb.set_label('Count')
    # plt.plot(line_x,line_y,'r--', linewidth=2)
    # plt.xlabel('Subjective MeanOS')
    # plt.ylabel('Objective MOQ')
    # plt.title('Confusion Matrix at Best Mean PCC Epoch')
    # save_name = new_folder_name + "/Subjective_vs_Objective_MeanPCC_" + load_file[:-2].replace("/","-") + ".png"
    # plt.savefig(save_name,dpi=300,format='png')




# print("Best PCC: ", best_pcc)



#Comment to put the code at a more readable position after saving
