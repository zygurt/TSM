import xgboost as xgb
import pandas as pd
import numpy as np
from tqdm import tqdm
import scipy.fftpack as scifft
import matplotlib.pyplot as plt
import datetime, sys, pickle, os
import scipy.io as sio

training_target = 0

# with open('MOVs.csv','r') as f:
#     MOVs = pd.read_csv(f)
#
# feat_info = [['MOS',1],
#          ['TSM',1],
#          ['WinModDiff1B',1],
#          ['AvgModDiff1B',1],
#          ['AvgModDiff2B',1],
#          ['RmsNoiseLoudB',1],
#          ['BandwidthRefB',1],
#          ['BandwidthTestB',1],
#          ['TotalNMRB',1],
#          ['RelDistFramesB',1],
#          ['MFPDB',1],
#          ['ADBB',1],
#          ['EHSB',1],
#          ['DM',1],
#          ['SER',1],
#         ]

load_file = 'data/MOVs_Final_To_Test_Source_20200416.mat'
features = sio.loadmat(load_file)

# #Code for chosing particular features
# chosen_features = np.concatenate((np.arange(0,21),np.arange(42,49),np.arange(65,69)),axis=0)
# chosen_features = np.arange(0,69)
# print(chosen_features)
# features['MOVs'] = features['MOVs'][:,chosen_features]


# Normalise the input features
#Make sure that normalising is per feature, not for the whole thing. (Checked and all good.)
temp_mean = np.mean(features['MOVs'][0:5280,4:],0)
# print("Temp mean: ", temp_mean)
temp_std = np.std(features['MOVs'][0:5280,4:],0)
features_norm = (features['MOVs'][:,4:]-temp_mean)/temp_std

# Feat_names = features['OMOV']
# print(Feat_names)
# for n in np.arange(4,69):
#     print(features['OMOV'][n])
#     feat[n] = features['OMOV'][n]

val_percent = 0.1
train_order = np.random.permutation(5280)
num_train_samples = int(np.floor(len(train_order)*(1-val_percent)))
num_val_samples = 5280-num_train_samples
num_test_samples = features_norm.shape[0]-5280

# print(num_train_samples)
# print(num_val_samples)
# print(num_test_samples)


train_dat = np.zeros((num_train_samples,features_norm.shape[1]-4))
val_dat = np.zeros((num_val_samples,features_norm.shape[1]-4))
test_dat = np.zeros((num_test_samples,features_norm.shape[1]-4))

train_labels = np.zeros((num_train_samples,1))
val_labels = np.zeros((num_val_samples,1))
test_labels = np.zeros((num_test_samples,1))

for n in range(num_train_samples):
    train_dat[n,:] = features_norm[train_order[n],4:]
    train_labels[n] = features['MOVs'][train_order[n],training_target]

for n in range(num_val_samples):
    val_dat[n,:] = features_norm[train_order[n+num_train_samples],4:]
    val_labels[n] = features['MOVs'][train_order[n+num_train_samples],training_target]
# min_test_label = 1000
# max_test_label = -1000
for n in range(num_test_samples):
    test_dat[n,:] = features_norm[n+5280,4:]
    test_labels[n] = features['MOVs'][n+5280,training_target]
#     if test_labels[n]<min_test_label:
#         min_test_label = test_labels[n]
#     if test_labels[n]>max_test_label:
#         max_test_label = test_labels[n]
# print("Min label: ", min_test_label)
# print("Max label: ", max_test_label)

# print(test_labels)

# print("Training Data Shape: ", train_dat.shape)
# print("Val Data Shape: ", val_dat.shape)
# print("Testing Data Shape: ", test_dat.shape)
# sys.exit()

Dtrain = xgb.DMatrix(train_dat,label=train_labels)
Dval = xgb.DMatrix(val_dat,label=val_labels)
Dtest = xgb.DMatrix(test_dat)


val_pred= []
test_pred = []
val_metrics = []
model_params = []
feature_scores = []
min_mae = np.inf

depthvals = range(3,10)
etavals =[0.01,0.03]
binvals = [256]
#estimatorvals = [16,32,64,96]
childweightvals = [1,1000,10000]
subsamplevals = [0.5,0.75,0.9,1.]
samplebytreevals = [0.5,0.75,0.9,1.]
total_num =len(depthvals)*len(etavals)*len(binvals)*len(childweightvals)*len(subsamplevals)*len(samplebytreevals)
progress =0
print('')
for _max_depth in depthvals:
    for _eta in etavals:
        for _max_bin in binvals:
            for _min_child_weight in childweightvals:
                for _subsample in subsamplevals:
                    for _colsample_bytree in samplebytreevals:
                        #for _n_estimators in estimatorvals:
                        params = {
                            # Parameters that we are going to tune.
                            'max_depth':_max_depth,
                            #'lambda': _lambda,
                            'min_child_weight':_min_child_weight,
                            'learning_rate':_eta,
                            'subsample': _subsample,
                            'colsample_bytree': _colsample_bytree,
                            # Other parameters
                        }
                        params['gpu_id']=1
                        params['tree_method'] = 'gpu_hist'
                        params['max_bin'] = _max_bin
                        params['objective'] = 'reg:squarederror'
                        params['eval_metric'] = 'rmse'
                        #params[]

                        model = xgb.train(params,Dtrain,1000,evals=[[Dval,'Validation']],early_stopping_rounds=50,verbose_eval=False)
                        val_metrics.append(model.best_score)
                        if model.best_score < min_mae:
                            min_mae = model.best_score
                            best_model = progress
                        fscores = model.get_fscore()
                        # feature_scores.append([np.mean([fscores['f'+str(j)] if 'f'+str(j) in fscores.keys() else 0 for j in range(cum_feat_size[i],cum_feat_size[i+1])]) for i in range(len(feat))])
                        model_params.append([_max_depth, _eta, _max_bin, _min_child_weight, _subsample, _colsample_bytree])
                        val_pred.append(model.predict(Dval,ntree_limit=model.best_iteration+1))
                        test_pred.append(model.predict(Dtest,ntree_limit=model.best_iteration+1))
                        progress+=1
                        print('\r%i/%i (%3.2f%%) Current rmse: %1.6f; best rmse: %1.6f from model %i'%(progress,total_num,progress/float(total_num)*100,model.best_score,min_mae,best_model)),

best_models = np.argsort(val_metrics)
# print("Best_models: ", best_models)
print('Best model is model %i (rmse:%f), which corresponds to:'%(best_models[0]+1,val_metrics[best_models[0]]))
print(model_params[best_models[0]])


# Create a folder for saving results
print("Save Results")
new_folder_name = 'plots/RFN/'+str(datetime.datetime.now())[:19].replace(" ","_").replace(":","-")
if not os.path.exists(new_folder_name): os.makedirs(new_folder_name) # create log directory.
# save model to file
pickle.dump(model, open(new_folder_name+"/OMOQ_RFN.pickle.dat", "wb"))
pickle.dump(val_metrics, open(new_folder_name+"/OMOQ_RFN_val_metrics.pickle.dat", "wb"))
pickle.dump(model_params, open(new_folder_name+"/OMOQ_RFN_model_params.pickle.dat", "wb"))
pickle.dump(val_pred, open(new_folder_name+"/OMOQ_RFN_val_pred.pickle.dat", "wb"))
pickle.dump(test_pred, open(new_folder_name+"/OMOQ_RFN_test_pred.pickle.dat", "wb"))

# load the model
# model = pickle.load(open("OMOQ_RFN.pickle.dat", "rb"))

# xgb.plot_importance(model, ax=None, )
# plt.savefig('RFN_Importance.png',dpi=300,format='png')

# np.mean(np.abs(val_pred[best_models[0]] - val_labels[:,0]))


print("Plotting results")
line_x = [1, 5]
line_y = line_x
plt.figure()
plt.hist2d(test_labels.reshape(-1),test_pred[best_models[0]], bins=40, range = [[0, 6],[0, 6]])
cb = plt.colorbar()
cb.set_label('Count')
plt.plot(line_x,line_y,'r--', linewidth=2)
plt.xlabel('Subjective MeanOS')
plt.ylabel('Objective MOQ')
plt.title('Confusion Matrix for Test Files for Single Best Model')
plt.savefig(new_folder_name+'/RFN_Test_Subjective_vs_Objective.png',dpi=300,format='png')

line_x = [1, 5]
line_y = line_x
plt.figure()
plt.hist2d(val_labels.reshape(-1),val_pred[best_models[0]], bins=40, range = [[0, 6],[0, 6]])
cb = plt.colorbar()
cb.set_label('Count')
plt.plot(line_x,line_y,'r--', linewidth=2)
plt.xlabel('Subjective MeanOS')
plt.ylabel('Objective MOQ')
plt.title('Confusion Matrix for Validation Files for Single Best Model')
plt.savefig(new_folder_name+'/RFN_Val_Subjective_vs_Objective.png',dpi=300,format='png')


# print("Test outputs: ", test_pred[best_models[0]])
# print()

val_pcc = np.corrcoef(val_pred[best_models[0]],val_labels.reshape(-1))[0,1]
test_pcc = np.corrcoef(test_pred[best_models[0]],test_labels.reshape(-1))[0,1]

print("Val pcc for best model = ", val_pcc)
print("Test pcc for best model = ", test_pcc)



# # Is this section averaging the models to find the optimal one?
min_mae = np.inf
optimal_model = 0
for k in range(50):
    # print("k = ", k)
    mae = np.mean(np.abs(np.mean([val_pred[i] for i in best_models[:k]],0) - val_labels.reshape(-1)))
    if mae<min_mae:
        min_mae = mae
        optimal_model = k
print("Min MAE: ", min_mae, "Optimal Model: ", optimal_model)

# print("Test labels: ", test_labels.reshape(-1))
# print("Test Predictions: ", test_pred[best_models[0]])


test_pred_submit = np.mean([test_pred[i] for i in best_models[:optimal_model]],0)
# print("Test Predictions Final: ", test_pred_submit)

test_pcc_final = np.corrcoef(test_pred_submit.reshape(-1),test_labels.reshape(-1))[0,1]
print("pcc for best model = ", test_pcc_final)


# Save out the results to a CSV
print("Saving out results to CSV")
with open(new_folder_name+"/Test_Results.csv", "a") as results: results.write("Target, Prediction, Prediction Final\n")
with open(new_folder_name+"/Test_Results.csv", "a") as results:
    for n in range(len(test_labels.reshape(-1))):
        results.write(str(test_labels.reshape(-1)[n]))
        results.write(", ")
        results.write(str(test_pred[best_models[0]][n]))
        results.write(", ")
        results.write(str(test_pred_submit[n])+ "\n")


file1 = open(new_folder_name+"/RFN_Results.txt","a")
file1.write("Feature File = ")
file1.write(load_file)
file1.write("\n")
file1.write("Training Target (0 is MeanOS, 1 is MedianOS) = ")
file1.write(str(training_target))
file1.write("\n")
file1.write('Best model is model %i (rmse:%f), which corresponds to:'%(best_models[0]+1,val_metrics[best_models[0]]))
file1.write("\n")
file1.write(str(model_params[best_models[0]]))
file1.write("\n")
file1.write("Val pcc for best model = ")
file1.write(str(val_pcc))
file1.write("\n")
file1.write("Test pcc for best model = ")
file1.write(str(test_pcc))
file1.write("\n")
file1.write("Min MAE: ")
file1.write(str(min_mae))
file1.write(str("Optimal Model: "))
file1.write(str(optimal_model))
file1.write("\n")
file1.write("pcc for best model = ")
file1.write(str(test_pcc_final))
file1.write("\n")
file1.close()



#
# fname='submission_'+str(datetime.datetime.now())[:19].replace(" ","_").replace(":","-")+'.csv'
# print('Saving to file log/'+fname)
# with open('log/'+fname,'w') as f:
#     f.write("seg_id,time_to_failure\n")
#     for I,i in enumerate(test_fids):
#         f.write('%s,%1.4f\n'%(i.replace('.csv',''),test_pred_submit[I]))


#Send an email once the processing is complete.
#Add code here so that the results are included in the email.

import smtplib, ssl

port = 465  # For SSL
smtp_server = ""
sender_email = ""  # Enter your address
receiver_email = ""  # Enter receiver address
# password = input("Type your password and press enter: ")
password = ""
message = """\
From:

Subject: RFN Processing Complete

To: 

Val pcc of {}, test pcc of {}.
The processing on stink has finished.""".format(val_pcc, test_pcc)

context = ssl.create_default_context()
with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
    server.login(sender_email, password)
    server.sendmail(sender_email, receiver_email, message)
print("Email Sent")
