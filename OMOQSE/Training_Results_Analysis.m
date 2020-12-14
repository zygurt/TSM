% Analysing Network output

close all
clear all
clc
addpath('./Functions/');
addpath('../Functions/');
load('../Subjective_Testing/Plotting_Data_Anon_No_Outliers.mat')
rows_to_read = 30;
csv_filelist = rec_filelist('data/CSVs/');
disp(csv_filelist)
p = 1;
for f = 1:size(csv_filelist,1)
    temp = char(csv_filelist{f});
    temp = split(temp,'/');
    temp = char(temp{end});
    if temp(1)~='.' %Avoiding hidden files created by OS X
      fprintf('%s\n',temp)
      res(p).name = temp(1:end-4);
      res(p).data = Import_OMOQSE_CSV(csv_filelist{f}, 2, rows_to_read+1);

      %Best epoch calculations
      res(p).data.Best_Mean_Loss = mean([res(p).data.TrainLoss,res(p).data.ValLoss,res(p).data.TestLoss],2);
      res(p).data.Best_Diff_Loss = max([res(p).data.TrainLoss,res(p).data.ValLoss,res(p).data.TestLoss],[],2) - ...
                                    min([res(p).data.TrainLoss,res(p).data.ValLoss,res(p).data.TestLoss],[],2);
      res(p).data.Best_Mean_PCC = mean([res(p).data.TrainPCC,res(p).data.ValPCC,res(p).data.TestPCC],2);
      res(p).data.Best_Diff_PCC = max([res(p).data.TrainPCC,res(p).data.ValPCC,res(p).data.TestPCC],[],2) - ...
                                    min([res(p).data.TrainPCC,res(p).data.ValPCC,res(p).data.TestPCC],[],2);
      p = p+1;
    end
end


for f = 1:size(res,2)
    legend_labels{f} = strrep(res(f).name(1:end),'_',' ');
end
disp(legend_labels)


% markers = {'.', 'o', 'x', '+', '*', 's', 'd', '.', 'o', 'x', '+', '*', 's', 'd', '.', 'o', 'x', '+', '*'};
markers = {'x', 'x', 'x', 'x', 'x', 'x', 'x', '+', '+', '+', '+', '+', '+', '+', '*', '*', '*', '*', '*', '*', '*', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 's', 's', 's', 's', 's', 's', 's', 'd', 'd', 'd', 'd', 'd', 'd', 'd'};

%Do the same, but for the best distance
fprintf('Best Distance Measure')
for n = 1:size(res,2)
    res(n).data.Best_RMSE_distance = sqrt((res(n).data.Best_Mean_Loss).^2+(res(n).data.Best_Diff_Loss).^2);
    res(n).data = sortrows(res(n).data,'Best_RMSE_distance','ascend');
    fprintf('%s, %s, Seed: %d, Best_RMSE_distance: %.4f\n',res(n).name, res(n).data.Folder(1), res(n).data.Seed(1), res(n).data.Best_RMSE_distance(1))
end

for n = 1:size(res,2)
    res(n).data.Best_PCC_distance = sqrt((1-res(n).data.Best_Mean_PCC).^2+(res(n).data.Best_Diff_PCC).^2);
    res(n).data = sortrows(res(n).data,'Best_PCC_distance','ascend');
    fprintf('%s, %s, Seed: %d, Best_PCC_distance: %.4f\n',res(n).name, res(n).data.Folder(1), res(n).data.Seed(1), res(n).data.Best_PCC_distance(1))
end

figure('Position',[0 0 640 355])
hold on
for n = 1:size(res,2)
    res(n).data.Best_Final_distance = sqrt(res(n).data.Best_RMSE_distance.^2+res(n).data.Best_PCC_distance.^2);
    res(n).data = sortrows(res(n).data,'Best_Final_distance','ascend');
    res(n).Best_dist = res(n).data.Best_Final_distance(1);
    res(n).Best_Seed = res(n).data.Seed(1);
    res(n).Best_Epoch = res(n).data.BestEpoch(1);
    fprintf('%s, %s, Seed: %d, Epoch: %d,  Best_Final_distance: %.4f\n',res(n).name, res(n).data.Folder(1), res(n).data.Seed(1), res(n).data.BestEpoch(1), res(n).data.Best_Final_distance(1))
    plot(res(n).data.Best_RMSE_distance,res(n).data.Best_PCC_distance,markers{n})
end
% title('Final Distance')
xlabel('RMSE Distance')
ylabel('PCC Distance')
% axis([0.44 0.75 0.1 0.4])
legend(legend_labels,'location','eastoutside')%,'NumColumns',2)

set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');

set(gcf, 'Position', get(0, 'Screensize'));
% print('Output/Plots/EPSC/RMSE_PCC_Best_All_Alignments', '-depsc');
% print('Output/Plots/PNG/RMSE_PCC_Best_All_Alignments', '-dpng');



%Create Boxplot of best distances for each network configuration

best_a_data = zeros(rows_to_read,size(res,2));
for n = 1:size(res,2)
    best_a_data(:,n) = res(n).data.Best_Final_distance;
end

[~,I] = sort(median(best_a_data),'descend');
figure('Position',[146 318 551 360])
boxplot(best_a_data(:,I),'labels',legend_labels(I),'notch','on');

% title('Boxplot for Best Overall Distance Measure')
ylabel('Overall Distance ($\mathcal{D}$)','Interpreter','latex')
xtickangle(45)
set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');

% set(gcf, 'Position', get(0, 'Screensize'));
% print('Output/Plots/EPSC/Boxplot_Overall_Dist_Median_Sort', '-depsc');
% print('Output/Plots/PNG/Boxplot_Overall_Dist_Median_Sort', '-dpng');

% close all

%% Generate latex output

best_res_data = zeros(size(res,2),8);
for n = 1:size(res,2)
    best_res_data(n,1) = res(n).data.TrainLoss(1); 
    best_res_data(n,2) = res(n).data.TrainPCC(1); 
    best_res_data(n,3) = res(n).data.ValLoss(1); 
    best_res_data(n,4) = res(n).data.ValPCC(1); 
    best_res_data(n,5) = res(n).data.TestLoss(1); 
    best_res_data(n,6) = res(n).data.TestPCC(1); 
%     best_res_data(n,3) = res(n).data.Best_Mean_Loss(1);
%     best_res_data(n,4) = res(n).data.Best_Diff_Loss(1);
%     best_res_data(n,5) = res(n).data.Best_Mean_PCC(1);
%     best_res_data(n,6) = res(n).data.Best_Diff_PCC(1);
    best_res_data(n,7) = median([res(n).data.Best_Final_distance]);
    best_res_data(n,8) = res(n).data.Best_Final_distance(1);
    configs{n} = strrep(res(n).name(1:end),'_',' ');
end

best_res_data(11,7) = 1.349895709; %Nan value adjustment for LSTM POW
best_res_data(5,7) = 1.363956813; %Nan value adjustment for BLSTM POW

[~, I] = sort(best_res_data(:,7),'descend');


input.data = best_res_data(I,:);


input.tablePlacement = 'ht';
% input.tableColLabels = {'$\mathcal{L}_{te}$', '$\rho_{te}$', '$\overline{\mathcal{L}}$','$\Delta\mathcal{L}$', '$\overline{\rho}$','$\Delta\rho$', '$\widetilde{\mathcal{D}}$', '$\text{min}(\mathcal{D})$'};
input.tableColLabels = {'$\mathcal{L}_{te}$', '$\rho_{te}$', ...
                        '$\mathcal{L}_{te}$', '$\rho_{te}$', ...
                        '$\mathcal{L}_{te}$', '$\rho_{te}$', ...
                        '$\widetilde{\mathcal{D}}$', '$\text{min}(\mathcal{D})$'};

input.tableRowLabels = configs(I);
input.dataFormat = {'%.3f'};
input.tableColumnAlignment = 'c';
input.tableBorders = 1;
input.tableCaption = 'Training, Validation and Test Loss ($\mathcal{L}$) and PCC ($\rho$), median overall distance ($\widetilde{\mathcal{D}}$) and minimum overall distance ($\text{min}(\mathcal{D})$).  Best results in bold.';
input.tableLabel = 'RMSE_PCC';
input.makeCompleteLatexDocument = 0;
fprintf('Writing RMSE_PCC.tex latex table\n')
fid = fopen('Output/Tex/RMSE_PCC_Pow.tex','w');
latex = JASAlatexTable(input,fid);
fclose(fid);


%% Compute confidence interval for all subjective scores

% x = randi(50, 1, 100);                      % Create Data
% SEM = std(x)/sqrt(length(x));               % Standard Error
% ts = tinv([0.025  0.975],length(x)-1);      % T-Score
% CI = mean(x) + ts*SEM;
addpath('../Functions');
% load('../Subjective_Testing/Plotting_Data_Anon_No_Outliers.mat')
load('../Subjective_Testing/Plotting_Data_RMSE.mat')
%MeanOS
%CNN percentiles
RMSE = res(5).data.MeanLoss(1);
rho = res(5).data.MeanPCC(1);
% fid = fopen('log_Final.txt','a');
p = inv_prctile(rho,[u.pearson_corr_MeanOS_norm],'up');
fprintf('For CNN, the mean PCC of %g is at the %d Percentile\n', rho, p);
p = inv_prctile(RMSE,[u.RMSE_norm],'down');
fprintf('For CNN, the mean RMSE of %g is at the %d Percentile\n', RMSE, p);

%CNN Test percentiles
RMSE = res(5).data.TestLoss(1);
rho = res(5).data.TestPCC(1);
% fid = fopen('log_Final.txt','a');
p = inv_prctile(rho,[u.pearson_corr_MeanOS_norm],'up');
fprintf('For CNN, the Test PCC of %g is at the %d Percentile\n', rho, p);
p = inv_prctile(RMSE,[u.RMSE_norm],'down');
fprintf('For CNN, the Test RMSE of %g is at the %d Percentile\n', RMSE, p);

%BGRU-FT percentiles
RMSE = res(2).data.MeanLoss(1);
rho = res(2).data.MeanPCC(1);
% fid = fopen('log_Final.txt','a');
p = inv_prctile(rho,[u.pearson_corr_MeanOS_norm],'up');
fprintf('For BGRU-FT, the mean PCC of %g is at the %d Percentile\n', rho, p);
p = inv_prctile(RMSE,[u.RMSE_norm],'down');
fprintf('For BGRU-FT, the mean RMSE of %g is at the %d Percentile\n', RMSE, p);

%BGRU-FT Test percentiles
RMSE = res(2).data.TestLoss(1);
rho = res(2).data.TestPCC(1);
% fid = fopen('log_Final.txt','a');
p = inv_prctile(rho,[u.pearson_corr_MeanOS_norm],'up');
fprintf('For BGRU-FT, the Test PCC of %g is at the %d Percentile\n', rho, p);
p = inv_prctile(RMSE,[u.RMSE_norm],'down');
fprintf('For BGRU-FT, the Test RMSE of %g is at the %d Percentile\n', RMSE, p);




figure
histogram([u.pearson_corr_MeanOS_norm],100)
title('Subjective session PCC')

figure
histogram([u.RMSE_norm],100)
title('Subjective session RMSE')

