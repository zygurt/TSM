% Analysing Network output

close all
clear all
clc
addpath('.\Functions\');
addpath('..\Functions\');
load('..\Subjective_Testing\Plotting_Data_Anon_No_Outliers.mat')

csv_filelist = rec_filelist('data\CSVs\Paper_Results');
disp(csv_filelist)
p = 1;
for f = 1:size(csv_filelist,1)
    temp = char(csv_filelist{f});
    temp = split(temp,'/');
    temp = char(temp{end});
    if temp(1)~='.' %Avoiding hidden files created by OS X
      fprintf('%s\n',temp)
      res(p).name = temp(1:end-4);
      res(p).data = Import_Training_CSV(csv_filelist{f}, 2, 101);
     
      %Best epoch calculations
      res(p).data.Best_Mean_Loss = mean([res(p).data.BDETrainLoss,res(p).data.BDEValLoss,res(p).data.BDETestLoss],2);
      res(p).data.Best_Diff_Loss = max([res(p).data.BDETrainLoss,res(p).data.BDEValLoss,res(p).data.BDETestLoss],[],2) - ...
                                    min([res(p).data.BDETrainLoss,res(p).data.BDEValLoss,res(p).data.BDETestLoss],[],2);
      res(p).data.Best_Mean_PCC = mean([res(p).data.BDETrainPCC,res(p).data.BDEValPCC,res(p).data.BDETestPCC],2);
      res(p).data.Best_Diff_PCC = max([res(p).data.BDETrainPCC,res(p).data.BDEValPCC,res(p).data.BDETestPCC],[],2) - ...
                                    min([res(p).data.BDETrainPCC,res(p).data.BDEValPCC,res(p).data.BDETestPCC],[],2);
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
    res(n).Best_Epoch = res(n).data.BestDistEpoch(1);
    fprintf('%s, %s, Seed: %d, Epoch: %d,  Best_Final_distance: %.4f\n',res(n).name, res(n).data.Folder(1), res(n).data.Seed(1), res(n).data.BestDistEpoch(1), res(n).data.Best_Final_distance(1))
    plot(res(n).data.Best_RMSE_distance,res(n).data.Best_PCC_distance,markers{n})
end
% title('Final Distance')
xlabel('RMSE Distance')
ylabel('PCC Distance')
axis([0.44 0.75 0.1 0.4])
legend(legend_labels(1:end-1),'location','northoutside','NumColumns',3)

set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');
% set(gcf, 'Position', get(0, 'Screensize'));
% print('plots/MATLAB/TIFF/RMSE_PCC_Best_All_Alignments', '-dtiff');
% print('plots/MATLAB/EPSC/RMSE_PCC_Best_All_Alignments', '-depsc');
% print('plots/MATLAB/PNG/RMSE_PCC_Best_All_Alignments', '-dpng');



%Create Boxplot of best distances for each network configuration
    
best_a_data = zeros(100,size(res,2));
for n = 1:size(res,2)
    best_a_data(:,n) = res(n).data.Best_Final_distance;
end

[~,I] = sort(median(best_a_data),'descend');  %Sort by minimum or median
figure('Position',[146 318 551 360])
boxplot(best_a_data(:,I),'labels',legend_labels(I),'notch','on');

% title('Boxplot for Best Overall Distance Measure')
ylabel('Overall Distance ($\mathcal{D}$)','Interpreter','latex')
xlabel('Network Input Features')
xtickangle(45)
set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');

% set(gcf, 'Position', get(0, 'Screensize'));
% print('plots/MATLAB/TIFF/Boxplot_Overall_Dist_Median_Sort', '-dtiff');
print('plots/MATLAB/EPSC/Boxplot_Overall_Dist_Median_Sort', '-depsc');
print('plots/MATLAB/PNG/Boxplot_Overall_Dist_Median_Sort', '-dpng');

% close all

%% Generate latex output

best_res_data = zeros(size(res,2),5);
for n = 1:size(res,2)
    best_res_data(n,1) = res(n).data.Best_Mean_Loss(1);
    best_res_data(n,2) = res(n).data.Best_Diff_Loss(1);
    best_res_data(n,3) = res(n).data.Best_Mean_PCC(1);
    best_res_data(n,4) = res(n).data.Best_Diff_PCC(1);
    best_res_data(n,5) = median([res(n).data.Best_Final_distance]);
    best_res_data(n,6) = res(n).data.Best_Final_distance(1);
    configs{n} = strrep(res(n).name(1:end),'_',' ');
end

[~, I] = sort(best_res_data(:,5),'descend');

input.data = best_res_data(I,:);

input.tablePlacement = 'ht';
input.tableColLabels = {'$\overline{\mathcal{L}}$','$\Delta\mathcal{L}$', '$\overline{\rho}$','$\Delta\rho$', '$\widetilde{\mathcal{D}}$', '$\text{min}(\mathcal{D})$'};
input.tableRowLabels = configs(I);
input.dataFormat = {'%.3f'};
input.tableColumnAlignment = 'c';
input.tableBorders = 1;
input.tableCaption = 'RSME loss mean ($\overline{\mathcal{L}}$) and range ($\Delta\mathcal{L}$), PCC mean ($\overline{\rho}$) and range ($\Delta\rho$), median overall distance ($\widetilde{\mathcal{D}}$) and minimum overall distance ($\text{min}(\mathcal{D})$).  Best results in bold.';
input.tableLabel = 'RMSE_PCC';
input.makeCompleteLatexDocument = 0;
fprintf('Writing RMSE_PCC.tex latex table\n')
fid = fopen('RMSE_PCC.tex','w');
latex = JASAlatexTable(input,fid);
fclose(fid);
