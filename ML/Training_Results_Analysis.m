% Analysing Network output

close all
clear all
clc
addpath('.\Functions\');
addpath('..\Functions\');
load('..\Subjective_Testing\Plotting_Data_Anon_No_Outliers.mat')

csv_filelist = rec_filelist('data\CSVs\');
% csv_filelist = csv_filelist(2:end); %Skipping the ._ file created by os x
% csv_filelist = csv_filelist(2)
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
% legend_labels(size(legend_labels,2)+1) = {'Subjective Results'};
disp(legend_labels)


% markers = {'.', 'o', 'x', '+', '*', 's', 'd', '.', 'o', 'x', '+', '*', 's', 'd', '.', 'o', 'x', '+', '*'};
markers = {'x', 'x', 'x', 'x', 'x', 'x', 'x', '+', '+', '+', '+', '+', '+', '+', '*', '*', '*', '*', '*', '*', '*', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 's', 's', 's', 's', 's', 's', 's', 'd', 'd', 'd', 'd', 'd', 'd', 'd'};
% fprintf('Early Stopping Distance Measure')
% for n = 1:size(res,2)
%     res(n).data.RMSE_distance = sqrt((res(n).data.Mean_Loss).^2+res(n).data.Diff_Loss.^2);
%     res(n).data = sortrows(res(n).data,'RMSE_distance','ascend');
%     % fprintf('%s, %s, Seed: %d, RMSE_distance: %.4f\n',res(n).name, res(n).data.Folder(1), res(n).data.Seed(1), res(n).data.RMSE_distance(1))
% end
% 
% for n = 1:size(res,2)
%     res(n).data.PCC_distance = sqrt((1-res(n).data.Mean_PCC).^2+res(n).data.Diff_PCC.^2);
%     res(n).data = sortrows(res(n).data,'PCC_distance','ascend');
%     % fprintf('%s, %s, Seed: %d, PCC_distance: %.4f\n',res(n).name, res(n).data.Folder(1), res(n).data.Seed(1), res(n).data.PCC_distance(1))
% end

% figure('Position',[1680-500 200 1000 600])
% hold on
% for n = 1:size(res,2)
%     res(n).data.Final_distance = sqrt(res(n).data.RMSE_distance.^2+res(n).data.PCC_distance.^2);
%     res(n).data = sortrows(res(n).data,'Final_distance','ascend');
%     % fprintf('%s, %s, Seed: %d, Epoch: %d,  Final_distance: %.4f\n',res(n).name, res(n).data.Folder(1), res(n).data.Seed(1), res(n).data.BestEpoch(1), res(n).data.Final_distance(1))
%     plot(res(n).data.RMSE_distance,res(n).data.PCC_distance,markers{n})
% end
% title('Final Distance')
% xlabel('RMSE Distance')
% ylabel('PCC Distance')
% legend(legend_labels(1:end-1),'location','best')
% 
% set(gca,...
%     'FontSize', 12, ...
%     'FontName', 'Times');
% set(gcf, 'Position', get(0, 'Screensize'));
% print('plots/MATLAB/TIFF/RMSE_PCC_Early_Stop_All_Alignments', '-dtiff');
% print('plots/MATLAB/EPSC/RMSE_PCC_Early_Stop_All_Alignments', '-depsc');
% print('plots/MATLAB/PNG/RMSE_PCC_Early_Stop_All_Alignments', '-dpng');


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
print('plots/MATLAB/TIFF/RMSE_PCC_Best_All_Alignments', '-dtiff');
print('plots/MATLAB/EPSC/RMSE_PCC_Best_All_Alignments', '-depsc');
print('plots/MATLAB/PNG/RMSE_PCC_Best_All_Alignments', '-dpng');

    
%Run anova test on the results
%This will show how similar the means are to each other.
%Lower means closer
% a_data = zeros(100,size(res,2));
% for n = 1:size(res,2)
%     a_data(:,n) = res(n).data.Final_distance;
% end

% figure('Position',[1680-500 200 900 400])
% boxplot(a_data,'labels',legend_labels(1:end-1),'notch','on');
% 
% title('Boxplot for Overall Distance Measure')
% ylabel('Overall Distance')
% xtickangle(45)
% set(gcf, 'Position', get(0, 'Screensize'));
% print('plots/MATLAB/TIFF/Boxplot_Overall_Dist_Early_Stop_All_Alignments', '-dtiff');
% print('plots/MATLAB/EPSC/Boxplot_Overall_Dist_Early_Stop_All_Alignments', '-depsc');
% print('plots/MATLAB/PNG/Boxplot_Overall_Dist_Early_Stop_All_Alignments', '-dpng');

best_a_data = zeros(100,size(res,2));
for n = 1:size(res,2)
    best_a_data(:,n) = res(n).data.Best_Final_distance;
end

% [~,I] = sort(min(best_a_data),'descend');
% figure('Position',[146 318 1211 559])
% boxplot(best_a_data(:,I),'labels',legend_labels(I),'notch','on');
% 
% % title('Boxplot for Best Overall Distance Measure')
% ylabel('Overall Distance')
% xtickangle(45)
% set(gca,...
%     'FontSize', 12, ...
%     'FontName', 'Times');

% set(gcf, 'Position', get(0, 'Screensize'));
% print('plots/MATLAB/TIFF/Boxplot_Overall_Dist_Min_Sort', '-dtiff');
% print('plots/MATLAB/EPSC/Boxplot_Overall_Dist_Min_Sort', '-depsc');
% print('plots/MATLAB/PNG/Boxplot_Overall_Dist_Min_Sort', '-dpng');


% [~,I] = sort(mean(best_a_data),'descend');
% figure('Position',[146 318 1211 559])
% boxplot(best_a_data(:,I),'labels',legend_labels(I),'notch','on');
% 
% % title('Boxplot for Best Overall Distance Measure')
% ylabel('Overall Distance ($\mathcal{D}$)','Interpreter','latex')
% % ('$\rho$','Interpreter','latex')
% xtickangle(45)
% set(gca,...
%     'FontSize', 12, ...
%     'FontName', 'Times');

% set(gcf, 'Position', get(0, 'Screensize'));
% print('plots/MATLAB/TIFF/Boxplot_Overall_Dist_Mean_Sort', '-dtiff');
% print('plots/MATLAB/EPSC/Boxplot_Overall_Dist_Mean_Sort', '-depsc');
% print('plots/MATLAB/PNG/Boxplot_Overall_Dist_Mean_Sort', '-dpng');


[~,I] = sort(median(best_a_data),'descend');
figure('Position',[146 318 551 360])
boxplot(best_a_data(:,I),'labels',legend_labels(I),'notch','on');

% title('Boxplot for Best Overall Distance Measure')
ylabel('Overall Distance ($\mathcal{D}$)','Interpreter','latex')
% ('$\rho$','Interpreter','latex')
xtickangle(45)
set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');

% set(gcf, 'Position', get(0, 'Screensize'));
print('plots/MATLAB/TIFF/Boxplot_Overall_Dist_Median_Sort', '-dtiff');
print('plots/MATLAB/EPSC/Boxplot_Overall_Dist_Median_Sort', '-depsc');
print('plots/MATLAB/PNG/Boxplot_Overall_Dist_Median_Sort', '-dpng');



% figure('Position',[146 318 1211 559])
% boxplot(best_a_data,'labels',legend_labels,'notch','on');
% 
% % title('Boxplot for Best Overall Distance Measure')
% ylabel('Overall Distance ($\mathcal{D}$)','Interpreter','latex')
% % ('$\rho$','Interpreter','latex')
% xtickangle(45)
% set(gca,...
%     'FontSize', 12, ...
%     'FontName', 'Times');
% 
% set(gcf, 'Position', get(0, 'Screensize'));
% print('plots/MATLAB/TIFF/Boxplot_Overall_Dist_Name_Sort', '-dtiff');
% print('plots/MATLAB/EPSC/Boxplot_Overall_Dist_Name_Sort', '-depsc');
% print('plots/MATLAB/PNG/Boxplot_Overall_Dist_Name_Sort', '-dpng');


% close all

%% Generate latex output


addpath('..\..\External\');

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
%%
% %Load the Test.CSV before this section
% %1:80 is FuzzyTSM
% %81:160 is Elastique
% %161:240 is NMF
% Test_TSM = [a(5281:end).TSM];
% figure
% hold on
% plot(Test.SMOS(1:80),Test.OMOS(1:80),'.')
% plot(Test.SMOS(81:160),Test.OMOS(81:160),'.')
% plot(Test.SMOS(161:240),Test.OMOS(161:240),'.')
% hold off
% legend('FuzzyTSM','Elastique','NMF','location','best')
% xlabel('SMOS')
% ylabel('OMOS')
%
%
% figure
% hold on
% plot(Test_TSM(1:80),Test.OMOS(1:80),'.')
% plot(Test_TSM(81:160),Test.OMOS(81:160),'.')
% plot(Test_TSM(161:240),Test.OMOS(161:240),'.')
% hold off
% legend('FuzzyTSM','Elastique','NMF','location','best')
% xlabel('TSM Ratio')
% ylabel('OMOS')






% figure%('Position',[1680-500 200 500 300])
% hold on
% for f = 1:size(res,2)
%     plot(res(f).data.Sum_Loss,res(f).data.Mean_PCC,'.')
% end
% hold off
% legend(legend_labels(1:end-1),'location','best')
% xlabel('Sum of Loss')
% ylabel('Mean PCC')
%
% set(gca,...
%     'FontSize', 12, ...
%     'FontName', 'Times');
%
% print('plots/MATLAB/TIFF/Sum_Loss_MeanPCC', '-dtiff');
% print('plots/MATLAB/EPSC/Sum_Loss_MeanPCC', '-depsc');
% print('plots/MATLAB/PNG/Sum_Loss_MeanPCC', '-dpng');
%
% figure%('Position',[1680-500 200 500 300])
% hold on
% for f = 1:size(res,2)
%     plot(res(f).data.Sum_Loss,res(f).data.Diff_PCC,'.')
% end
% hold off
% legend(legend_labels(1:end-1),'location','best')
% xlabel('Sum of Loss')
% ylabel('PCC Difference')
%
% set(gca,...
%     'FontSize', 12, ...
%     'FontName', 'Times');
%
% print('plots/MATLAB/TIFF/Sum_Loss_PCCDifference', '-dtiff');
% print('plots/MATLAB/EPSC/Sum_Loss_PCCDifference', '-depsc');
% print('plots/MATLAB/PNG/Sum_Loss_PCCDifference', '-dpng');
%
% figure%('Position',[1680-500 200 500 300])
% hold on
% for f = 1:size(res,2)
%     [N,EDGES] = histcounts(res(f).data.Sum_Loss,20);
%     plot((EDGES(1:end-1)+EDGES(2:end))/2,N);
% end
% hold off
% legend(legend_labels(1:end-1),'location','best')
% xlabel('Sum of Loss')
% ylabel('Count')
%
% set(gca,...
%     'FontSize', 12, ...
%     'FontName', 'Times');
%
% print('plots/MATLAB/TIFF/Sum_Loss_Hist', '-dtiff');
% print('plots/MATLAB/EPSC/Sum_Loss_Hist', '-depsc');
% print('plots/MATLAB/PNG/Sum_Loss_Hist', '-dpng');
%
% % Plot the MeanPCCs
% figure%('Position',[1680-500 200 500 300])
% hold on
% for f = 1:size(res,2)
%     [N,EDGES] = histcounts(res(f).data.Mean_PCC,10,'Normalization','probability');
%     plot((EDGES(1:end-1)+EDGES(2:end))/2,N);
% end
% [N,EDGES] = histcounts([u.pearson_corr_MeanOS_norm],20,'Normalization','probability'); %'BinWidth',0.05
% plot((EDGES(1:end-1)+EDGES(2:end))/2,N);
%
% ylabel('Probability')
% hold off
% legend(legend_labels,'location','best')
% xlabel('$\rho$','Interpreter','latex')
%
% set(gca,...
%     'FontSize', 12, ...
%     'FontName', 'Times');
%
% print('plots/MATLAB/TIFF/MeanPCC_Hist', '-dtiff');
% print('plots/MATLAB/EPSC/MeanPCC_Hist', '-depsc');
% print('plots/MATLAB/PNG/MeanPCC_Hist', '-dpng');
%
% %Create matrix for anova test
% anova_loss_mat = res(1).data.Sum_Loss;
% for f = 2:size(res,2)
%     anova_loss_mat = [anova_loss_mat res(f).data.Sum_Loss];
% end
% anova1(anova_loss_mat,legend_labels(1:end-1));
% pause(1)
% ylabel('Sum of RMSE Loss')
% xtickangle(30)
%
% set(gca,...
%     'FontSize', 12, ...
%     'FontName', 'Times');
%
% print('plots/MATLAB/TIFF/Anova_Sum_Loss', '-dtiff');
% print('plots/MATLAB/EPSC/Anova_Sum_Loss', '-depsc');
% print('plots/MATLAB/PNG/Anova_Sum_Loss', '-dpng');
%
% anova_pcc_mat = res(1).data.Mean_PCC;
% for f = 2:size(res,2)
%     anova_pcc_mat = [anova_pcc_mat res(f).data.Mean_PCC];
% end
% anova1(anova_pcc_mat,legend_labels(1:end-1));
% pause(1)
% ylabel('Mean PCC')
% xtickangle(30)
%
% set(gca,...
%     'FontSize', 12, ...
%     'FontName', 'Times');
%
% print('plots/MATLAB/TIFF/Anova_MeanPCC', '-dtiff');
% print('plots/MATLAB/EPSC/Anova_MeanPCC', '-depsc');
% print('plots/MATLAB/PNG/Anova_MeanPCC', '-dpng');
%
% anova_distance_mat = res(1).data.Sum_Loss_Diff_PCC_distance;
% for f = 2:size(res,2)
%     anova_distance_mat = [anova_distance_mat res(f).data.Sum_Loss_Diff_PCC_distance];
% end
% anova1(anova_distance_mat,legend_labels(1:end-1));
% pause(1)
% ylabel('Sum_Loss_Diff_PCC_distance Measure')
% xtickangle(30)
%
% set(gca,...
%     'FontSize', 12, ...
%     'FontName', 'Times');
%
% print('plots/MATLAB/TIFF/Anova_Distance', '-dtiff');
% print('plots/MATLAB/EPSC/Anova_Distance', '-depsc');
% print('plots/MATLAB/PNG/Anova_Distance', '-dpng');
% % legend(legend_labels(1:end-1),'location','best')



% %Linear and Third Order Mapping
% train_len = 4572;
% val_len = 528;
% test_len = 240;
% for n = 1:size(csv_filelist,1)
%
%     %     res(n).data = sortrows(res(n).data,'Sum_Loss_Diff_PCC_distance','ascend');
%     for k = 1:100
%         load_folder = res(n).data.Folder(k);
%         load_file = strcat(load_folder, '/Test.csv');
%         fprintf('n=%d, k=%d\n',n,k)
%         Test = Import_Estimate_CSV(load_file, 2, test_len);
%
%         %Linear Mapping
%         [P,S] = polyfit(Test.OMOS,Test.SMOS,1);
%         Test.OMOS_Mapped = polyval(P,Test.OMOS);
%         [P_mapped,S_mapped] = polyfit(Test.OMOS_Mapped,Test.SMOS,1);
% %         figure
% %         plot(Test.SMOS,Test.OMOS,'b.')
% %         hold on
% %         x = 1:0.1:5;
% %         %     plot(x,polyval(P,x),'b--')
% %         plot(Test.SMOS,Test.OMOS_Mapped,'r.')
% %         plot(x,polyval(P_mapped,x),'r--')
% %         plot(x,x,'k:')
% %         hold off
% %         axis([0.9, 5.1, 0.9, 5.1])
% %         xlabel('MeanOS')
% %         ylabel('OMOQ')
% %         title('Linear Mapping')
% %         legend('Raw Results','Linear Mapped Results','Mapped Linear Fit','Ideal','location','best')
%         RMSE = sqrt(mean((Test.SMOS-Test.OMOS).^2));
%         RMSE_Linear_Mapped = sqrt(mean((Test.SMOS-Test.OMOS_Mapped).^2));
%         PCC_RAW = corr(Test.OMOS,Test.SMOS);
%         PCC_Linear = corr(Test.OMOS_Mapped,Test.SMOS);
%
%         res(n).data.Test_RMSE(k) = RMSE;
%         res(n).data.Test_RMSE_Linear_Mapped(k) = RMSE_Linear_Mapped;
%         res(n).data.Test_PCC_RAW(k) = PCC_RAW;
%         res(n).data.Test_PCC_Linear(k) = PCC_Linear;
%
%         %Third Order Mapping
%         [P_third,S_third] = polyfit(Test.OMOS,Test.SMOS,3);
%         Test.OMOS_third_Mapped = polyval(P_third,Test.OMOS);
%         [P_third_mapped,S_third_mapped] = polyfit(Test.OMOS_third_Mapped,Test.SMOS,1);
% %         figure
% %         plot(Test.SMOS,Test.OMOS,'b.')
% %         hold on
% %         %     plot(x,polyval(P_third,x),'b--')
% %         plot(Test.SMOS,Test.OMOS_third_Mapped,'r.')
% %         plot(x,polyval(P_third_mapped, x),'r--')
% %         plot(x,x,'k:')
% %         hold off
% %         axis([0.9, 5.1, 0.9, 5.1])
% %         xlabel('MeanOS')
% %         ylabel('OMOQ')
% %         title('3rd Order Mapping')
% %         legend('Raw Results','3rd Order Mapped Results','Mapped 3rd Order Fit','Ideal','location','best')
%
%         RMSE_Third_Order_Mapped = sqrt(mean((Test.SMOS-Test.OMOS_third_Mapped).^2));
%         PCC_third = corr(Test.OMOS_third_Mapped,Test.SMOS);
%         res(n).data.Test_RMSE_Third_Order_Mapped(k) = RMSE_Third_Order_Mapped;
%         res(n).data.Test_PCC_third(k) = PCC_third;
% %         fprintf('RMSE: %.6f, RMSE Linear: %.6f, RMSE Third Order: %.6f\n',RMSE, RMSE_Linear_Mapped, RMSE_Third_Order_Mapped)
% %         fprintf('PCC: %.6f, PCC Linear: %.6f, PCC Third Order: %.6f\n',PCC_RAW, PCC_Linear, PCC_third)
%     end
%     res(n).data.RMSE_Linear_Improvement = res(n).data.Test_RMSE_Linear_Mapped - res(n).data.Test_RMSE;
%     res(n).data.PCC_Linear_Improvement = res(n).data.Test_PCC_Linear - res(n).data.Test_PCC_RAW;
%     res(n).data.RMSE_Third_Improvement = res(n).data.Test_RMSE_Third_Order_Mapped - res(n).data.Test_RMSE;
%     res(n).data.PCC_Third_Improvement = res(n).data.Test_PCC_third - res(n).data.Test_PCC_RAW;
% end
%
% fprintf('Improvements based on mapping\n')
% for n = 1:size(res,2)
%     fprintf('%s\n',res(n).name)
%     avg_RMSE = mean(res(1).data.Test_RMSE);
%     avg_RMSE_Linear = mean(res(n).data.RMSE_Linear_Improvement);
%     avg_RMSE_Third = mean(res(n).data.RMSE_Third_Improvement);
%
%     avg_RMSE_Linear_per = abs(avg_RMSE_Linear)/avg_RMSE*100;
%     avg_RMSE_Third_per = abs(avg_RMSE_Third)/avg_RMSE*100;
%
%     avg_PCC = mean(res(1).data.Test_PCC_RAW);
%     avg_PCC_Linear = mean(res(n).data.PCC_Linear_Improvement);
%     avg_PCC_Third = mean(res(n).data.PCC_Third_Improvement);
%
%     avg_PCC_Linear_per = abs(avg_PCC_Linear)/avg_PCC*100;
%     avg_PCC_Third_per = abs(avg_PCC_Third)/avg_PCC*100;
%
%     fprintf('Linear RMSE: %2.2f%%. Third RMSE: %2.2f%%\n',avg_RMSE_Linear_per,avg_RMSE_Third_per)
%     fprintf('Linear PCC: %2.2f%%. Third PCC: %2.2f%%\n',avg_PCC_Linear_per,avg_PCC_Third_per)
%
% end


%%
% testfile = 'plots/Best FCN/2020-02-04_14-12-35PEAQ_SMALL_TEST/Test.csv';
% trainfile ='plots/Best FCN/2020-02-04_14-12-35PEAQ_SMALL_TEST/Train.csv';
% validatefile ='plots/Best FCN/2020-02-04_14-12-35PEAQ_SMALL_TEST/Val.csv';
%
% Test = table2array(Import_Network_Output(testfile, 2, 241));
% Train = table2array(Import_Network_Output(trainfile, 2, 4753));
% Validate = table2array(Import_Network_Output(validatefile, 2, 529));
%
% All_output = [Train ; Validate ; Test];
%
%
% figure('Position',[0 0 500 250])
% h = histogram2(All_output(:,1),All_output(:,2),[50 50],'FaceColor','flat');
% h.ShowEmptyBins = 'Off';
% h.DisplayStyle = 'tile';
% h.EdgeAlpha = 0;
%
% ax = gca;
% ax.GridColor = [0.4 0.4 0.4];
% ax.GridLineStyle = '--';
% ax.GridAlpha = 0.5;
% ax.XGrid = 'off';
% ax.YGrid = 'off';
% ax.Layer = 'top';
% view(2)
% % colormap(flipud(gray));
% c = colorbar;
% c.Label.String = 'Count';
% xlabel('Subjective MOS')
% ylabel('Objective Output')
% axis([1 5 1 5])
%
% hold on
% p = plot([1 5],[1 5],'r--');
% p.LineWidth = 2;
% hold off
%
%
% set(gca,...
%     'FontSize', 12, ...
%     'FontName', 'Times');
% print('New_PEAQ_vs_MeanOS', '-dtiff');
% print('New_PEAQ_vs_MeanOS', '-depsc');
% print('New_PEAQ_vs_MeanOS', '-dsvg');



%%

% fname = 'plots/FCN/2020-02-11_14-40-43ALL_TEST/Dist.csv'; %Seed 55
% % fname = 'plots/FCN/2020-02-11_14-45-19ALL_TEST/Dist.csv'; %Seed 56
%
%
%
% [Losstr,Lossval,Losste,PCCtr,PCCval,PCCte,rdist,pccdist,overalldist] = Import_Dist_file(fname, 2, 1001);
%
% Loss_val_velocity = Lossval(2:end)-Lossval(1:end-1);
% Loss_val_accel = Loss_val_velocity(2:end)-Loss_val_velocity(1:end-1);
% figure
% subplot(311)
% plot(Lossval)
% title('Validation Loss')
% subplot(312)
% plot(Loss_val_velocity)
% title('Velocity')
% subplot(313)
% plot(Loss_val_accel)
% title('Acceleration')
%
% Loss_val_velocity = zeros(size(Lossval));
% smooth_length = 60;
% for n = smooth_length+1:length(Lossval)
%     Loss_val_velocity(n) = mean(Lossval(n-smooth_length+1:n)-Lossval(n-smooth_length:n-1));
% end
%
% mean_loss_val_vel = zeros(size(Lossval));
% for n = 80+1:length(Loss_val_velocity)
%     mean_loss_val_vel(n) = mean(Loss_val_velocity(n-50:n));
% end
%
% Loss_val_accel = zeros(size(Lossval));
% for n = 2*smooth_length+1:length(Lossval)
%     Loss_val_accel(n) = mean(Loss_val_velocity(n-smooth_length+1:n)-Loss_val_velocity(n-smooth_length:n-1));
% end
%
%
% figure
% subplot(411)
% plot(Lossval)
% title('Validation Loss')
% subplot(412)
% plot(Loss_val_velocity)
% title('Velocity')
% subplot(413)
% plot(mean_loss_val_vel)
% title('Mean(80) velocity')
% subplot(414)
% plot(Loss_val_accel)
% title('Acceleration')
% %%
%
% %PCC
%
% PCC_val_velocity = PCCval(2:end)-PCCval(1:end-1);
% PCC_val_accel = PCC_val_velocity(2:end)-PCC_val_velocity(1:end-1);
% figure
% subplot(311)
% plot(PCCval)
% title('Validation PCC')
% subplot(312)
% plot(PCC_val_velocity)
% title('Velocity')
% subplot(313)
% plot(PCC_val_accel)
% title('Acceleration')
%
% PCC_val_velocity = zeros(size(PCCval));
% smooth_length = 40;
% for n = smooth_length+1:length(PCCval)
%     PCC_val_velocity(n) = mean(PCCval(n-smooth_length+1:n)-PCCval(n-smooth_length:n-1));
% end
%
% PCC_val_accel = zeros(size(PCCval));
% for n = 2*smooth_length+1:length(PCCval)
%     PCC_val_accel(n) = mean(PCC_val_velocity(n-smooth_length+1:n)-PCC_val_velocity(n-smooth_length:n-1));
% end
%
%
% figure
% subplot(311)
% plot(PCCval)
% title('Validation PCC')
% subplot(312)
% plot(PCC_val_velocity)
% title('Velocity')
% subplot(313)
% plot(PCC_val_accel)
% title('Acceleration')
