%Generate the plots for the paper

close all
clear all
% clc
addpath('./Functions/');
addpath('../Functions/');
% load('../Subjective_Testing/Plotting_Data.mat')

folder_name = './models/FCN/2020-06-21_05-13-02TO_TEST_SOURCE/';%MeanOS
distcsv = [folder_name 'Dist.csv']; 
epochs = 800;
Dist = Import_Dist_CSV(distcsv, 2, epochs+1);
% lines = {':', ':', ':', '--', '--', '--', '-', '-', '-'};
lines = {'-', '-', '-', '-', '-', '-', '-', '-', '-'};
grey_lines= {'k-', 'k:', 'k--', 'k-', 'k:', 'k--', 'k--', 'k--', 'k--'};
alpha_val = [0 0 0 0.6 0.6 0.6];

figure('Position',[0 50 575 335]);
hold on
chosen_cols = [1,2,3,4,5,6];
for n = chosen_cols
   plot(1:size(Dist,1),table2array(Dist(:,n)),grey_lines{n},'color',[0 0 0]+alpha_val(n))%,'LineWidth',2)
    
end
hold off
%xlabel
xlabel('Epoch')
%ylabel
ylabel('RMSE($\mathcal{L}$) and PCC($\rho$)','interpreter','latex')

%Plot a vertical line at the optimal epoch
%Find the minimum Overall Distance
[~,I] = min(table2array(Dist(:,9)));
%Place vertical line at that location
hold on
line([I, I],[0.8*min(min(table2array(Dist(I,chosen_cols)))), 1.1*max(max(table2array(Dist(I,chosen_cols))))],'Color','black','LineStyle','-.')
hold off

axis([1 epochs, min(min(table2array(Dist(:,chosen_cols)))), max(max(table2array(Dist(:,chosen_cols))))])

% legend_items = Dist.Properties.VariableNames(chosen_cols);
% legend_items{length(legend_items)+1} = 'Chosen Epoch';
new_legend_items = {'$\mathcal{L}_{tr}$','$\mathcal{L}_{val}$','$\mathcal{L}_{te}$','$\rho_{tr}$','$\rho_{val}$','$\rho_{te}$','Chosen Epoch'};
legend(new_legend_items,'location','southeast','interpreter','latex')

set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');

% print('Plots/MATLAB/TIFF/Distance_Measure', '-dtiff');
print('plots/MATLAB/EPSC/Distance_Measure', '-depsc');
print('plots/MATLAB/PNG/Distance_Measure', '-dpng');


%Plot Confusion matrixes

%Test Confusion Matrix
test_csv = [folder_name 'Test.csv'];
Test = Import_Test_CSV(test_csv, 2, 241);

figure('Position',[0 50 630 479]);
bins = linspace(1,5,40);
h = histogram2(Test.SMOS,Test.OMOS, bins, bins,'FaceColor','flat');
h.ShowEmptyBins = 'On';
h.DisplayStyle = 'tile';
h.EdgeAlpha = 0;

ax = gca;
ax.GridColor = [0.4 0.4 0.4];
ax.GridLineStyle = '--';
ax.GridAlpha = 0.5;
ax.XGrid = 'on';
ax.YGrid = 'on';
ax.Layer = 'top';
axis([1 5 1 5 0 1])
view(2)
% colormap(flipud(gray));
c = colorbar;
c.Label.String = 'Count';

xticks(1:5);
xticklabels({'Bad', 'Poor', 'Fair', 'Good', 'Excellent'})
yticks(1:5);
yticklabels({'Bad', 'Poor', 'Fair', 'Good', 'Excellent'})
xlabel('Subjective Raw MeanOS Score');
ylabel('Objective Score');

set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');

% print('plots/MATLAB/TIFF/All_Results_Hist_Mean', '-dtiff');
print('plots/MATLAB/EPSC/All_Results_Hist_Mean', '-depsc');
print('plots/MATLAB/PNG/All_Results_Hist_Mean', '-dpng');


%Test Confusion Matrix
train_csv = [folder_name 'Train.csv'];
Train = Import_Test_CSV(train_csv, 2, 4753);

figure('Position',[0 50 630 479]);
bins = linspace(1,5,40);
h = histogram2(Train.SMOS,Train.OMOS, bins, bins,'FaceColor','flat');
h.ShowEmptyBins = 'On';
h.DisplayStyle = 'tile';
h.EdgeAlpha = 0;

ax = gca;
ax.GridColor = [0.4 0.4 0.4];
ax.GridLineStyle = '--';
ax.GridAlpha = 0.5;
ax.XGrid = 'on';
ax.YGrid = 'on';
ax.Layer = 'top';
axis([1 5 1 5 0 1])
view(2)
% colormap(flipud(gray));
c = colorbar;
c.Label.String = 'Count';

xticks(1:5);
xticklabels({'Bad', 'Poor', 'Fair', 'Good', 'Excellent'})
yticks(1:5);
yticklabels({'Bad', 'Poor', 'Fair', 'Good', 'Excellent'})
xlabel('Subjective Raw MeanOS Score');
ylabel('Objective Score');

set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');

% print('plots/MATLAB/TIFF/All_Results_Hist_Mean', '-dtiff');
print('plots/MATLAB/EPSC/All_Results_Hist_Mean', '-depsc');
print('plots/MATLAB/PNG/All_Results_Hist_Mean', '-dpng');


%Validation Confusion Matrix
val_csv = [folder_name 'Val.csv'];
Val = Import_Test_CSV(val_csv, 2, 529);

figure('Position',[0 50 630 479]);
bins = linspace(1,5,40);
h = histogram2(Val.SMOS,Val.OMOS, bins, bins,'FaceColor','flat');
h.ShowEmptyBins = 'On';
h.DisplayStyle = 'tile';
h.EdgeAlpha = 0;

ax = gca;
ax.GridColor = [0.4 0.4 0.4];
ax.GridLineStyle = '--';
ax.GridAlpha = 0.5;
ax.XGrid = 'on';
ax.YGrid = 'on';
ax.Layer = 'top';
axis([1 5 1 5 0 1])
view(2)
% colormap(flipud(gray));
c = colorbar;
c.Label.String = 'Count';

xticks(1:5);
% xticklabels({'Bad', 'Poor', 'Fair', 'Good', 'Excellent'})
yticks(1:5);
% yticklabels({'Bad', 'Poor', 'Fair', 'Good', 'Excellent'})
xlabel('Subjective Raw MeanOS Score');
ylabel('Objective Score');

set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');

% print('plots/MATLAB/TIFF/All_Results_Hist_Mean', '-dtiff');
print('plots/MATLAB/EPSC/All_Results_Hist_Mean', '-depsc');
print('plots/MATLAB/PNG/All_Results_Hist_Mean', '-dpng');


%% Find the files that are furtherst away for Subjective score.

load('../Subjective_Testing/Results_Anon_No_Outliers.mat');

[~,I] = sort(abs(Test.SMOS-Test.OMOS),'descend');
for n = 1:20
    fprintf('SMOS: %.3f, OMOS: %.3f Delta: %.3f for %s\n',Test.SMOS(I(n)),Test.OMOS(I(n)),Test.OMOS(I(n))-Test.SMOS(I(n)),a(5280+I(n)).name)
end

figure
plot([a(5281:end).std_MOS],Test.OMOS-Test.SMOS,'.')
xlabel('MOS Standard Deviation')
ylabel('OMOS-SMOS')
figure
plot([a(5281:end).mean_MOS],Test.OMOS-Test.SMOS,'.')
xlabel('MOS')
ylabel('OMOS-SMOS')

mean(Test.SMOS(1:80)-Test.OMOS(1:80)) %0.0161 and 0.0102 for MeanOS and MeanOS Raw
mean(Test.SMOS(81:160)-Test.OMOS(81:160)) % -0.0055 and 0.0201 for MeanOS and MeanOS Raw
mean(Test.SMOS(161:240)-Test.OMOS(161:240)) % -0.1175 and -0.1913 for MeanOS and MeanOS Raw





%Compute confidence interval for all subjective scores

% x = randi(50, 1, 100);                      % Create Data
% SEM = std(x)/sqrt(length(x));               % Standard Error
% ts = tinv([0.025  0.975],length(x)-1);      % T-Score
% CI = mean(x) + ts*SEM;
addpath('../Functions');
load('../Subjective_Testing/Plotting_Data_Anon_No_Outliers.mat')

%MeanOS
RMSE = 0.4903;
rho = 0.8642;
% fid = fopen('log_Final.txt','a');
p = inv_prctile(rho,[u.pearson_corr_MeanOS_norm],'up');
fprintf('PCC Percentile for To Test MeanOS is %d\n', p);
p = inv_prctile(RMSE,[u.RMSE_norm],'down');
fprintf('RMSE Percentile for To Test MeanOS is %d\n', p);

% fclose(fid);

%MeanOS Raw
RMSE = 0.4740;
rho = 0.8586;
p = inv_prctile(rho,[u.pearson_corr_mean],'up');
fprintf('PCC Percentile for To Test MeanOS Raw is %d\n', p);
p = inv_prctile(RMSE,[u.RMSE],'down');
fprintf('RMSE Percentile for To Test MeanOS Raw is %d\n', p);


%Combination
RMSE = 0.4774;
rho = 0.8728;
p = inv_prctile(rho,[u.pearson_corr_MeanOS_norm],'up');
fprintf('PCC Percentile for Combination is %d\n', p);
p = inv_prctile(RMSE,[u.RMSE_norm],'down');
fprintf('RMSE Percentile for Combination is %d\n', p);
