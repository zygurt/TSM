%Generate Additional Plots
close all
clear all
clc

addpath("Functions")
%Confusion Matrix for Test Train Val
fig_pos = [114 351 580 403];

%% ----------  BGRU plotting -------------------

load_folder = "models/GRU/2020-08-31_19-48-48";  %Epoch 18

Test = import_GRU_TestTrainVal_file(strcat(load_folder, "/Test.csv"), 2, 241);

figure('Position',fig_pos);
bins = linspace(1,5,40);
h = histogram2(Test.SMOS,Test.OMOS18, bins, bins,'FaceColor','flat');
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
yticklabels({'Bad (1)', 'Poor (2)', 'Fair (3)', 'Good (4)', 'Excellent (5)'})
xlabel('SMOS');
ylabel('BGRU OMOS');

set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');

print('Output/Plots/EPSC/GRU_Test_Confusion', '-depsc');
print('Output/Plots/PNG/GRU_Test_Confusion', '-dpng');


Train = import_GRU_TestTrainVal_file(strcat(load_folder, "/Train.csv"), 2, 4753);

figure('Position',fig_pos);
bins = linspace(1,5,40);
h = histogram2(Train.SMOS,Train.OMOS18, bins, bins,'FaceColor','flat');
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
yticklabels({'Bad (1)', 'Poor (2)', 'Fair (3)', 'Good (4)', 'Excellent (5)'})
xlabel('SMOS');
ylabel('BGRU OMOS');

set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');

print('Output/Plots/EPSC/GRU_Train_Confusion', '-depsc');
print('Output/Plots/PNG/GRU_Train_Confusion', '-dpng');

hold on
plot3(Test.SMOS,Test.OMOS18,ones(size(Test.OMOS18)),'r.', 'MarkerSize',10)
hold off

print('Output/Plots/EPSC/GRU_Train_Test_Overlay', '-depsc');
print('Output/Plots/PNG/GRU_Train_Test_Overlay', '-dpng');

Val = import_GRU_TestTrainVal_file(strcat(load_folder, "/Val.csv"), 2, 529);

figure('Position',fig_pos);
bins = linspace(1,5,40);
h = histogram2(Val.SMOS,Val.OMOS18, bins, bins,'FaceColor','flat');
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
yticklabels({'Bad (1)', 'Poor (2)', 'Fair (3)', 'Good (4)', 'Excellent (5)'})
xlabel('SMOS');
ylabel('BGRU OMOS');

set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');

print('Output/Plots/EPSC/GRU_Val_Confusion', '-depsc');
print('Output/Plots/PNG/GRU_Val_Confusion', '-dpng');



%% ----------  CNN plotting -------------------


load_folder = "models/CNN/2020-08-23_15-11-52";  %Epoch 89
chosen_epoch = 89;

Test = import_CNN_TestTrainVal_file(strcat(load_folder, "/Test.csv"), 2, 241);

figure('Position',fig_pos);
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
yticklabels({'Bad (1)', 'Poor (2)', 'Fair (3)', 'Good (4)', 'Excellent (5)'})
xlabel('SMOS');
ylabel('CNN OMOS');

set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');

print('Output/Plots/EPSC/CNN_Test_Confusion', '-depsc');
print('Output/Plots/PNG/CNN_Test_Confusion', '-dpng');


Train = import_CNN_TestTrainVal_file(strcat(load_folder, "/Train.csv"), 2, 4753);

figure('Position',fig_pos);
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
yticklabels({'Bad (1)', 'Poor (2)', 'Fair (3)', 'Good (4)', 'Excellent (5)'})
xlabel('SMOS');
ylabel('CNN OMOS');

set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');

print('Output/Plots/EPSC/CNN_Train_Confusion', '-depsc');
print('Output/Plots/PNG/CNN_Train_Confusion', '-dpng');



hold on
plot3(Test.SMOS,Test.OMOS,ones(size(Test.OMOS)),'r.', 'MarkerSize',10)
hold off

print('Output/Plots/EPSC/CNN_Train_Test_Overlay', '-depsc');
print('Output/Plots/PNG/CNN_Train_Test_Overlay', '-dpng');



Val = import_CNN_TestTrainVal_file(strcat(load_folder, "/Val.csv"), 2, 529);

figure('Position',fig_pos);
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
xticklabels({'Bad', 'Poor', 'Fair', 'Good', 'Excellent'})
yticks(1:5);
yticklabels({'Bad (1)', 'Poor (2)', 'Fair (3)', 'Good (4)', 'Excellent (5)'})
xlabel('SMOS');
ylabel('CNN OMOS');

set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');

print('Output/Plots/EPSC/CNN_Val_Confusion', '-depsc');
print('Output/Plots/PNG/CNN_Val_Confusion', '-dpng');







