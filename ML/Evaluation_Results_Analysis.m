% Analysing Network output

close all
clear all
clc
addpath('./Functions/');
addpath('../Functions/');
% eval = Import_Evaluation_CSV('log/Eval/2020-01-30_17-24-33_EVAL/Eval.csv', 2, 1301);
% eval2 = Import_Evaluation_CSV('log/Eval/2020-01-31_16-26-25_EVAL/Eval.csv', 2, 781);
% eval = Import_Evaluation_CSV('log/Eval/2020-02-05_14-37-57_EVAL/Eval.csv', 2, 2081);
% eval = Import_Evaluation_CSV('log/Eval/2020-02-04_21-30-41_EVAL/Eval.csv', 2, 2081);
% eval = Import_Evaluation_CSV('log/Eval/2020-02-19_17-12-46_EVAL/Eval.csv', 2, 2081);
% eval = Import_Evaluation_CSV('log/Eval/2020-02-28_15-43-14_EVAL275/Eval.csv', 2, 2081);

% eval = Import_Evaluation_CSV('log/Eval/2020-02-28_15-45-43_EVAL279/Eval.csv', 2, 2081);
% eval = Import_Evaluation_CSV('log/Eval/2020-02-28_15-46-02_EVAL519/Eval.csv', 2, 2081);
% eval = Import_Evaluation_CSV('log/Eval/2020-02-28_15-46-37_EVAL_Other_301/Eval.csv', 2, 2081);
% eval = Import_Evaluation_CSV('log/Eval/2020-03-06_18-06-20_EVAL_455/Eval.csv', 2, 2081);
% eval = Import_Evaluation_CSV('log/Eval/2020-03-11_17-43-42_EVAL_Combination/Eval.csv', 2, 2081);

% eval = Import_Evaluation_CSV('log/Eval/2020-04-06_15-17-14_EVAL_Combination/Eval.csv', 2, 2081); %MeanOS?
% eval = Import_Evaluation_CSV('log/Eval/2019-04-06_15-42-07_EVAL_Combination/Eval.csv', 2, 2081); %MeanOS Raw?
% eval = Import_Evaluation_CSV('log/Eval/2020-04-16_23-36-05_RMSE_TO_TEST_MEANOS/Eval.csv', 2, 2081); %MeanOS Raw?


% RMSE Best models
% eval = Import_Evaluation_CSV('log/Eval/2020-04-17_10-57-45_TO_TEST_INCL_Source_MEANOS/Eval.csv', 2, 5201); %MeanOS
% eval = Import_Evaluation_CSV('log/Eval/2020-04-19_15-35-50_RMSE_EVAL_COMBINATION/Eval.csv', 2, 5201); %Combination

% eval = Import_Evaluation_CSV('log/Eval/2020-04-23_17-03-32_TO_TEST_INCL_Source_MEANOS_Extended/Eval.csv', 2, 5601); %MeanOS
% eval = Import_Evaluation_CSV('log/Eval/2020-04-23_16-39-11_RMSE_EVAL_COMBINATION_EXTENDED/Eval.csv', 2, 5601); %Combination
% eval = Import_Evaluation_CSV('log/Eval/2020-04-29_17-26-58_PEAQBSmall_MEANOS_Extended/Eval.csv', 2, 5601); %PEAQB SmallMeanOS
% eval = Import_Evaluation_CSV('log/Eval/2020-05-15_15-23-28_TO_TEST_INCL_Source_MEANOS_Raw_Extended/Eval.csv', 2, 5601); %MeanOS Raw
eval = Import_Evaluation_CSV('log/Eval/EVAL_ALL/Eval_All.csv', 2, 6001); %MeanOS

% eval2 = Import_Evaluation_CSV('log/Eval/2020-02-03_15-49-20_EVAL/Eval.csv', 2, 781);
% eval = [eval;eval2];
category = {};
for n = 1:height(eval)
    name = split(char(eval.RefFile(n)),'/');
    switch char(name(end))
        case 'Alto_Sax_15.wav'
            category{n,1} = 'Solo';%'Solo_Harmonic';
        case 'Ardour_2.wav'
            category{n,1} = 'Music';
        case 'Brass_and_perc_9.wav'
            category{n,1} = 'Music';
        case 'Child_4.wav'
            category{n,1} = 'Voice';
        case 'Female_2.wav'
            category{n,1} = 'Voice';
        case 'Female_4.wav'
            category{n,1} = 'Voice';
        case 'Jazz_3.wav'
            category{n,1} = 'Music';
        case 'Male_16.wav'
            category{n,1} = 'Voice';
        case 'Male_22.wav'
            category{n,1} = 'Voice';
        case 'Male_6.wav'
            category{n,1} = 'Voice';
        case 'Mexican_Flute_02.wav'
            category{n,1} = 'Solo';%'Solo_Harmonic';
        case 'Oboe_piano_1.wav'
            category{n,1} = 'Music';
        case 'Ocarina_02.wav'
            category{n,1} = 'Solo';%'Solo_Harmonic';
        case 'Rock_4.wav'
            category{n,1} = 'Music';
        case 'Saxophones_6.wav'
            category{n,1} = 'Music';
        case 'Solo_flute_2.wav'
            category{n,1} = 'Solo';%'Solo_Harmonic';
        case 'Synth_Bass_2.wav'
            category{n,1} = 'Solo';%'Solo_Percussive';
        case 'Triangle_02.wav'
            category{n,1} = 'Solo';%'Solo_Percussive';
        case 'Woodwinds_4.wav'
            category{n,1} = 'Music';
        case 'You_mean_this_one.wav'
            category{n,1} = 'Music';
    end
end

eval = [eval category];
% TSM = [0.3268, 0.5620, 0.7641, 0.8375, 0.9109, 1, 1.241, 1.4543];
TSM = [0.2257,0.2635,0.3268,0.4444,0.5620,0.6631,0.7641,0.8008,0.8375,0.8742,0.9109,0.9555,1,1.1205,1.241,1.3477,1.4543,1.6272,1.8042,2.1632]; %All Eval Values

% TSM = eval.TSM(2:6);  %Offset because 145 is before 32
eval = sortrows(eval,[3 4 1],'ascend');

%Remove MOS for Elastique for Beta <0.25
eval.OMOS(801:820) = NaN;
Overall_order = [8,2,4,9,5,11,14,10,15,6,13,12,3,1,7];
Methods = {'DIPL','ESOLA','EL','FESOLA','FuzzyPV','HPTSM','IPL','NMFTSM','PV','PIPL','PSPL','SPL','WSOLA','SuTVS','uTVS'};
% chosen_methods = [8,6,12,3,5,14,2,4,7];
% 'PV','IPL','WSOLA','FESOLA','HPTSM','uTVS','Elastique','FuzzyPV','NMFTSM',
% lines_spec = {'o-','o-','o-','+-','+-','+-','--','--','--','x--','x--','x--','x--'};
% points_spec = {'o','o','o','+','+','+','.','.','.','x','x','x','x'};
points_spec= {'-^','-', '--d', '.-', '--^', '-x', '-+','--v', '-o', '-', '-', '-', '-*', '->', '-s'};
grey_lines= {'k-^','k-', 'k--d', 'k.-', 'k--^', 'k-x', 'k-+','k--v', 'k-o', 'k-', 'k-', 'k-', 'k-*', 'k->', 'k-s'};
% o+*.xsdv^ <>ph are the marker options
m = 1;
results = zeros(20,length(TSM),length(Methods)); % (source files,TSM,methods)
for l = 1:length(Methods)
    for k = 1:length(TSM)
        for n = 1:20 %Number of evaluation source files
            results(n,k,l) = eval.OMOS(m);
            m = m+1;
        end
    end
end




method_TSM_means = mean(results,1,'omitnan');
method_means = mean(method_TSM_means(:,[2:12 14:20],:));
% method_means = mean(method_TSM_means(:,[2:12 14:20],:));
for n = 1:length(method_means)
  fprintf('%s mean: %g\n',char(Methods(n)),method_means(:,:,n))
end
fprintf('\nMethods used in Subjective Testing\n')
% chosen_methods = [2,3,4,5,6,7,8,12,13];
% chosen_methods = [8,6,12,3,5,14,2,4,7]+1; %After adding Driedger IPL
% Methods = {'DrIPL','ESOLA','Elastique','FESOLA','FuzzyPV','HPTSM','IPL','NMFTSM','PV','Phavorit IPL','Phavorit SPL','SPL','WSOLA','uTVSSubj','uTVS'};
chosen_methods = [9,7,13,4,6,14,3,5,8,2,12,10,11,15,1];
figure('Position',[50 50 688 386])
hold on
for n = chosen_methods
   plot(TSM,method_TSM_means(:,:,n),points_spec{n})%,'LineWidth',1.2)
   fprintf('%s mean: %g\n',char(Methods(n)),method_means(:,:,n))
end
hold off
% title('Overall Means for TSM Methods')
xlabel('Time-Scale Ratio (\beta)')
ylabel('OMOS')
axis([0.2 2.2 1 5])
% legend(Methods(chosen_methods),'Location','NorthOutside','NumColumns',5)
overlay_text = text(1.7,1.3,'(d) All Signals');
overlay_text.FontSize = 12;
overlay_text.FontName = 'Times';
set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');
print('plots/MATLAB/TIFF/Method_Means_To_Test_Incl_Source_MeanOS', '-dtiff');
print('plots/MATLAB/EPSC/Method_Means_To_Test_Incl_Source_MeanOS', '-depsc');
print('plots/MATLAB/PNG/Method_Means_To_Test_Incl_Source_MeanOS', '-dpng');

figure('Position',[50 50 640 320])
figure(45)
subplot(2,2,1)
overall = reshape(method_TSM_means(:,:,:),size(method_TSM_means,2),size(method_TSM_means,3));
im = image(overall(:,Overall_order)','CDataMapping','scaled');
set(im,'AlphaData',~isnan(overall(:,Overall_order)'))
colormap(parula);
c = colorbar;
c.Label.String = 'OMOS';
xticks(1:size(TSM,2))
xticklabels(TSM);
xtickangle(45);
yticks(1:size(method_TSM_means,3))
yticklabels(Methods(Overall_order));
xlabel('Time-Scale Ratio (\beta)')
title('Overall')
set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');
% print('plots/MATLAB/TIFF/Overall_Image', '-dtiff');
% print('plots/MATLAB/EPSC/Overall_Image', '-depsc');
% print('plots/MATLAB/PNG/Overall_Image', '-dpng');

%% Histogram of TSM against MOS
% figure('Position',[517 401 627 401])
% h = histogram2(eval.TSM,eval.OMOS,[20 100],'FaceColor','flat');
% h.ShowEmptyBins = 'off';
% h.DisplayStyle = 'tile';
% h.EdgeAlpha = 0;
% 
% ax = gca;
% ax.GridColor = [0.4 0.4 0.4];
% ax.GridLineStyle = '--';
% ax.GridAlpha = 0.5;
% ax.XGrid = 'off';
% ax.YGrid = 'on';
% ax.Layer = 'top';
% view(2)
% axis([0.2 1.5 1 5])
% % colormap(flipud(gray));
% c = colorbar;
% c.Label.String = 'Count';
% xlabel('Time-Scale Ratio (\beta)')
% ylabel('OMOS')
% set(gca,...
%     'FontSize', 12, ...
%     'FontName', 'Times');
% print('plots/MATLAB/TIFF/TSM_OMOS_Combination', '-dtiff');
% print('plots/MATLAB/EPSC/TSM_OMOS_Combination', '-depsc');
% print('plots/MATLAB/PNG/TSM_OMOS_Combination', '-dpng');

%% Plot OMOS TSM curves for Music, Solo and Voice
eval = sortrows(eval,[6 3 4],'ascend');
T = length(TSM);
num_Music = 8;
num_Solo = 6;
% num_Solo_Harmonic = 4;
% num_Solo_Percussive = 2;
num_Voice = 6;
Music = zeros(num_Music,T,length(Methods));
Solo = zeros(num_Solo,T,length(Methods));
Voice = zeros(num_Voice,T,length(Methods));
Music_Offset = 1;
Solo_Offset = num_Music*T*length(Methods)+Music_Offset;
Voice_Offset = Solo_Offset+num_Solo*T*length(Methods);
%Music Plotting
for n = 1:length(Methods)
    for k = 1:T
%          ((n-1)*num_Music*T)+(k-1)*num_Music+1:((n-1)*num_Music*T)+(k-1)*num_Music+T
        Music(:,k,n) = eval.OMOS(((n-1)*num_Music*T)+(k-1)*num_Music+Music_Offset:((n-1)*num_Music*T)+(k-1)*num_Music+num_Music+Music_Offset-1);
        %eval.OMOS(T*num_Complex*(n-1)+1:T*num_Complex*(n-1)+T*num_Complex)
    end
end
% figure('Position',[517 401 627 401]);
% for n = 1:length(Methods)
%     boxplot(Music(:,:,n),TSM,'colors','k');
%     title(Methods(n))
%     xlabel('Time-Scale Ratio (\beta)')
%     ylabel('OMOS')
%     V = axis;
%     V(3) = 1;
%     V(4) = 5;
%     axis(V)
%     savename = sprintf('Music_%s_Boxplot',char(Methods(n)));
%     %Add code to save out each of the plots
%     set(gca,...
%     'FontSize', 12, ...
%     'FontName', 'Times');
% 
% print(['plots/MATLAB/EPSC/',savename,'.eps'], '-depsc');
% print(['plots/MATLAB/PNG/',savename,'.png'], '-dpng');
% end

% figure
% hold on
% for n = 1:length(Methods)
%     plot(eval.TSM(T*num_Music*(n-1)+1:T*num_Music*(n-1)+T*num_Music),eval.OMOS(T*num_Music*(n-1)+1:T*num_Music*(n-1)+T*num_Music),points_spec{n})
% end
% hold off
% legend(Methods,'location','bestoutside')
method_TSM_means_Music = mean(Music,1,'omitnan');
% figure('Position',[50 50 640 320])
figure(45)
subplot(2,2,2)
overall = reshape(method_TSM_means_Music(:,:,:),size(method_TSM_means_Music,2),size(method_TSM_means_Music,3));
im = image(overall(:,Overall_order)','CDataMapping','scaled');
set(im,'AlphaData',~isnan(overall(:,Overall_order)'))
colormap(parula);
c = colorbar;
c.Label.String = 'OMOS';
xticks(1:size(TSM,2))
xticklabels(TSM);
xtickangle(45);
yticks(1:size(method_TSM_means_Music,3))
yticklabels(Methods(Overall_order));
xlabel('Time-Scale Ratio (\beta)')
title('Music')
set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');
% print('plots/MATLAB/TIFF/Music_Image', '-dtiff');
% print('plots/MATLAB/EPSC/Music_Image', '-depsc');
% print('plots/MATLAB/PNG/Music_Image', '-dpng');


%Make the plot that does the mean of each column for each method
figure('Position',[50 50 688 464])
hold on
for n = chosen_methods
   plot(TSM,mean(Music(:,:,n)),points_spec{n})%,'LineWidth',1.2)
   fprintf('%s mean: %g\n',char(Methods(n)),method_means(:,:,n))
end
hold off
% title('Music Means for TSM Methods')
xlabel('Time-Scale Ratio (\beta)')
ylabel('OMOS')
axis([0.2 2.2 1 5])
legend(Methods(chosen_methods),'Location','NorthOutside','NumColumns',5)
overlay_text = text(1.7,1.3,'(a) Music Signals');
overlay_text.FontSize = 12;
overlay_text.FontName = 'Times';
set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');
print('plots/MATLAB/TIFF/Music_Means_To_Test_Incl_Source_MeanOS', '-dtiff');
print('plots/MATLAB/EPSC/Music_Means_To_Test_Incl_Source_MeanOS', '-depsc');
print('plots/MATLAB/PNG/Music_Means_To_Test_Incl_Source_MeanOS', '-dpng');

%Do the same for Solo and Voice
%Solo Plotting
for n = 1:length(Methods)
    for k = 1:T
%          ((n-1)*num_Solo*T)+(k-1)*num_Solo+1:((n-1)*num_Solo*T)+(k-1)*num_Solo+T
        Solo(:,k,n) = eval.OMOS(((n-1)*num_Solo*T)+(k-1)*num_Solo+Solo_Offset:((n-1)*num_Solo*T)+(k-1)*num_Solo+Solo_Offset-1+num_Solo);
        %eval.OMOS(T*num_Complex*(n-1)+1:T*num_Complex*(n-1)+T*num_Complex)
    end
end
% figure('Position',[1239 600 681 516]);
% for n = 1:length(Methods)
%     boxplot(Solo(:,:,n),TSM,'colors','k');
%     title(Methods(n))
%     xlabel('Time-Scale Ratio (\beta)')
%     ylabel('OMOS')
%     V = axis;
%     V(3) = 1;
%     V(4) = 5;
%     axis(V)
%     savename = sprintf('Solo_%s_Boxplot',char(Methods(n)));
%     %Add code to save out each of the plots
%     set(gca,...
%     'FontSize', 12, ...
%     'FontName', 'Times');
% 
% print(['plots/MATLAB/EPSC/',savename,'.eps'], '-depsc');
% print(['plots/MATLAB/PNG/',savename,'.png'], '-dpng');
% end

% figure
% hold on
% for n = 1:length(Methods)
%     plot(eval.TSM(T*num_Music*(n-1)+1:T*num_Music*(n-1)+T*num_Music),eval.OMOS(T*num_Music*(n-1)+1:T*num_Music*(n-1)+T*num_Music),points_spec{n})
% end
% hold off
% legend(Methods,'location','bestoutside')

method_TSM_means_Solo = mean(Solo,1,'omitnan');
figure('Position',[50 50 640 320])
figure(45)
subplot(2,2,3)
overall = reshape(method_TSM_means_Solo(:,:,:),size(method_TSM_means_Solo,2),size(method_TSM_means_Solo,3));
im = image(overall(:,Overall_order)','CDataMapping','scaled');
set(im,'AlphaData',~isnan(overall(:,Overall_order)'))
colormap(parula);
c = colorbar;
c.Label.String = 'OMOS';
xticks(1:size(TSM,2))
xticklabels(TSM);
xtickangle(45);
yticks(1:size(method_TSM_means_Solo,3))
yticklabels(Methods(Overall_order));
xlabel('Time-Scale Ratio (\beta)')
title('Solo')
set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');
% print('plots/MATLAB/TIFF/Solo_Image', '-dtiff');
% print('plots/MATLAB/EPSC/Solo_Image', '-depsc');
% print('plots/MATLAB/PNG/Solo_Image', '-dpng');

%Make the plot that does the mean of each column for each method
figure('Position',[50 50 688 386])
hold on
for n = chosen_methods
   plot(TSM,mean(Solo(:,:,n)),points_spec{n})%,'LineWidth',1.2)
   fprintf('%s mean: %g\n',char(Methods(n)),method_means(:,:,n))
end
hold off
% title('Solo Means for TSM Methods')
xlabel('Time-Scale Ratio (\beta)')
ylabel('OMOS')
axis([0.2 2.2 1 5])
% legend(Methods(chosen_methods),'Location','NorthOutside','NumColumns',5)
overlay_text = text(1.7,1.3,'(b) Solo Signals');
overlay_text.FontSize = 12;
overlay_text.FontName = 'Times';
set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');
print('plots/MATLAB/TIFF/Solo_Means_To_Test_Incl_Source_MeanOS', '-dtiff');
print('plots/MATLAB/EPSC/Solo_Means_To_Test_Incl_Source_MeanOS', '-depsc');
print('plots/MATLAB/PNG/Solo_Means_To_Test_Incl_Source_MeanOS', '-dpng');

%Voice Plotting
for n = 1:length(Methods)
    for k = 1:T
%          ((n-1)*num_Voice*T)+(k-1)*num_Voice+1:((n-1)*num_Voice*T)+(k-1)*num_Voice+T
        Voice(:,k,n) = eval.OMOS(((n-1)*num_Voice*T)+(k-1)*num_Voice+Voice_Offset:((n-1)*num_Voice*T)+(k-1)*num_Voice+Voice_Offset-1+num_Voice);
        %eval.OMOS(T*num_Complex*(n-1)+1:T*num_Complex*(n-1)+T*num_Complex)
    end
end
% figure('Position',[1239 600 681 516]);
% for n = 1:length(Methods)
%     boxplot(Voice(:,:,n),TSM,'colors','k');
%     title(Methods(n))
%     xlabel('Time-Scale Ratio (\beta)')
%     ylabel('OMOS')
%     V = axis;
%     V(3) = 1;
%     V(4) = 5;
%     axis(V)
%     savename = sprintf('Voice_%s_Boxplot',char(Methods(n)));
%     %Add code to save out each of the plots
%     set(gca,...
%     'FontSize', 12, ...
%     'FontName', 'Times');
% 
% print(['plots/MATLAB/EPSC/',savename,'.eps'], '-depsc');
% print(['plots/MATLAB/PNG/',savename,'.png'], '-dpng');
% end

% figure
% hold on
% for n = 1:length(Methods)
%     plot(eval.TSM(T*num_Voice*(n-1)+1:T*num_Voice*(n-1)+T*num_Voice),eval.OMOS(T*num_Voice*(n-1)+1:T*num_Voice*(n-1)+T*num_Voice),points_spec{n})
% end
% hold off
% legend(Methods,'location','bestoutside')

method_TSM_means_Voice = mean(Voice,1,'omitnan');
figure('Position',[50 50 640 320])
figure(45) 
subplot(2,2,4)
overall = reshape(method_TSM_means_Voice(:,:,:),size(method_TSM_means_Voice,2),size(method_TSM_means_Voice,3));
im = image(overall(:,Overall_order)','CDataMapping','scaled');
set(im,'AlphaData',~isnan(overall(:,Overall_order)'))
colormap(parula);
c = colorbar;
c.Label.String = 'OMOS';
xticks(1:size(TSM,2))
xticklabels(TSM);
xtickangle(45);
yticks(1:size(method_TSM_means_Voice,3))
yticklabels(Methods(Overall_order));
xlabel('Time-Scale Ratio (\beta)')
title('Voice')
set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');
set(gcf, 'Position', get(0, 'Screensize'));
print('plots/MATLAB/TIFF/Class_Means_Subplot', '-dtiff');
print('plots/MATLAB/EPSC/Class_Means_Subplot', '-depsc');
print('plots/MATLAB/PNG/Class_Means_Subplot', '-dpng');
% print('plots/MATLAB/TIFF/Voice_Image', '-dtiff');
% print('plots/MATLAB/EPSC/Voice_Image', '-depsc');
% print('plots/MATLAB/PNG/Voice_Image', '-dpng');

%Make the plot that does the mean of each column for each method
figure('Position',[50 50 688 386])
hold on
for n = chosen_methods
   plot(TSM,mean(Voice(:,:,n)),points_spec{n})%,'LineWidth',1.2)
   fprintf('%s mean: %g\n',char(Methods(n)),method_means(:,:,n))
end
hold off
% title('Voice Means for TSM Methods')
xlabel('Time-Scale Ratio (\beta)')
ylabel('OMOS')
axis([0.2 2.2 1 5])
% legend(Methods(chosen_methods),'Location','NorthOutside','NumColumns',5)
overlay_text = text(1.7,1.3,'(c) Voice Signals');
overlay_text.FontSize = 12;
overlay_text.FontName = 'Times';
set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');
print('plots/MATLAB/TIFF/Voice_Means_To_Test_Incl_Source_MeanOS', '-dtiff');
print('plots/MATLAB/EPSC/Voice_Means_To_Test_Incl_Source_MeanOS', '-depsc');
print('plots/MATLAB/PNG/Voice_Means_To_Test_Incl_Source_MeanOS', '-dpng');

% close all
%% Create Eval.tex table

addpath('..\..\External\');
% load('../Subjective_Testing/Plotting_Data_RMSE.mat');
%Create the array to be converted to a table

eval_data = zeros(size(Methods,2),24);
for n = 1:size(eval_data,1)
    eval_data(n,:) = [mean(results(:,:,n),1),mean(mean(Music(:,[2:12 14:20],n))),mean(mean(Solo(:,[2:12 14:20],n))),mean(mean(Voice(:,[2:12 14:20],n))),mean(mean(results(:,[2:12 14:20],n)))];
    configs{n} = strrep(char(Methods(n)),'_',' ');
end

%Sort the array
[~, I] = sort(eval_data(:,end),'ascend');


input.data = eval_data(I,:);

input.tablePlacement = 'ht';
input.tableColLabels = {'22.57\%','26.35\%','32.68\%','44.44\%','56.20\%', ...
                        '66.31\%','76.41\%','80.08\%','83.75\%','87.42\%', ... 
                        '91.09\%','95.55\%','100\%','112.05\%','124.1\%', ...
                        '134.77\%','145.43\%','162.72\%','180.42\%','216.32\%',...
                        'Music','Solo','Voice','Overall'};
input.tableRowLabels = configs(I);
input.dataFormat = {'%.3f'};
input.tableColumnAlignment = 'c';
input.tableBorders = 1;
input.tableCaption = 'Mean OMOS for each class of file and overall result. Means calculated without $\beta$ of 0.2257 and 1. Methods in order left to right are: NMFTSM, ESOLA, FESOLA, PV, FuzzyPV, Phavorit SPL, uTVS used in Subjective testing, Phavorit IPL, uTVS, HPTSM, WSOLA, SPL, Elastique, Driedger''s IPL and IPL.';
input.tableLabel = 'Eval';
input.makeCompleteLatexDocument = 0;
input.transposeTable = 1;
fprintf('Writing Eval.tex latex table\n')
fid = fopen('To_Test_Incl_Source_MeanOS_Eval_ALL.tex','w');
latex = JASAlatexTable(input,fid);
fclose(fid);


%% Sum Loss MeanPCC plot and Sum Loss PCC Difference plots
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

%% Sum Loss Histogram
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

%% MeanPCC Histogram
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

%% ANOVA Plots
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


%%
%Code for saving out each eval result
% d = input.data([1:13,15],:);
% save('To_Test_MeanOS.mat','d','-v7');

D = load('To_Test_MeanOS.mat');
TO_TEST_MEANOS = D.d;
D = load('To_Test_MeanOS_Raw.mat');
TO_TEST_MEANOS_RAW = D.d;
D = load('Combination_MeanOS.mat');
COMBINATION_MEANOS = D.d;
delta_MOS = TO_TEST_MEANOS-TO_TEST_MEANOS_RAW;
figure('Position',[50 50 627 313])
set(groot,'defaultAxesTickLabelInterpreter','latex'); 
im = image(1:20,1:14,delta_MOS(:,1:20),'CDataMapping','scaled');
set(im,'AlphaData',~isnan(delta_MOS(:,1:20)))
colormap(parula);
c = colorbar;
c.Label.String = '\Delta OMOS (Norm-Raw)';

% ylabel('OMOS(MeanOS Norm) - OMOS(MeanOS Raw)')
xlabel('Time-Scale Ratio (\beta)')
% ylabel(h,'\Delta OMOS (Norm-Raw)')
xticks(1:size(delta_MOS,2))
xticklabels(TSM);
xtickangle(45);
yticks(1:size(delta_MOS,1))
yticklabels(input.tableRowLabels([1:13,15]));
% ytickangle(45);

set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');
print('plots/MATLAB/TIFF/OMOS_Norm_Raw', '-dtiff');
print('plots/MATLAB/EPSC/OMOS_Norm_Raw', '-depsc');
print('plots/MATLAB/PNG/OMOS_Norm_Raw', '-dpng');

figure('Position',[50 50 640 320])
plot(TSM,mean(TO_TEST_MEANOS(:,1:20),'omitnan'))
hold on
plot(TSM,mean(TO_TEST_MEANOS_RAW(:,1:20),'omitnan'))
% plot(TSM,mean(COMBINATION_MEANOS(:,1:20),'omitnan'))
hold off
axis([0.2 2.2 1 5])
legend('Norm','Raw','Location','Best') %,'Combination'


%% Cohen's d calculation


for n = 1:15
    for k = 1:15
        d(n,k) = cohen(results(:,2,Overall_order(n)),results(:,2,Overall_order(k)));
    end
end

figure
image(1:15,1:15,abs(d),'CDataMapping','scaled');
colorbar;
xticks(1:size(Methods,2))
xticklabels(Methods(Overall_order));
xtickangle(90);
yticks(1:size(Methods,2))
yticklabels(Methods(Overall_order));