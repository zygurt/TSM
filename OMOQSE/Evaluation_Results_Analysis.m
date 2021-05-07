% Analysing Network output

close all
clear all
clc
addpath('./Functions/');
addpath('../Functions/');

% eval = Import_Eval_CSV('logs/Eval/2020-08-28_17-56-51CNN_MFCC_Delta_2020-08-23_15-11-52_1iteration/Eval.csv', 2, 6001); %1 Eval Passes
% eval = Import_Eval_CSV('logs/Eval/2020-08-28_17-57-41CNN_MFCC_Delta_2020-08-23_15-11-52_8iterations/Eval.csv', 2, 6001); %8 Eval Pass
% eval = Import_Eval_CSV('logs/Eval/2020-08-28_21-25-29CNN_MFCC_Delta_2020-08-23_15-11-52_16iterations/Eval.csv', 2, 6001); %16 Eval Pass
% eval_tex_name = 'Output/Tex/CNN_Eval_ALL.tex';

% eval = Import_Eval_CSV('logs/Eval/2020-08-31_16-16-02BGRU_FT_2020-08-31_15-07-55/Eval.csv', 2, 6001); %Seed 6
% eval2 = Import_Eval_CSV('logs/Eval/2020-09-01_13-44-46BGRU_FT_2020-08-30_01-54-09/Eval.csv', 2, 6001); %Seed 28
% eval3 = Import_Eval_CSV('logs/Eval/2020-08-31_14-32-18BGRU_FT_2020-08-31_12-07-23/Eval.csv', 2, 6001); %Seed 28

%Final BGRU FT evaluation
eval = Import_Eval_CSV('logs/Eval/2020-09-01_14-31-24BGRU_FT_2020-08-31_19-48-48/Eval.csv', 2, 6001); %Seed 28
eval_tex_name = 'Output/Tex/BGRU_Eval_ALL.tex';

%ALSO CHANGE THE FOLDER THAT THE IMAGES ARE SAVED INTO
% /Plots/CNN/
% /Plots/GRU/

% eval_corr12 = corr(eval.OMOS,eval2.OMOS);
% eval_corr13 = corr(eval.OMOS,eval3.OMOS);
% eval_corr23 = corr(eval2.OMOS,eval3.OMOS);
% 
% figure()
% h = histogram2(eval.OMOS,eval2.OMOS,[100 100],'FaceColor','flat');
% h.ShowEmptyBins = 'Off';
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
% axis([1,5,1,5])
% title('1-2')
% % colormap(flipud(gray));
% c = colorbar;
% c.Label.String = 'Count';
% 
% figure()
% h = histogram2(eval.OMOS,eval3.OMOS,[100 100],'FaceColor','flat');
% h.ShowEmptyBins = 'Off';
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
% axis([1,5,1,5])
% title('1-3')
% % colormap(flipud(gray));
% c = colorbar;
% c.Label.String = 'Count';
% 
% figure()
% h = histogram2(eval2.OMOS,eval3.OMOS,[100 100],'FaceColor','flat');
% h.ShowEmptyBins = 'Off';
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
% axis([1,5,1,5])
% title('2-3')
% % colormap(flipud(gray));
% c = colorbar;
% c.Label.String = 'Count';
% 
% figure()
% h = histogram2(eval3.TSM/100,eval3.OMOS,[100 100],'FaceColor','flat');
% h.ShowEmptyBins = 'Off';
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
% axis([0.1,2.3,1,5])
% title('eval3 vs TSM')
% % colormap(flipud(gray));
% c = colorbar;
% c.Label.String = 'Count';

%Find the matching Reference file name

Ref_File_Path = '../Subjective_Testing/Source/Objective/';
ref_filelist = rec_filelist(Ref_File_Path);

%Find the matching Source audio file
for n = 1:size(eval,1)
    match = 0;
    q = 1;
    test_name = char(eval.Filename(n));
%     test_name = split(eval.Filename(n),'"');
%     test_name = char(test_name(end-1));
    while ~match
        source = split(ref_filelist(q),'/');
        source = char(source(end));
        match = startsWith(test_name,source(1:end-4));
        if(test_name(length(source(1:end-3)))~= '_')
            match = 0;
        end
        if ~match
            q = q+1;
        end
    end
    eval.ref_file(n) = ref_filelist(q);
%     TSM = split(test_name,'_');
%     filelist.TSM_per(n) = str2double(TSM(end-1,1));
    methods = split(eval.Filename(n),'_');
    eval.method(n) = methods(end-2);
%     if(strcmp(eval.method(n),'IPL') || strcmp(eval.method(n),'SPL'))
%         eval.method(n) = strcat(methods(end-3),'_',methods(end-2));
%     end
%     m = methods(end-2);
%     m_minus1 = methods(end-3);
    if((strcmp(methods(end-2),'IPL') || strcmp(methods(end-2),'SPL')) && strcmp(methods(end-3),'Phavorit'))
        eval.method(n) = strcat(methods(end-3),'_',methods(end-2));
    end
end





category = {};
for n = 1:height(eval)
    name = split(char(eval.ref_file(n)),'/');
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
TSM = [0.2257,0.2635,0.3268,0.4444,0.5620,0.6631,0.7641,0.8008,0.8375,0.8742,0.9109,0.9555,1,1.1205,1.241,1.3477,1.4543,1.6272,1.8042,2.1632]; %All Eval Values
eval_orig = eval; %Set aside for later processing.
% TSM = eval.TSM(2:6);  %Offset because 145 is before 32
eval = sortrows(eval,[5 2 1],'ascend'); %(Method, TSM, Test Filename)

%Remove MOS for Elastique for Beta <0.25
eval.OMOS(801:820) = NaN;
Overall_order_a = [8,2,4,9,5,11,14,10,15,6,13,12,3,1,7];

Methods = {'DIPL','ESOLA','EL','FESOLA','FuzzyPV','HPTSM','IPL','NMFTSM','PV','PIPL','PSPL','SPL','WSOLA', 'uTVS', 'SuTVS'};
% chosen_methods = [8,6,12,3,5,14,2,4,7];
% 'PV','IPL','WSOLA','FESOLA','HPTSM','uTVS','Elastique','FuzzyPV','NMFTSM',
% lines_spec = {'o-','o-','o-','+-','+-','+-','--','--','--','x--','x--','x--','x--'};
% points_spec = {'o','o','o','+','+','+','.','.','.','x','x','x','x'};
points_spec= {'-^','-', '--d', '.-', '--^', '-x', '-+','--v', '-o', '-', '-', '-', '-*', '->', '-s'};
grey_lines= {'k-^','k-', 'k--d', 'k.-', 'k--^', 'k-x', 'k-+','k--v', 'k-o', 'k-', 'k-', 'k-', 'k-*', 'k->', 'k-s'};
line_colour = {[0,0.4470,0.7410],[0.8500, 0.3250, 0.0980],[0.9290, 0.6940, 0.1250],[0.4940, 0.1840, 0.5560],[0.4660, 0.6740, 0.1880],[0.3010, 0.7450, 0.9330],[0.6350, 0.0780, 0.1840]};
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


% OMOV = Import_FCN_Eval_Feat('../ML/data/Features/MOVs_Eval_20200622Interpolate_to_test.csv', 2, 6001);
% OMOV = horzcat(OMOV,table(eval.OMOS));
% feat_est = table2array(OMOV(:,9:end));
% feat_est(801:820,end) = 1;
% feat_est = table2array(sortrows(array2table(feat_est),1,'ascend'));
% 
% feat_corr_slow = abs(corr(feat_est(1:3600,:)));
% figure('Position',[1700 200 900 900])
% imshow(feat_corr_slow,'InitialMagnification','fit','colormap',parula)
% title('GRU OMOS and Hand Crafted Features, \beta < 1')
% colorbar
% f = sprintf('Output/Plots/GRU/OMOS_FCN_Feat_Slower_Correlation');
% print([f '.png'],'-dpng')
% print([f '.eps'],'-depsc')
% 
% feat_corr_fast = abs(corr(feat_est(3601:end,:)));
% figure('Position',[1700 200 900 900])
% imshow(feat_corr_fast,'InitialMagnification','fit','colormap',parula)
% title('GRU OMOS and Hand Crafted Features, \beta > 1')
% colorbar
% f = sprintf('Output/Plots/GRU/OMOS_FCN_Feat_Faster_Correlation');
% print([f '.png'],'-dpng')
% print([f '.eps'],'-depsc')
% 
% feat_corr_split = 0.5.*(feat_corr_slow+feat_corr_fast);
% figure('Position',[0 0 1149 900])
% imshow(feat_corr_split,'InitialMagnification','fit','colormap',parula)
% title('GRU OMOS and Hand Crafted Features, Average')
% colorbar
% f = sprintf('Output/Plots/GRU/OMOS_FCN_Feat_Average_Correlation');
% print([f '.png'],'-dpng')
% print([f '.eps'],'-depsc')


method_TSM_means = mean(results,1,'omitnan');
method_TSM_variance = var(results,1,'omitnan');
method_means = mean(method_TSM_means(:,[2:12 14:20],:));
overall = reshape(method_TSM_means(:,:,:),size(method_TSM_means,2),size(method_TSM_means,3));
[~,Overall_order] = sort(mean(overall([2:12,14:end],:),1),'ascend');
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
% print('plots/MATLAB/TIFF/Method_Means_To_Test_Incl_Source_MeanOS', '-dtiff');
%print('Output/Plots/GRU/EPSC/Overall_Means_GRU_MeanOS', '-depsc');
%print('Output/Plots/GRU/PNG/Overall_Means_GRU_MeanOS', '-dpng');


%Variance plotting example
figure()
colour_count = 0;
for n=chosen_methods
plot(TSM,method_TSM_means(:,:,n),points_spec{n}, 'Color', line_colour{1+mod(colour_count,length(line_colour))})
hold on
for k=1:length(TSM)
    plot([TSM(k),TSM(k)],[method_TSM_means(1,k,n)-method_TSM_variance(1,k,n),method_TSM_means(1,k,n)+method_TSM_variance(1,k,n)],points_spec{n},'Color', line_colour{1+mod(colour_count,length(line_colour))})
end
colour_count = colour_count+1;
end
hold off
xlabel('Time-Scale Ratio (\beta)')
ylabel('OMOS')
axis([0.2 2.2 1 5])
title('BGRU-FT Overall')
set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');

% figure('Position',[50 50 640 320])
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

% print('Output/Plots/GRU/EPSC/Overall_Image', '-depsc');
% print('Output/Plots/GRU/PNG/Overall_Image', '-dpng');

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
% print('Output/Plots/GRU/TIFF/TSM_OMOS_Combination', '-dtiff');
% print('Output/Plots/GRU/EPSC/TSM_OMOS_Combination', '-depsc');
% print('Output/Plots/GRU/PNG/TSM_OMOS_Combination', '-dpng');

%% Plot OMOS TSM curves for Music, Solo and Voice
eval = sortrows(eval,[6 5 2],'ascend'); %Var6(Class), Method, TSM
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
% print(['Output/Plots/GRU/EPSC/',savename,'.eps'], '-depsc');
% print(['Output/Plots/GRU/PNG/',savename,'.png'], '-dpng');
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
% print('Output/Plots/GRU/EPSC/Music_Image', '-depsc');
% print('Output/Plots/GRU/PNG/Music_Image', '-dpng');


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
legend(Methods(chosen_methods),'Location','NorthOutside')%,'NumColumns',5)
overlay_text = text(1.7,1.3,'(a) Music Signals');
overlay_text.FontSize = 12;
overlay_text.FontName = 'Times';
set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');

%print('Output/Plots/GRU/EPSC/Music_Means_GRU_MeanOS', '-depsc');
%print('Output/Plots/GRU/PNG/Music_Means_GRU_MeanOS', '-dpng');

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
% print(['Output/Plots/GRU/EPSC/',savename,'.eps'], '-depsc');
% print(['Output/Plots/GRU/PNG/',savename,'.png'], '-dpng');
% end

% figure
% hold on
% for n = 1:length(Methods)
%     plot(eval.TSM(T*num_Music*(n-1)+1:T*num_Music*(n-1)+T*num_Music),eval.OMOS(T*num_Music*(n-1)+1:T*num_Music*(n-1)+T*num_Music),points_spec{n})
% end
% hold off
% legend(Methods,'location','bestoutside')

method_TSM_means_Solo = mean(Solo,1,'omitnan');
% figure('Position',[50 50 640 320])
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

% print('Output/Plots/GRU/EPSC/Solo_Image', '-depsc');
% print('Output/Plots/GRU/PNG/Solo_Image', '-dpng');
fprintf('Difference between most methods at beta=0.87 for solo files is %g\n',max(overall(10,[1:7,9:end]))-min(overall(10,[1:7,9:end])))
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
% legend(Methods(chosen_methods),'Location','NorthOutside')%,'NumColumns'
overlay_text = text(1.7,1.3,'(b) Solo Signals');
overlay_text.FontSize = 12;
overlay_text.FontName = 'Times';
set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');

%print('Output/Plots/GRU/EPSC/Solo_Means_GRU_MeanOS', '-depsc');
%print('Output/Plots/GRU/PNG/Solo_Means_GRU_MeanOS', '-dpng');

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
% print(['Output/Plots/GRU/EPSC/',savename,'.eps'], '-depsc');
% print(['Output/Plots/GRU/PNG/',savename,'.png'], '-dpng');
% end

% figure
% hold on
% for n = 1:length(Methods)
%     plot(eval.TSM(T*num_Voice*(n-1)+1:T*num_Voice*(n-1)+T*num_Voice),eval.OMOS(T*num_Voice*(n-1)+1:T*num_Voice*(n-1)+T*num_Voice),points_spec{n})
% end
% hold off
% legend(Methods,'location','bestoutside')

method_TSM_means_Voice = mean(Voice,1,'omitnan');
% figure('Position',[50 50 640 320])
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
% print('Output/Plots/GRU/TIFF/Class_Means_Subplot', '-dtiff');
figure(45)
%print('Output/Plots/GRU/EPSC/Class_Means_Subplot', '-depsc');
%print('Output/Plots/GRU/PNG/Class_Means_Subplot', '-dpng');
% print('Output/Plots/GRU/EPSC/Voice_Image', '-depsc');
% print('Output/Plots/GRU/PNG/Voice_Image', '-dpng');

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
% legend(Methods(chosen_methods),'Location','NorthOutside')%,'NumColumns'
overlay_text = text(1.7,1.3,'(c) Voice Signals');
overlay_text.FontSize = 12;
overlay_text.FontName = 'Times';
set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');
%print('Output/Plots/GRU/EPSC/Voice_Means_GRU_MeanOS', '-depsc');
%print('Output/Plots/GRU/PNG/Voice_Means_GRU_MeanOS', '-dpng');

% close all
%% Create Eval.tex table

% addpath('..\..\External\');
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
fid = fopen(eval_tex_name,'w');
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
% print('Output/Plots/GRU/TIFF/Sum_Loss_MeanPCC', '-dtiff');
% print('Output/Plots/GRU/EPSC/Sum_Loss_MeanPCC', '-depsc');
% print('Output/Plots/GRU/PNG/Sum_Loss_MeanPCC', '-dpng');
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
% print('Output/Plots/GRU/TIFF/Sum_Loss_PCCDifference', '-dtiff');
% print('Output/Plots/GRU/EPSC/Sum_Loss_PCCDifference', '-depsc');
% print('Output/Plots/GRU/PNG/Sum_Loss_PCCDifference', '-dpng');
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
% print('Output/Plots/GRU/TIFF/Sum_Loss_Hist', '-dtiff');
% print('Output/Plots/GRU/EPSC/Sum_Loss_Hist', '-depsc');
% print('Output/Plots/GRU/PNG/Sum_Loss_Hist', '-dpng');

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
% print('Output/Plots/GRU/TIFF/MeanPCC_Hist', '-dtiff');
% print('Output/Plots/GRU/EPSC/MeanPCC_Hist', '-depsc');
% print('Output/Plots/GRU/PNG/MeanPCC_Hist', '-dpng');
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
% print('Output/Plots/GRU/TIFF/Anova_Sum_Loss', '-dtiff');
% print('Output/Plots/GRU/EPSC/Anova_Sum_Loss', '-depsc');
% print('Output/Plots/GRU/PNG/Anova_Sum_Loss', '-dpng');
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
% print('Output/Plots/GRU/TIFF/Anova_MeanPCC', '-dtiff');
% print('Output/Plots/GRU/EPSC/Anova_MeanPCC', '-depsc');
% print('Output/Plots/GRU/PNG/Anova_MeanPCC', '-dpng');
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
% print('Output/Plots/GRU/TIFF/Anova_Distance', '-dtiff');
% print('Output/Plots/GRU/EPSC/Anova_Distance', '-depsc');
% print('Output/Plots/GRU/PNG/Anova_Distance', '-dpng');
% % legend(legend_labels(1:end-1),'location','best')


%%
% %Code for saving out each eval result
% % When using MeanOS evaluation uncomment this line
% % d = input.data;
% % save('To_Test_MeanOS.mat','d','-v7');
% % When using MeanOS Raw evaluation uncomment this line
% % d = input.data;
% % save('To_Test_MeanOS_Raw.mat','d','-v7');
%
% D = load('To_Test_MeanOS.mat');
% TO_TEST_MEANOS = D.d;
% D = load('To_Test_MeanOS_Raw.mat');
% TO_TEST_MEANOS_RAW = D.d;
% % D = load('Combination_MeanOS.mat');
% % COMBINATION_MEANOS = D.d;
% delta_MOS = TO_TEST_MEANOS-TO_TEST_MEANOS_RAW;
% figure('Position',[50 50 627 313])
% set(groot,'defaultAxesTickLabelInterpreter','latex');
% im = image(1:24,1:15,delta_MOS(:,1:24),'CDataMapping','scaled');
% set(im,'AlphaData',~isnan(delta_MOS))
% colormap(parula);
% c = colorbar;
% c.Label.String = '\Delta OMOS (Norm-Raw)';
%
% % ylabel('OMOS(MeanOS Norm) - OMOS(MeanOS Raw)')
% xlabel('Time-Scale Ratio (\beta)')
% % ylabel(h,'\Delta OMOS (Norm-Raw)')
% xticks(1:size(delta_MOS,2))
% xticklabels([{TSM},'Music','Solo','Voice','Overall']);
% xtickangle(45);
% yticks(1:size(delta_MOS,1))
% yticklabels(input.tableRowLabels([1:13,15]));
% % ytickangle(45);
%
% set(gca,...
%     'FontSize', 12, ...
%     'FontName', 'Times');
% % print('Output/Plots/GRU/TIFF/OMOS_Norm_Raw', '-dtiff');
% print('Output/Plots/GRU/EPSC/OMOS_Norm_Raw', '-depsc');
% print('Output/Plots/GRU/PNG/OMOS_Norm_Raw', '-dpng');


% load('../Subjective_Testing/Plotting_Data_Anon_No_Outliers.mat');
% d = [a.mean_MOS_norm]-[a.mean_MOS];
% fprintf('The average difference between Subjective Norm and Raw scores is %g\n',mean(d));
% fprintf('The average difference between Norm and Raw Predictions is %g\n',mean(mean(delta_MOS,'omitnan')));
%
% figure('Position',[50 50 640 320])
% plot(TSM,mean(TO_TEST_MEANOS(:,1:20),'omitnan'))
% hold on
% plot(TSM,mean(TO_TEST_MEANOS_RAW(:,1:20),'omitnan'))
% % plot(TSM,mean(COMBINATION_MEANOS(:,1:20),'omitnan'))
% hold off
% axis([0.2 2.2 1 5])
% legend('Norm','Raw','Location','Best') %,'Combination'




%% Consideration of statistical significance between methods
% Cohen's d calculation
d = zeros(15,15,19);
TT2 = zeros(15,15,19);
T_val = zeros(15,15,19);
for ts = 2:20
for n = 1:15
    for k = 1:n
        d(n,k,ts) = cohen(results(:,ts,Overall_order(n)),results(:,ts,Overall_order(k)));
        TT2(n,k,ts) = ttest2(results(:,ts,Overall_order(n)),results(:,ts,Overall_order(k)),'alpha',0.1);
        if TT2(n,k,ts)
            if mean(results(:,ts,Overall_order(n))) > mean(results(:,ts,Overall_order(k)))
                T_val(n,k,ts) = 1;
            elseif mean(results(:,ts,Overall_order(n))) < mean(results(:,ts,Overall_order(k)))
                T_val(n,k,ts) = -1;
            end
        end
    end
end
end

figure
image(1:15,1:15,mean(abs(d),3),'CDataMapping','scaled');
title('Mean Absolute Cohen''s d for \beta>0.25')
xticks(1:size(Methods,2))
xticklabels(Methods(Overall_order));
xtickangle(90);
yticks(1:size(Methods,2))
yticklabels(Methods(Overall_order));
c = colorbar;
c.Label.String = 'Cohen''s d';
set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');
%print('Output/Plots/GRU/EPSC/Cohens_d', '-depsc');
%print('Output/Plots/GRU/PNG/Cohens_d', '-dpng');

figure
image(1:15,1:15,sum(T_val,3),'CDataMapping','scaled');
title('Sum tTest H (A>B=1, A<B=-1, Else=0) for \beta>0.25')
xticks(1:size(Methods,2))
xticklabels(Methods(Overall_order));
xtickangle(90);
xlabel('B')
yticks(1:size(Methods,2))
yticklabels(Methods(Overall_order));
ylabel('A')
c = colorbar;
c.Label.String = '\Sigma(H=\pm1 or 0)';
set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');
%print('Output/Plots/GRU/EPSC/Sum_tTest2_H', '-depsc');
%print('Output/Plots/GRU/PNG/Sum_tTest2_H', '-dpng');

for n = 1:15
    for k = 1:15
        [ttest_H(n,k),ttest_P(n,k)] = ttest2(reshape(results(:,[2:12,14:20],Overall_order(n)),[360,1]), ...
                                             reshape(results(:,[2:12,14:20],Overall_order(k)),[360,1]), ...
                                             'alpha',0.05);
        mean_diff(n,k) = method_means(1,1,Overall_order(n))-method_means(1,1,Overall_order(k));
    end
end

figure
image(1:15,1:15,ttest_H,'CDataMapping','scaled');
title('tTest2(A,B) H for All Evaluation Files')
xticks(1:size(Methods,2))
xticklabels(Methods(Overall_order));
xtickangle(90);
xlabel('B')
yticks(1:size(Methods,2))
yticklabels(Methods(Overall_order));
ylabel('A')
c = colorbar;
c.Label.String = 'H (Reject Equal Means)';
set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');
%print('Output/Plots/GRU/EPSC/Overall_tTest2_H', '-depsc');
%print('Output/Plots/GRU/PNG/Overall_tTest2_H', '-dpng');

figure
image(1:15,1:15,ttest_P,'CDataMapping','scaled');
title('tTest2(A,B) P Value for All Evaluation Files')
xticks(1:size(Methods,2))
xticklabels(Methods(Overall_order));
xtickangle(90);
xlabel('B')
yticks(1:size(Methods,2))
yticklabels(Methods(Overall_order));
ylabel('A')
c = colorbar;
c.Label.String = 'P Value';

set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');
%print('Output/Plots/GRU/EPSC/Overall_tTest2_P', '-depsc');
%print('Output/Plots/GRU/PNG/Overall_tTest2_P', '-dpng');

%Masked P values

figure
image(1:15,1:15,(-1*ttest_H+1).*ttest_P,'CDataMapping','scaled');
% title('tTest2(A,B) P Value for All Evaluation Files')
xticks(1:size(Methods,2))
xticklabels(Methods(Overall_order));
xtickangle(90);
xlabel('B')
yticks(1:size(Methods,2))
yticklabels(Methods(Overall_order));
ylabel('A')
c = colorbar;
c.Label.String = 'P Value';

set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');
%print('Output/Plots/GRU/EPSC/Masked_P_GRU', '-depsc');
%print('Output/Plots/GRU/PNG/Masked_P_GRU', '-dpng');





figure
image(1:15,1:15,mean_diff,'CDataMapping','scaled');
title('Overall Mean Difference (A-B)')
xticks(1:size(Methods,2))
xticklabels(Methods(Overall_order));
xtickangle(90);
xlabel('B')
yticks(1:size(Methods,2))
yticklabels(Methods(Overall_order));
ylabel('A')
c = colorbar;
c.Label.String = 'A-B';

set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');
%print('Output/Plots/GRU/EPSC/Overall_Mean_Difference', '-depsc');
%print('Output/Plots/GRU/PNG/Overall_Mean_Difference', '-dpng');

%Threshold [0.08991, 0.106
threshold = 0.098;
figure
im=image(1:15,1:15,mean_diff,'CDataMapping','scaled');
set(im,'AlphaData',abs(mean_diff)>threshold)
title('Overall Mean Difference (A-B)>0.098')
xticks(1:size(Methods,2))
xticklabels(Methods(Overall_order));
xtickangle(90);
xlabel('B')
yticks(1:size(Methods,2))
yticklabels(Methods(Overall_order));
ylabel('A')
c = colorbar;
c.Label.String = 'A-B';

set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');
%print('Output/Plots/GRU/EPSC/Threshold_Overall_Mean_Difference', '-depsc');
%print('Output/Plots/GRU/PNG/Threshold_Overall_Mean_Difference', '-dpng');

% %Top block
% fprintf('Top Block\n')
% fprintf('Minimum difference: %g\n',min(min(mean_diff(13:15,13:15))))
% fprintf('Maximum difference: %g\n',max(max(mean_diff(13:15,13:15))))
%
% fprintf('Difference to next block: %g\n',method_means(1,1,))
%
% fprintf('Middle Block\n')
% fprintf('Minimum difference: %g\n',min(min(mean_diff(6:12,6:12))))
% fprintf('Maximum difference: %g\n',max(max(mean_diff(6:12,6:12))))
%


%% Distributions of eval results

figure('Position',[0 0 500 300])
histogram(eval.OMOS,50)
title('Distribution of Evaluation Estimates')
xlabel('OMOS')
ylabel('Count')
set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');
%print('Output/Plots/GRU/EPSC/Eval_Distribution', '-depsc');
%print('Output/Plots/GRU/PNG/Eval_Distribution', '-dpng');

figure('Position',[0 0 500 300])
eval_sort = sortrows(eval,[2,3],'ascend');
eval_sort.OMOS(281:300) = 1;
% h = histogram2(eval_sort.TSM(1:3900)/100,eval_sort.OMOS(1:3900),[50 50],'FaceColor','flat');
h = histogram2(eval_sort.TSM/100,eval_sort.OMOS,[50 50],'FaceColor','flat');
h.ShowEmptyBins = 'Off';
h.DisplayStyle = 'tile';
h.EdgeAlpha = 0;
view(2)
title('Evaluation Estimates')
xlabel('Time-Scale Ratio (\beta)')
ylabel('OMOS')
colorbar
set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');
%print('Output/Plots/GRU/EPSC/Evaluation_TSM', '-depsc');
%print('Output/Plots/GRU/PNG/Evaluation_TSM', '-dpng');


%% Comparison to FGRU output.
addpath('../ML/Functions/');
eval_fcnn = Import_Evaluation_CSV('../ML/logs/Eval/2020-06-23_14-10-46_TO_TEST_Source/Eval.csv', 2, 6001);
for n = 1:height(eval_fcnn)
    temp = split(eval_fcnn.TestFile(n),'/');
    eval_fcnn.fname(n) = temp(end);
    
end
% eval_sort = sortrows(eval,[5,2],'ascend');
% eval_sort.OMOS(801:820) = 1;
eval_fcnn_sort = sortrows(eval_fcnn,6,'ascend');
eval_orig_sort = sortrows(eval_orig,1,'ascend');
corr_fcnn = corr(eval_fcnn_sort.OMOS,eval_orig_sort.OMOS);
fprintf('Correlation between FGRU OMOS and OMOQSE: %g\n',corr_fcnn)

figure('Position',[0 50 630 479]);
bins = linspace(1,5,40);
h = histogram2(eval_fcnn_sort.OMOS,eval_orig_sort.OMOS, bins, bins,'FaceColor','flat');
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

% title('FGRU OMOS vs GRU OMOS')
xlabel('OMOQDE OMOS')
ylabel('GRU OMOQSE OMOS')
xticks(1:5);
xticklabels({'Bad', 'Poor', 'Fair', 'Good', 'Excellent'})
yticks(1:5);
yticklabels({'Bad (1)', 'Poor (2)', 'Fair (3)', 'Good (4)', 'Excellent (5)'})
% colormap(flipud(gray));
c = colorbar;
c.Label.String = 'Count';
set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');
print('Output/Plots/GRU/EPSC/OMOQ_vs_GRU_OMOQSE', '-depsc');
print('Output/Plots/GRU/PNG/OMOQ_vs_GRU_OMOQSE', '-dpng');