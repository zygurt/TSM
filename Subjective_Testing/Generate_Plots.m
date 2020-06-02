%Generate Plots
close all
clear all
clc

%%
addpath('../Functions');
load('Results_Anon_No_Outliers.mat')
% load('Results_v8_MAD_STD_1st_outliers_removed.mat')
% load('Results_v8_MAD_STD_2nd_outliers_removed.mat')
load('Full_Source_filelist.mat');
recalculate = 1;
fid = fopen('log_Anon.txt','a');
fprintf(fid,'\n\n%s\n',date);
fclose(fid);
TSM = [38, 44, 53, 65, 78, 82, 99, 138, 166, 192];
TSM_methods = {'PV','IPL','WSOLA','FESOLA','HPTSM','uTVS'};

if recalculate == 1
    fprintf('Recalculating Plotting Data\n')
    %For each source file
    for n = 1:length(a)
        %Split out the file
        fname_res = strsplit(strrep(a(n).name,'\','/'),'/');
        fname_res = char(fname_res{3});

        %Find the method (Check this for IPL PV)
        fname_res_method = strsplit(fname_res,'_');
        fname_res_method = char(fname_res_method{end-2});

        %Find the speed
        fname_res_speed = strsplit(fname_res,'_');
        fname_res_speed = floor(str2double(char(fname_res_speed{end-1})));

        array_pos = find(TSM == fname_res_speed);

        %Find the appropriate source file and put the mean_MOS into a vector as
        %filelist(n).method(Time scale)
        match = 0;
        s_index = 1;
        while ~match
            fname_source = strsplit(filelist(s_index).location,'/');
            fname_source_file = char(fname_source{end});
            fname_source_file = fname_source_file(1:end-4);

            source = fname_source_file;
            match = strncmp(fname_res,source,length(source));
            if(fname_res(length(source)+1)~= '_')
                match = 0;
            end
            if ~match
                s_index = s_index+1;
            end
        end
        if ~(strcmp(fname_res_method, 'FuzzyTSM') || strcmp(fname_res_method, 'NMFTSM') || strcmp(fname_res_method, 'Elastique'))
            %Add the Opinion score to the master list
            filelist(s_index).(fname_res_method)(array_pos) = a(n).mean_MOS_norm;
            filename = split(filelist(s_index).location,'/');
            a(n).cat = char(filename{2});
        else
            if ~isfield(filelist,'FuzzyTSM')
                filelist(28).FuzzyTSM = [];
            end
            if ~isfield(filelist,'NMFTSM')
                filelist(28).NMFTSM = [];
            end
            if ~isfield(filelist,'Elastique')
                filelist(28).Elastique = [];
            end
            filelist(s_index).(fname_res_method) = [filelist(s_index).(fname_res_method) a(n).mean_MOS_norm];
            filename = split(filelist(s_index).location,'/');
            name = char(filename{3});
            switch name
                case 'Alto_Sax_15.wav'
                    filelist(s_index).ObjCat = 'Solo';
                    a(n).cat = 'Solo';
                case 'Ardour_2.wav'
                    filelist(s_index).ObjCat = 'Music';
                    a(n).cat = 'Music';
                case 'Brass_and_perc_9.wav'
                    filelist(s_index).ObjCat = 'Music';
                    a(n).cat = 'Music';
                case 'Child_4.wav'
                    filelist(s_index).ObjCat = 'Voice';
                    a(n).cat = 'Voice';
                case 'Female_2.wav'
                    filelist(s_index).ObjCat = 'Voice';
                    a(n).cat = 'Voice';
                case 'Female_4.wav'
                    filelist(s_index).ObjCat = 'Voice';
                    a(n).cat = 'Voice';
                case 'Jazz_3.wav'
                    filelist(s_index).ObjCat = 'Music';
                    a(n).cat = 'Music';
                case 'Male_16.wav'
                    filelist(s_index).ObjCat = 'Voice';
                    a(n).cat = 'Voice';
                case 'Male_22.wav'
                    filelist(s_index).ObjCat = 'Voice';
                    a(n).cat = 'Voice';
                case 'Male_6.wav'
                    filelist(s_index).ObjCat = 'Voice';
                    a(n).cat = 'Voice';
                case 'Mexican_Flute_02.wav'
                    filelist(s_index).ObjCat = 'Solo';
                    a(n).cat = 'Solo';
                case 'Oboe_piano_1.wav'
                    filelist(s_index).ObjCat = 'Music';
                    a(n).cat = 'Music';
                case 'Ocarina_02.wav'
                    filelist(s_index).ObjCat = 'Solo';
                    a(n).cat = 'Solo';
                case 'Rock_4.wav'
                    filelist(s_index).ObjCat = 'Music';
                    a(n).cat = 'Music';
                case 'Saxophones_6.wav'
                    filelist(s_index).ObjCat = 'Music';
                    a(n).cat = 'Music';
                case 'Solo_flute_2.wav'
                    filelist(s_index).ObjCat = 'Solo';
                    a(n).cat = 'Solo';
                case 'Synth_Bass_2.wav'
                    filelist(s_index).ObjCat = 'Solo';
                    a(n).cat = 'Solo';
                case 'Triangle_02.wav'
                    filelist(s_index).ObjCat = 'Solo';
                    a(n).cat = 'Solo';
                case 'Woodwinds_4.wav'
                    filelist(s_index).ObjCat = 'Music';
                    a(n).cat = 'Music';
                case 'You_mean_this_one.wav'
                    filelist(s_index).ObjCat = 'Voice';
                    a(n).cat = 'Voice';
            end



        end
    end

    %Add the file Category to the master list
    for n = 1:length(filelist)
        fname_source = strsplit(filelist(n).location,'/');
        filelist(n).file_cat = char(fname_source{2});
    end


    HPTSM = zeros(88, length(TSM));
    PV = zeros(88, length(TSM));
    FESOLA = zeros(88, length(TSM));
    IPL = zeros(88, length(TSM));
    WSOLA = zeros(88, length(TSM));
    uTVS = zeros(88, length(TSM));


    for n = 1:27
        for k = 1:length(TSM)
            HPTSM(n,k) = filelist(n).HPTSM(k);
            PV(n,k) = filelist(n).PV(k);
            FESOLA(n,k) = filelist(n).FESOLA(k);
            WSOLA(n,k) = filelist(n).WSOLA(k);
            IPL(n,k) = filelist(n).IPL(k);
            uTVS(n,k) = filelist(n).uTVS(k);
        end
    end
    for n = 48:108
        for k = 1:length(TSM)
            HPTSM(n-20,k) = filelist(n).HPTSM(k);
            PV(n-20,k) = filelist(n).PV(k);
            FESOLA(n-20,k) = filelist(n).FESOLA(k);
            WSOLA(n-20,k) = filelist(n).WSOLA(k);
            IPL(n-20,k) = filelist(n).IPL(k);
            uTVS(n-20,k) = filelist(n).uTVS(k);
        end
    end

    for k = 1:length(TSM)
        var_name = sprintf('TSM%d',k);
        TSM_res.(var_name) = [PV(:,k) IPL(:,k) WSOLA(:,k) FESOLA(:,k) HPTSM(:,k) uTVS(:,k)];
    end


    % Determine the kind of file for each source
    Num_Music = 0;
    Num_Solo = 0;
    Num_Voice = 0;
    Num_Objective = 0;

    for n = 1:length(filelist)
        if strcmp(filelist(n).file_cat,'Music')
            Num_Music = Num_Music+1;
        elseif strcmp(filelist(n).file_cat,'Solo')
            Num_Solo = Num_Solo+1;
        elseif strcmp(filelist(n).file_cat,'Voice')
            Num_Voice = Num_Voice+1;
        elseif strcmp(filelist(n).file_cat,'Objective')
            Num_Objective = Num_Objective+1;
        end
    end

    %Group file results based on kind of file

    %For all the a(n).name
    %Find the source file type
    %Add the a(n).mean_MOS_norm to the appropriate array


    for n = 1:27
        for k = 1:length(TSM)
            for m = 1:length(TSM_methods)
                if strcmp(filelist(n).file_cat,'Music')
                    Music.TSM_method(n,k,m) = filelist(n).(TSM_methods{m})(k);
                    Music.type(n,m,k) = filelist(n).(TSM_methods{m})(k);
                elseif strcmp(filelist(n).file_cat,'Solo')
                    Solo.TSM_method(n-(Num_Music),k,m) = filelist(n).(TSM_methods{m})(k);
                    Solo.type(n-(Num_Music),m,k) = filelist(n).(TSM_methods{m})(k);
                elseif strcmp(filelist(n).file_cat,'Voice')
                    Voice.TSM_method(n-(Num_Music+Num_Solo),k,m) = filelist(n).(TSM_methods{m})(k);
                    Voice.type(n-(Num_Music+Num_Solo),m,k) = filelist(n).(TSM_methods{m})(k);
                end
            end
        end
    end


    for n = 28:88
        for k = 1:length(TSM)
            for m = 1:length(TSM_methods)
                if strcmp(filelist(n+20).file_cat,'Music')
                    Music.TSM_method(n,k,m) = filelist(n+20).(TSM_methods{m})(k);
                    Music.type(n,m,k) = filelist(n+20).(TSM_methods{m})(k);
                elseif strcmp(filelist(n+20).file_cat,'Solo')
                    Solo.TSM_method(n-(Num_Music),k,m) = filelist(n+20).(TSM_methods{m})(k);
                    Solo.type(n-(Num_Music),m,k) = filelist(n+20).(TSM_methods{m})(k);
                elseif strcmp(filelist(n+20).file_cat,'Voice')
                    Voice.TSM_method(n-(Num_Music+Num_Solo),k,m) = filelist(n+20).(TSM_methods{m})(k);
                    Voice.type(n-(Num_Music+Num_Solo),m,k) = filelist(n+20).(TSM_methods{m})(k);
                end
            end
        end
    end

    Music.type_mean = mean(Music.type,3);
    Solo.type_mean = mean(Solo.type,3);
    Voice.type_mean = mean(Voice.type,3);

    for n = 1:27
        for t = 1:length(TSM)
            anova2_data((n-1)*length(TSM)+t,1) = filelist(n).PV(t);
            anova2_data((n-1)*length(TSM)+t,2) = filelist(n).IPL(t);
            anova2_data((n-1)*length(TSM)+t,3) = filelist(n).WSOLA(t);
            anova2_data((n-1)*length(TSM)+t,4) = filelist(n).FESOLA(t);
            anova2_data((n-1)*length(TSM)+t,5) = filelist(n).HPTSM(t);
            anova2_data((n-1)*length(TSM)+t,6) = filelist(n).uTVS(t);
        end
    end

    for n = 28:88
        for t = 1:length(TSM)
            anova2_data((n-1)*length(TSM)+t,1) = filelist(n+20).PV(t);
            anova2_data((n-1)*length(TSM)+t,2) = filelist(n+20).IPL(t);
            anova2_data((n-1)*length(TSM)+t,3) = filelist(n+20).WSOLA(t);
            anova2_data((n-1)*length(TSM)+t,4) = filelist(n+20).FESOLA(t);
            anova2_data((n-1)*length(TSM)+t,5) = filelist(n+20).HPTSM(t);
            anova2_data((n-1)*length(TSM)+t,6) = filelist(n+20).uTVS(t);
        end
    end


    save('Plotting_Data_Anon_No_Outliers.mat','a','res_filelist','u','filelist','Music','Solo','Voice','PV','IPL','WSOLA','FESOLA','HPTSM','uTVS','TSM_res','anova2_data')
else
    load('Plotting_Data_Anon_No_Outliers.mat')
end





%% ------------------------Plot all the responses MEAN OPINION SCORE--------------------------
%Plot all the normalised responses
% fprintf('Plotting All Responses Mean\n')
% figure('Position',[0 0 700 500])
% All = [a.mean_MOS_norm];
% [~,I] = sort(All);
% hold on
% for n = 1:length(All)
%     plot(n*ones(size(a(I(n)).MOS_norm)),a(I(n)).MOS_norm,'.r');
%     plot(n,a(I(n)).mean_MOS_norm,'xk');
% end
% hold off
% title_text = sprintf('All Opinion Scores Ordered by Ascending MeanOS');
% title(title_text)
% xlabel('File')
% ylabel('Opinion Score')
% axis('tight')
%
% set(gca,...
%     'FontSize', 12, ...
%     'FontName', 'Times');
%
%
% print('Plots/TIFF/All_Results_Mean', '-dtiff');
% print('Plots/EPSC/All_Results_Mean', '-depsc');
% print('Plots/SVG/All_Results_Mean', '-dsvg');

%% ------------------------Plot all the responses MEDIAN OPINION SCORE--------------------------
%Plot all the normalised responses
% fprintf('Plotting All Responses Median\n')
% figure('Position',[0 0 700 500])
% All = [a.median_MOS_norm];
% [~,I] = sort(All);
% hold on
% for n = 1:length(All)
%     plot(n*ones(size(a(I(n)).MOS_norm)),a(I(n)).MOS_norm,'.r');
%     plot(n,a(I(n)).median_MOS_norm,'xk');
% end
% hold off
% title_text = sprintf('All Opinion Scores Ordered by Ascending MedianOS');
% title(title_text)
% xlabel('File')
% ylabel('MOS')
% axis('tight')
%
% set(gca,...
%     'FontSize', 12, ...
%     'FontName', 'Times');
%
% print('Plots/TIFF/All_Results_Median', '-dtiff');
% print('Plots/EPSC/All_Results_Median', '-depsc');
% print('Plots/SVG/All_Results_Median', '-dsvg');

%% ----------------------- 2D Histogram of all responses MEAN -------------------------------
fprintf('2D Histogram of all responses MEAN\n')
figure('Position',[0 0 500 300])
All = [a(1:5280).mean_MOS_norm];
[~,I] = sort(All);
max_len = 1;
for n = 1:length(I)
    len = length(a(n).mean_MOS_norm);
    if len>max_len
        max_len = len;
    end
end
All_results = zeros(length(I),max_len);
x = [];
y = [];
for n = 1:length(All)
    x = [x ; n*ones(length(a(I(n)).MOS_norm),1)];
    y = [y a(I(n)).MOS_norm];
end

h = histogram2(x,y',[100 100],'FaceColor','flat');
h.ShowEmptyBins = 'Off';
h.DisplayStyle = 'tile';
h.EdgeAlpha = 0;

ax = gca;
ax.GridColor = [0.4 0.4 0.4];
ax.GridLineStyle = '--';
ax.GridAlpha = 0.5;
ax.XGrid = 'off';
ax.YGrid = 'on';
ax.Layer = 'top';
view(2)
% colormap(flipud(gray));
c = colorbar;
c.Label.String = 'Count';
% c.Label.String = 'Probability';
% title_text = sprintf('All Opinion Scores Ordered by Ascending MeanOS (%d Ratings)',size([a.MOS],2));
% title_text = sprintf('All Opinion Scores Ordered by Ascending MeanOS');
% title(title_text)

yticks(1:5);
yticklabels({'Bad', 'Poor', 'Fair', 'Good', 'Excellent'})
xticks([]);
xlabel('File')
ylabel('Opinion Rating')
axis('tight')
hold on
p = plot3(1:length(I), [a(I).mean_MOS_norm],(1+max(max(h.BinCounts)))*ones(1,length(I)),'r--');
p.LineWidth = 2;
hold off


% hold on
% for n=1:5280
%     plot3(n, median(a(I(n)).MOS_norm), max(max(h.BinCounts)),'c.');
% %     p.LineWidth = 2;
% end
% hold off

set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');

print('Plots/TIFF/All_Results_Hist_Mean', '-dtiff');
print('Plots/EPSC/All_Results_Hist_Mean', '-depsc');
print('Plots/PNG/All_Results_Hist_Mean', '-dpng');

%% ----------------------- 2D Histogram of all responses with objective files MEAN -------------------------------

% figure('Position',[0 0 800 500])
% All = [a.mean_MOS_norm];
% [~,I] = sort(All);
% max_len = 1;
% for n = 1:length(I)
%     len = length(a(n).mean_MOS_norm);
%     if len>max_len
%         max_len = len;
%     end
% end
% All_results = zeros(length(I),max_len);
% x = [];
% y = [];
% for n = 1:length(All)
%     x = [x ; n*ones(length(a(I(n)).MOS_norm),1)];
%     y = [y a(I(n)).MOS_norm];
% end
%
% h = histogram2(x,y',[100 100],'FaceColor','flat');
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
% % colormap(flipud(gray));
% c = colorbar;
% c.Label.String = 'Count';
% % c.Label.String = 'Probability';
% % title_text = sprintf('All Opinion Scores Ordered by Ascending MeanOS (%d Ratings)',size([a.MOS],2));
% % title_text = sprintf('All Opinion Scores Ordered by Ascending MeanOS');
% % title(title_text)
%
% yticks(1:5);
% yticklabels({'Bad', 'Poor', 'Fair', 'Good', 'Excellent'})
%
% xticks([]);
% % xlabel('File')
% ylabel('Opinion Rating')
% axis('tight')
% hold on
% p = plot3(1:length(I), [a(I).mean_MOS_norm],(1+max(max(h.BinCounts)))*ones(1,length(I)),'r');
% p.LineWidth = 2;
% hold off
%
%
% % hold on
% % for n=1:5280
% %     plot3(n, median(a(I(n)).MOS_norm), max(max(h.BinCounts)),'c.');
% % %     p.LineWidth = 2;
% % end
% % hold off
%
% set(gca,...
%     'FontSize', 12, ...
%     'FontName', 'Times');
%
% print('Plots/TIFF/All_Results_Hist_Mean_Obj', '-dtiff');
% print('Plots/EPSC/All_Results_Hist_Mean_Obj', '-depsc');
% print('Plots/PNG/All_Results_Hist_Mean_Obj', '-dpng');

%% ----------------------- 2D Histogram of all responses MEDIAN-------------------------------
fprintf('2D Histogram of all responses MEDIAN\n')
figure('Position',[0 0 500 300])
All = [a(1:5280).median_MOS_norm];
[~,I] = sort(All);
max_len = 1;
for n = 1:length(I)
    len = length(a(n).MOS_norm);
    if len>max_len
        max_len = len;
    end
end
All_results = zeros(length(I),max_len);
x = [];
y = [];
for n = 1:length(All)
    x = [x ; n*ones(length(a(I(n)).MOS_norm),1)];
    y = [y a(I(n)).MOS_norm];
end

h = histogram2(x,y',[100 100],'FaceColor','flat');
h.ShowEmptyBins = 'Off';
h.DisplayStyle = 'tile';
h.EdgeAlpha = 0;

ax = gca;
ax.GridColor = [0.4 0.4 0.4];
ax.GridLineStyle = '--';
ax.GridAlpha = 0.5;
ax.XGrid = 'off';
ax.YGrid = 'on';
ax.Layer = 'top';
view(2)
% colormap(flipud(gray));
c = colorbar;
c.Label.String = 'Count';
% c.Label.String = 'Probability';
% title_text = sprintf('All Opinion Scores Ordered by Ascending Median (%d Ratings)',size([a.MOS],2));
% title_text = sprintf('All Opinion Scores Ordered by Ascending MedianOS');
% title(title_text)

yticks(1:5);
yticklabels({'Bad', 'Poor', 'Fair', 'Good', 'Excellent'})

xticks([]);
xlabel('File')
ylabel('Opinion Rating')
axis('tight')
hold on
p = plot3(1:length(I), [a(I).median_MOS_norm],(1+max(max(h.BinCounts)))*ones(1,length(I)),'r--');
p.LineWidth = 1.5;
hold off


% hold on
% for n=1:5280
%     plot3(n, median(a(I(n)).MOS_norm), max(max(h.BinCounts)),'c.');
% %     p.LineWidth = 2;
% end
% hold off

set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');

print('Plots/TIFF/All_Results_Hist_Median', '-dtiff');
print('Plots/EPSC/All_Results_Hist_Median', '-depsc');
print('Plots/PNG/All_Results_Hist_Median', '-dpng');

%% ----------------------- 2D Histogram of all responses with objective files MEDIAN-------------------------------

% figure('Position',[0 0 800 500])
% All = [a.median_MOS_norm];
% [~,I] = sort(All);
% max_len = 1;
% for n = 1:length(I)
%     len = length(a(n).MOS_norm);
%     if len>max_len
%         max_len = len;
%     end
% end
% All_results = zeros(length(I),max_len);
% x = [];
% y = [];
% for n = 1:length(All)
%     x = [x ; n*ones(length(a(I(n)).MOS_norm),1)];
%     y = [y a(I(n)).MOS_norm];
% end
%
% h = histogram2(x,y',[100 100],'FaceColor','flat');
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
% % colormap(flipud(gray));
% c = colorbar;
% c.Label.String = 'Count';
% % c.Label.String = 'Probability';
% % title_text = sprintf('All Opinion Scores Ordered by Ascending Median (%d Ratings)',size([a.MOS],2));
% % title_text = sprintf('All Opinion Scores Ordered by Ascending MedianOS');
% % title(title_text)
%
% yticks(1:5);
% yticklabels({'Bad', 'Poor', 'Fair', 'Good', 'Excellent'})
%
% xticks([]);
% % xlabel('File')
% ylabel('Opinion Rating')
% axis('tight')
% hold on
% p = plot3(1:length(I), [a(I).median_MOS_norm],(1+max(max(h.BinCounts)))*ones(1,length(I)),'r');
% p.LineWidth = 2;
% hold off
%
%
% % hold on
% % for n=1:5280
% %     plot3(n, median(a(I(n)).MOS_norm), max(max(h.BinCounts)),'c.');
% % %     p.LineWidth = 2;
% % end
% % hold off
%
% set(gca,...
%     'FontSize', 12, ...
%     'FontName', 'Times');
%
% print('Plots/TIFF/All_Results_Hist_Median_Obj', '-dtiff');
% print('Plots/EPSC/All_Results_Hist_Median_Obj', '-depsc');
% print('Plots/PNG/All_Results_Hist_Median_Obj', '-dpng');

%% ----------------------- 2D Histogram of all responses pre-normalisation Sorted by mean -------------------------------
fprintf('2D Histogram of all responses pre-normalisation MEAN\n')
figure('Position',[0 0 500 300])
All = [a.mean_MOS];
[~,I] = sort(All);
max_len = 1;
for n = 1:length(I)
    len = length(a(n).MOS);
    if len>max_len
        max_len = len;
    end
end
All_results = zeros(length(I),max_len);
x = [];
y = [];
for n = 1:length(All)
    x = [x ; n*ones(length(a(I(n)).MOS),1)];
    y = [y a(I(n)).MOS];
end

h = histogram2(x,y',[100 100],'FaceColor','flat');
h.ShowEmptyBins = 'Off';
h.DisplayStyle = 'tile';
h.EdgeAlpha = 0;

ax = gca;
ax.GridColor = [0.4 0.4 0.4];
ax.GridLineStyle = '--';
ax.GridAlpha = 0.5;
ax.XGrid = 'off';
ax.YGrid = 'on';
ax.Layer = 'top';

view(2)
% colormap(flipud(gray));
c = colorbar;
c.Label.String = 'Count';
% title_text = sprintf('All Opinion Scores Pre-Normalisation \n Ordered by Ascending Mean (%d Ratings)',size([a.MOS],2));
% title(title_text)

yticks(1:5);
yticklabels({'Bad', 'Poor', 'Fair', 'Good', 'Excellent'})

xticks([]);
xlabel('File')
ylabel('Opinion Rating')
axis('tight')
hold on
p = plot3(1:length(a), [a(I).mean_MOS],max(max(h.BinCounts))*ones(1,length(a)),'r--');
p.LineWidth = 2;
hold off

set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');

print('Plots/TIFF/All_Results_Hist_mean_pre_Norm', '-dtiff');
print('Plots/EPSC/All_Results_Hist_mean_pre_Norm', '-depsc');
print('Plots/PNG/All_Results_Hist_mean_pre_Norm', '-dpng');


%% ----------------------- 2D Histogram of all responses pre-normalisation Sorted by median -------------------------------
fprintf('2D Histogram of all responses pre-normalisation MEDIAN\n')
figure('Position',[0 0 500 300])
for n = 1:length(a)
    a(n).median_MOS = median(a(n).MOS);
end
All = [a.median_MOS];
[~,I] = sort(All);
max_len = 1;
for n = 1:length(I)
    len = length(a(n).MOS);
    if len>max_len
        max_len = len;
    end
end
All_results = zeros(length(I),max_len);
x = [];
y = [];
for n = 1:length(All)
    x = [x ; n*ones(length(a(I(n)).MOS),1)];
    y = [y a(I(n)).MOS];
end

h = histogram2(x,y',[100 100],'FaceColor','flat');
h.ShowEmptyBins = 'Off';
h.DisplayStyle = 'tile';
h.EdgeAlpha = 0;

ax = gca;
ax.GridColor = [0.4 0.4 0.4];
ax.GridLineStyle = '--';
ax.GridAlpha = 0.5;
ax.XGrid = 'off';
ax.YGrid = 'on';
ax.Layer = 'top';

view(2)
% colormap(flipud(gray));
c = colorbar;
c.Label.String = 'Count';
% title_text = sprintf('All Opinion Scores Pre-Normalisation \n Ordered by Ascending Mean (%d Ratings)',size([a.MOS],2));
% title(title_text)

yticks(1:5);
yticklabels({'Bad', 'Poor', 'Fair', 'Good', 'Excellent'})

xticks([]);
% xlabel('File')
ylabel('Opinion Rating')
axis('tight')
hold on
p = plot3(1:length(a), [a(I).median_MOS],max(max(h.BinCounts))*ones(1,length(a)),'r--');
p.LineWidth = 2;
hold off

set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');

print('Plots/TIFF/All_Results_Hist_median_pre_Norm', '-dtiff');
print('Plots/EPSC/All_Results_Hist_median_pre_Norm', '-depsc');
print('Plots/PNG/All_Results_Hist_median_pre_Norm', '-dpng');


%% -----------------------Mean Absolute Difference Analysis------------------------------


% %Plot a Histogram of the Mean average distance from the mean for each set
% %response
% fprintf('Plotting Histogram of MAD\n')
% figure('Position',[1680-500 200 500 250])
% h = histogram([u.mean_abs_diff_norm_mean],'BinWidth',0.05);
% h.EdgeColor = 'k';
% h.FaceColor = [ 1,1,1];
% % title('Mean Absolute Difference Per Session')
% xlabel('$\bar{X}_s$','Interpreter','latex')
% ylabel('Count')
%
% set(gca,...
%     'FontSize', 12, ...
%     'FontName', 'Times');
%
% print('Plots/TIFF/MAD_mean', '-dtiff');
% print('Plots/EPSC/MAD_mean', '-depsc');
% print('Plots/PNG/MAD_mean', '-dpng');
%
% fprintf('Plotting Histogram of MAD\n')
% figure('Position',[1680-500 200 500 250])
% h = histogram([u.mean_abs_diff_norm_median]);
% h.EdgeColor = 'k';
% h.FaceColor = [ 1,1,1];
% title('Mean Absolute Difference (Rating - Median) Per Set')
% xlabel('Mean Absolute Difference')
% ylabel('Count')
%
% set(gca,...
%     'FontSize', 12, ...
%     'FontName', 'Times');
%
% print('Plots/TIFF/MAD_median', '-dtiff');
% print('Plots/EPSC/MAD_median', '-depsc');
% print('Plots/PNG/MAD_median', '-dpng');

% %Plot the mean absolute difference per user.
% %Modify the above code so that if someone does multiple sets, it groups the
% %Mean Absolute Differences
% figure
% hold on
% for n = 1:length(u)
%     plot(n*ones(size(u(n).mean_abs_diff_mean)),u(n).mean_abs_diff_mean,'.')
% end
% hold off
% title('Mean Absolute Difference (Rating-MOS)')
%
% figure
% MAD = [u.mean_abs_diff_mean];
% [A,MAD_I] = sort(MAD);
% plot(A)
% title('Mean Absolute Difference (Rating-Mean) Ascending Order')
% ylabel('MAD')




%% ------------Exploring standard deviation of files with more than 5 responses-------
% reduced_set = [];
% cutoff = 5;
% for n = 1:length(a)
%     if(a(n).num_responses>cutoff)
%         reduced_set = [reduced_set a(n).std_MOS_norm];
%     end
% end
% figure
% hist(reduced_set, round(length(reduced_set)/10))
% title_text = sprintf('Reduced set of Standard Deviation for files with >%d responses',cutoff);
% title(title_text)
% xlabel('Standard Deviation')

% figure('Position',[1680-500 200 500 250])
% plot([a.num_responses],[a.std_MOS_norm],'k.')
% plot_title = sprintf('Number of Responses vs. Std(Opinion Scores)');
% title(plot_title)
% xlabel('Number of Responses')
% ylabel('Standard Deviation')
% axis([0, 18, 0, 1.1*max([a.std_MOS_norm])])
%
% set(gca,...
%     'FontSize', 12, ...
%     'FontName', 'Times');
%
% print('Plots/TIFF/STD_vs_Responses', '-dtiff');
% print('Plots/EPSC/STD_vs_Responses', '-depsc');
% print('Plots/PNG/STD_vs_Responses', '-dpng');

fprintf('STD vs number of responses\n')
figure('Position',[1680-500 200 500 250])
h = histogram2([a.num_responses],[a.std_MOS_norm],'BinWidth',[1 0.05],'FaceColor','flat');
h.DisplayStyle = 'tile';
h.EdgeAlpha = 0;
view(2)
% colormap(gray);
c = colorbar;
c.Label.String = 'Count';
grid off
axis([min([a.num_responses]), max([a.num_responses]), 0, 1.1*max([a.std_MOS_norm])])
% title('Number of Responses vs Std(Opinion Scores)')
xlabel('Number of Ratings')
ylabel('$\sigma_s$','interpreter','latex','Rotation',0)

set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');

print('Plots/TIFF/STD_vs_Responses_Hist2', '-dtiff');
print('Plots/EPSC/STD_vs_Responses_Hist2', '-depsc');
print('Plots/PNG/STD_vs_Responses_Hist2', '-dpng');



%% --------------------------------ANOVA Testing--------------------------------
%Compute Anova testing on Each method for each time scale
% figure('Position',[1680-500 200 500 250])
% fprintf('Plotting Box plots for each time scale\n')
%
% for k = 1:length(TSM)
%     var_name = sprintf('TSM%d',k);
%     boxplot(TSM_res.(var_name),TSM_methods,'colors','k','notch','on');
% %     title_text = sprintf('Time-Scale Ratio of %d%%',TSM(k));
% %     title(title_text);
%     xlabel('Method')
%     ylabel('MOS')
%     axis([0.5 6.5 0.9 5.1])
%     file_name_text = sprintf('Boxplot_%d%%_TSM',TSM(k));
%
%     set(gca,...
%     'FontSize', 12, ...
%     'FontName', 'Times');
%
%     print(['Plots/TIFF/' file_name_text], '-dtiff');
%     print(['Plots/EPSC/' file_name_text], '-depsc');
%     print(['Plots/PNG/' file_name_text], '-dpng');
%
% end



% %Work on ANOVA2 using method and TSM as parameters



% [p,tbl,stats] = anova2(anova2_data,length(TSM));
% figure('Position',[1680-500 200 500 250])
% multcompare(stats)
% % title('ANOVA2 Comparison of Method and Time-Scale')
% yticklabels(fliplr(TSM_methods))
% title('')
% xlabel('MOS')
%
% set(gca,...
%     'FontSize', 12, ...
%     'FontName', 'Times');
%
% pause(1)
% print('Plots/TIFF/Anova2_method_time_scale_zoom', '-dtiff');
% print('Plots/EPSC/Anova2_method_time_scale_zoom', '-depsc');
% print('Plots/PNG/Anova2_method_time_scale_zoom', '-dpng');
% axis([0.9 5.1 0 7])
% print('Plots/TIFF/Anova2_method_time_scale', '-dtiff');
% print('Plots/EPSC/Anova2_method_time_scale', '-depsc');
% print('Plots/PNG/Anova2_method_time_scale', '-dpng');


%% ----------------- Overall Boxplots ----------------------
fprintf('Overall Boxplots\n')
figure('Position',[1680-500 200 500 250])
boxplot(anova2_data,TSM_methods, 'colors', 'k', 'notch', 'on')
xlabel('Method')
ylabel('MOS')

set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');

print('Plots/TIFF/Boxplot_Overall', '-dtiff');
print('Plots/EPSC/Boxplot_Overall', '-depsc');
print('Plots/PNG/Boxplot_Overall', '-dpng');

%% ------------------ Box plots per method -------------------
fprintf('Boxplots for each TSM method\n')
fprintf('Plotting Box plots for each method\n')
figure('Position',[1680-500 200 500 250])
box_label = [38, 44, 53, 65, 78, 82, 99, 138, 166, 192];
boxplot(PV,box_label,'colors','k','notch','on');
% title('Phase Vocoder (PV)')
xlabel('Time-Scale Ratio (% (Floored))')
ylabel('MOS');
axis([0.5 10.5 0.9 5.1])
set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');
print('Plots/TIFF/Boxplot_PV', '-dtiff');
print('Plots/EPSC/Boxplot_PV', '-depsc');
print('Plots/PNG/Boxplot_PV', '-dpng');


boxplot(IPL,box_label,'colors','k','notch','on');
% title('Identity Phase Locked Phase Vocoder (IPL)')
xlabel('Time-Scale Ratio (% (Floored))')
ylabel('MOS');
axis([0.5 10.5 0.9 5.1])
set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');
print('Plots/TIFF/Boxplot_IPL', '-dtiff');
print('Plots/EPSC/Boxplot_IPL', '-depsc');
print('Plots/PNG/Boxplot_IPL', '-dpng');


boxplot(HPTSM,box_label,'colors','k','notch','on');
% title('Harmonic Percussive Time Scale Modification (HPTSM)')
xlabel('Time-Scale Ratio (% (Floored))')
ylabel('MOS');
axis([0.5 10.5 0.9 5.1])
set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');
print('Plots/TIFF/Boxplot_HPTSM', '-dtiff');
print('Plots/EPSC/Boxplot_HPTSM', '-depsc');
print('Plots/PNG/Boxplot_HPTSM', '-dpng');


boxplot(WSOLA,box_label,'colors','k','notch','on');
% title('Waveform Similarity Overlap Add (WSOLA)')
xlabel('Time-Scale Ratio (% (Floored))')
ylabel('MOS');
axis([0.5 10.5 0.9 5.1])
set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');
print('Plots/TIFF/Boxplot_WSOLA', '-dtiff');
print('Plots/EPSC/Boxplot_WSOLA', '-depsc');
print('Plots/PNG/Boxplot_WSOLA', '-dpng');

figure('Position',[1920-500 200 500 300])
boxplot(FESOLA,box_label,'colors','k','notch','on');
% title('Fuzzy Epoch Synchronous Overlap Add (FESOLA)')
xlabel('Time-Scale Ratio (% (Floored))')
ylabel('MOS');
axis([0.5 10.5 0.9 5.1])
set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');
print('Plots/TIFF/Boxplot_FESOLA', '-dtiff');
print('Plots/EPSC/Boxplot_FESOLA', '-depsc');
print('Plots/PNG/Boxplot_FESOLA', '-dpng');


boxplot(uTVS,box_label,'colors','k','notch','on');
% title('Mel-Scale Filterbank Time Scale Modification (uTVS)')
xlabel('Time-Scale Ratio (% (Floored))')
ylabel('MOS');
axis([0.5 10.5 0.9 5.1])
set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');
print('Plots/TIFF/Boxplot_uTVS', '-dtiff');
print('Plots/EPSC/Boxplot_uTVS', '-depsc');
print('Plots/PNG/Boxplot_uTVS', '-dpng');



%Create latex table with methods vs time-scale values
fprintf('Create latex table with methods and time-scales\n')
input.data = zeros(10,6);
input.data(:,1) = mean(PV);
input.data(:,2) = mean(IPL);
input.data(:,3) = mean(WSOLA);
input.data(:,4) = mean(FESOLA);
input.data(:,5) = mean(HPTSM);
input.data(:,6) = mean(uTVS);
fid = fopen('log_Anon.txt','a');
fprintf(fid,'Methods and Time Scales\n');
latex_output = JASAlatexTable(input,fid);
fclose(fid);
%% --------------Plot based on file category-----------------------


%Complex, Solo, Music, Voice are
%number_of_files x number_of_time_scales x method in size

%Overall plotting

% figure('Position',[1920-500 200 500 300])
% boxplot(Complex.type_mean,TSM_methods,'colors','k')
% % title('Means across all Time-Scales for Complex Audio Files')
% xlabel('TSM Algorithm')
% ylabel('MOS')
% axis([0.5 6.5 0.9 5.1])
% set(gca,...
%     'FontSize', 12, ...
%     'FontName', 'Times');
% print('Plots/TIFF/Complex_mean_method', '-dtiff');
% print('Plots/EPSC/Complex_mean_method', '-depsc');
% print('Plots/PNG/Complex_mean_method', '-dpng');
%
% boxplot(Music.type_mean,TSM_methods,'colors','k','notch','on')
% % title('Means across all Time-Scales for Music Audio Files')
% xlabel('TSM Algorithm')
% ylabel('MOS')
% axis([0.5 6.5 0.9 5.1])
% set(gca,...
%     'FontSize', 12, ...
%     'FontName', 'Times');
% print('Plots/TIFF/Music_mean_method', '-dtiff');
% print('Plots/EPSC/Music_mean_method', '-depsc');
% print('Plots/PNG/Music_mean_method', '-dpng');
%
% boxplot(Solo.type_mean,TSM_methods,'colors','k','notch','on')
% % title('Means across all Time-Scales for Solo Audio Files')
% xlabel('TSM Algorithm')
% ylabel('MOS')
% axis([0.5 6.5 0.9 5.1])
% set(gca,...
%     'FontSize', 12, ...
%     'FontName', 'Times');
% print('Plots/TIFF/Solo_mean_method', '-dtiff');
% print('Plots/EPSC/Solo_mean_method', '-depsc');
% print('Plots/PNG/Solo_mean_method', '-dpng');
%
% boxplot(Voice.type_mean,TSM_methods,'colors','k','notch','on')
% % title('Means across all Time-Scales for Voice Audio Files')
% xlabel('TSM Algorithm')
% ylabel('MOS')
% axis([0.5 6.5 0.9 5.1])
% set(gca,...
%     'FontSize', 12, ...
%     'FontName', 'Times');
% print('Plots/TIFF/Voice_mean_method', '-dtiff');
% print('Plots/EPSC/Voice_mean_method', '-depsc');
% print('Plots/PNG/Voice_mean_method', '-dpng');
%
%
% figure('Position',[1920-500 200 500 300])
% for m = 1:length(TSM_methods)
%     boxplot(Complex.TSM_method(:,:,m),TSM,'colors','k')
% %     title_text = sprintf('%s for Complex Files', TSM_methods{m});
% %     title(title_text);
%     xlabel('Time Scale Ratio (% (Floored))')
%     ylabel('MOS')
%     axis([0.5 10.5 0.9 5.1])
%     file_name_text = sprintf('Boxplot_%s_for_Complex_files',TSM_methods{m});
%     set(gca,...
%     'FontSize', 12, ...
%     'FontName', 'Times');
%     print(['Plots/TIFF/' file_name_text], '-dtiff');
%     print(['Plots/EPSC/' file_name_text], '-depsc');
%     print(['Plots/PNG/' file_name_text], '-dpng');
% end
%
%
% for m = 1:length(TSM_methods)
%     boxplot(Music.TSM_method(:,:,m),TSM,'notch','on','colors','k')
% %     title_text = sprintf('%s for Music Files', TSM_methods{m});
% %     title(title_text);
%     xlabel('Time Scale Ratio (% (Floored))')
%     ylabel('MOS')
%     axis([0.5 10.5 0.9 5.1])
%     file_name_text = sprintf('Boxplot_%s_for_Music_files',TSM_methods{m});
%     set(gca,...
%     'FontSize', 12, ...
%     'FontName', 'Times');
%     print(['Plots/TIFF/' file_name_text], '-dtiff');
%     print(['Plots/EPSC/' file_name_text], '-depsc');
%     print(['Plots/PNG/' file_name_text], '-dpng');
% end
%
%
% for m = 1:length(TSM_methods)
%     boxplot(Solo.TSM_method(:,:,m),TSM,'notch','on','colors','k')
% %     title_text = sprintf('%s for Solo Files', TSM_methods{m});
% %     title(title_text);
%     xlabel('Time Scale Ratio (% (Floored))')
%     ylabel('MOS')
%     axis([0.5 10.5 0.9 5.1])
%     file_name_text = sprintf('Boxplot_%s_for_Solo_files',TSM_methods{m});
%     set(gca,...
%     'FontSize', 12, ...
%     'FontName', 'Times');
%     print(['Plots/TIFF/' file_name_text], '-dtiff');
%     print(['Plots/EPSC/' file_name_text], '-depsc');
%     print(['Plots/PNG/' file_name_text], '-dpng');
% end
%
%
% for m = 1:length(TSM_methods)
%     boxplot(Voice.TSM_method(:,:,m),TSM,'notch','on','colors','k')
% %     title_text = sprintf('%s for Voice Files', TSM_methods{m});
% %     title(title_text);
%     xlabel('Time Scale Ratio (% (Floored))')
%     ylabel('MOS')
%     axis([0.5 10.5 0.9 5.1])
%     file_name_text = sprintf('Boxplot_%s_for_Voice_files',TSM_methods{m});
%     set(gca,...
%     'FontSize', 12, ...
%     'FontName', 'Times');
%     print(['Plots/TIFF/' file_name_text], '-dtiff');
%     print(['Plots/EPSC/' file_name_text], '-depsc');
%     print(['Plots/PNG/' file_name_text], '-dpng');
% end

grey_lines= {'k-o', 'k-+', 'k-*', 'k.-', 'k-x', 'k-s'};
%Plot the averages of each method for the file category
% Complex.overall_mean = mean(Complex.TSM_method,1);
% figure('Position',[1680-500 200 500 400])
% hold on
% for n = 1:size(Complex.overall_mean,3)
%     plot(TSM,Complex.overall_mean(:,:,n),grey_lines{n})
% end
% % title('Mean MOS for Complex files')
% xlabel('Time-Scale Ratio (%)')
% ylabel('MOS')
% legend(TSM_methods,'Location','SouthEast');
% axis([0 200 0.9 5.1])
% hold off
% set(gca,...
%     'FontSize', 12, ...
%     'FontName', 'Times');
% print('Plots/TIFF/Complex_means', '-dtiff');
% print('Plots/EPSC/Complex_means', '-depsc');
% print('Plots/PNG/Complex_means', '-dpng');
fprintf('File class line graphs\n')
Music.overall_mean = mean(Music.TSM_method,1);
figure('Position',[1680-500 200 500 350])
hold on
for n = 1:size(Music.overall_mean,3)
    plot(TSM/100,Music.overall_mean(:,:,n),grey_lines{n})
end
% title('Mean MOS for Music files')
xlabel('Time-Scale Ratio (\beta)')
ylabel('MOS')
legend(TSM_methods,'Location','SouthEast');
axis([0.2 2 0.9 5.1])
hold off
set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');
print('Plots/TIFF/Music_means', '-dtiff');
print('Plots/EPSC/Music_means', '-depsc');
print('Plots/PNG/Music_means', '-dpng');

Solo.overall_mean = mean(Solo.TSM_method,1);
figure('Position',[1680-500 200 500 350])
hold on
for n = 1:size(Solo.overall_mean,3)
    plot(TSM/100,Solo.overall_mean(:,:,n),grey_lines{n})
end
% title('Mean MOS for Solo files')
xlabel('Time-Scale Ratio (\beta)')
ylabel('MOS')
legend(TSM_methods,'Location','SouthEast');
axis([0.2 2 0.9 5.1])
hold off
set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');
print('Plots/TIFF/Solo_means', '-dtiff');
print('Plots/EPSC/Solo_means', '-depsc');
print('Plots/PNG/Solo_means', '-dpng');

Voice.overall_mean = mean(Voice.TSM_method,1);
figure('Position',[1680-500 200 500 350])
hold on
for n = 1:size(Voice.overall_mean,3)
    plot(TSM/100,Voice.overall_mean(:,:,n),grey_lines{n})
end
% title('Mean MOS for Voice files')
xlabel('Time-Scale Ratio (\beta)')
ylabel('MOS')
legend(TSM_methods,'Location','SouthEast');
axis([0.2 2 0.9 5.1])
hold off
set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');
print('Plots/TIFF/Voice_means', '-dtiff');
print('Plots/EPSC/Voice_means', '-depsc');
print('Plots/PNG/Voice_means', '-dpng');


% overall_mean = mean([Complex.overall_mean ; Music.overall_mean ; Solo.overall_mean ; Voice.overall_mean],1);
overall_mean = mean([Music.overall_mean ; Solo.overall_mean ; Voice.overall_mean],1);

figure('Position',[1680-500 200 500 350])
hold on
for n = 1:size(overall_mean,3)
    plot(TSM/100,overall_mean(:,:,n),grey_lines{n})
end
% title('Mean MeanOS for All Files')
xlabel('Time-Scale Ratio (\beta)')
ylabel('MOS')
legend(TSM_methods,'Location','SouthEast');
axis([0.2 2 0.9 5.1])
hold off
set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');
print('Plots/PDF/Overall_means', '-dpdf');
print('Plots/EPSC/Overall_means', '-depsc');
print('Plots/PNG/Overall_means', '-dpng');



%% -------------------- Plot Age vs MAD -------------------------
%
% figure('Position',[1680-500 200 500 250])
% hold on
% for f = 1:length(u)
%     plot(u(f).age*ones(size(u(f).MMAD_norm_mean_new)),u(f).MMAD_norm_mean_new,'k.')
% end
% hold off
% % title('Age vs Mean Absolute Difference to Mean')
% xlabel('Age')
% ylabel('$\bar{X}_s$','interpreter','latex','Rotation',0)
% axis([min([u.age]) max([u.age]) 0 1.2])
% set(gca,...
%     'FontSize', 12, ...
%     'FontName', 'Times');
% print('Plots/PDF/Age_vs_MAD_mean', '-dpdf');
% print('Plots/EPSC/Age_vs_MAD_mean', '-depsc');
% print('Plots/PNG/Age_vs_MAD_mean', '-dpng');
fprintf('Age Related Plotting\n')
figure('Position',[1680-500 200 500 250])
hold on
for f = 1:length(u)
    plot(u(f).age*ones(size(u(f).RMSE)),u(f).RMSE,'k.')
end
hold off
% title('Age vs Mean Absolute Difference to Mean')
xlabel('Age')
ylabel('$\mathcal{L}$','interpreter','latex','Rotation',0)
axis([0.9*min([u.age]) 1.1*max([u.age]) 0.9*min([u.RMSE]) 1.1*max([u.RMSE])])
set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');
print('Plots/PDF/Age_vs_RMSE', '-dpdf');
print('Plots/EPSC/Age_vs_RMSE', '-depsc');
print('Plots/PNG/Age_vs_RMSE', '-dpng');

figure('Position',[1680-500 200 500 250])
hold on
for f = 1:length(u)
    plot(u(f).age*ones(size(u(f).RMSE_norm)),u(f).RMSE_norm,'k.')
end
hold off
% title('Age vs Mean Absolute Difference to Mean')
xlabel('Age')
ylabel('$\mathcal{L}$','interpreter','latex','Rotation',0)
axis([0.9*min([u.age]) 1.1*max([u.age]) 0.9*min([u.RMSE_norm]) 1.1*max([u.RMSE_norm])])
set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');
print('Plots/PDF/Age_vs_RMSE_norm', '-dpdf');
print('Plots/EPSC/Age_vs_RMSE_norm', '-depsc');
print('Plots/PNG/Age_vs_RMSE_norm', '-dpng');

% figure('Position',[1680-500 200 500 250])
% hold on
% for f = 1:length(u)
%     plot(u(f).age*ones(size(u(f).MMAD_norm_median_new)),u(f).MMAD_norm_median_new,'k.')
% end
% hold off
% title('Age vs Mean Absolute Difference to Median')
% xlabel('Age')
% ylabel('Mean Absolute Difference')
% axis([min([u.age]) max([u.age]) 0 1.2])
% set(gca,...
%     'FontSize', 12, ...
%     'FontName', 'Times');
% print('Plots/PDF/Age_vs_MAD_median', '-dpdf');
% print('Plots/EPSC/Age_vs_MAD_median', '-depsc');
% print('Plots/PNG/Age_vs_MAD_median', '-dpng');


%% --------------------- Plot the MAD for the number of files in session -----------

% figure('Position',[1680-500 200 500 250])
% plot([u.num_files],[u.mean_abs_diff_mean],'.')
% hold on
% % plot([u.num_files],[u.mean_abs_diff_norm],'.')
% plot([u.num_files],[u.mean_abs_diff_norm_mean],'.')
% hold off
% title('MAD Mean For Number Of Rated Files')
% xlabel('Ratings Per Session')
% ylabel('Mean Absolute Difference')
% legend('Raw results','Normalised Results'); %,'Normalised Results to Non-normalised Mean'
% set(gca,...
%     'FontSize', 12, ...
%     'FontName', 'Times');
% print('Plots/PDF/Ratings_vs_MAD_mean', '-dpdf');
% print('Plots/EPSC/Ratings_vs_MAD_mean', '-depsc');
% print('Plots/PNG/Ratings_vs_MAD_mean', '-dpng');
%
% figure('Position',[1680-500 200 500 250])
% plot([u.num_files],[u.mean_abs_diff_median],'.')
% hold on
% % plot([u.num_files],[u.mean_abs_diff_norm],'.')
% plot([u.num_files],[u.mean_abs_diff_norm_median],'.')
% hold off
% title('MAD Median For Number Of Rated Files')
% xlabel('Ratings Per Session')
% ylabel('Mean Absolute Difference')
% legend('Raw results','Normalised Results'); %,'Normalised Results to Non-normalised Mean'
% set(gca,...
%     'FontSize', 12, ...
%     'FontName', 'Times');
% print('Plots/PDF/Ratings_vs_MAD_median', '-dpdf');
% print('Plots/EPSC/Ratings_vs_MAD_median', '-depsc');
% print('Plots/PNG/Ratings_vs_MAD_median', '-dpng');


%% --------------------- Plot the new MAD compared to old MAD ---------------------
%MEAN

% figure('Position',[1680-500 200 500 250])
% for n = 1:length(u)
%     u(n).MAD_diff = u(n).mean_abs_diff_mean - u(n).mean_abs_diff_norm_mean;
% end
% plot(abs([u.MAD_diff]),'k.')
% title('Comparison of MAD values (Old-New Mean)')
%
% figure('Position',[1680-500 200 500 250])
% plot([u.num_files],abs([u.MAD_diff]),'k.')
% title('MAD Mean difference Pre and Post Normalisation')
% xlabel('Number of Files in Session')
% ylabel('MAD difference')
% set(gca,...
%     'FontSize', 12, ...
%     'FontName', 'Times');
% print('Plots/PDF/MAD_difference_pre_post_norm_mean', '-dpdf');
% print('Plots/EPSC/MAD_difference_pre_post_norm_mean', '-depsc');
% print('Plots/PNG/MAD_difference_pre_post_norm_mean', '-dpng');
%
% diff_threshold = 0.5;
% for n = 1:length(u)
%     for k = 1:length(u(n).MAD_diff)
%         if u(n).MAD_diff(k) > diff_threshold
%             fprintf('%s with mean MAD difference of %.3f with %d files. %s\n',u(n).name, u(n).MAD_diff(k), u(n).num_files(k), u(n).key);
%         end
%     end
% end
%
% %MEDIAN
%
% figure('Position',[1680-500 200 500 250])
% for n = 1:length(u)
%     u(n).MAD_diff_median = u(n).mean_abs_diff_median - u(n).mean_abs_diff_norm_median;
% end
% plot(abs([u.MAD_diff_median]),'k.')
% title('Comparison of MAD values (Old-New Median)')
%
% figure('Position',[1680-500 200 500 250])
% plot([u.num_files],abs([u.MAD_diff_median]),'k.')
% title('MAD Median difference Pre and Post Normalisation')
% xlabel('Number of Files in Session')
% ylabel('MAD difference')
% set(gca,...
%     'FontSize', 12, ...
%     'FontName', 'Times');
% print('Plots/PDF/MAD_difference_pre_post_norm_median', '-dpdf');
% print('Plots/EPSC/MAD_difference_pre_post_norm_median', '-depsc');
% print('Plots/PNG/MAD_difference_pre_post_norm_median', '-dpng');
%
% diff_threshold = 0.5;
% for n = 1:length(u)
%     for k = 1:length(u(n).MAD_diff_median)
%         if u(n).MAD_diff_median(k) > diff_threshold
%             fprintf('%s with median MAD difference of %.3f with %d files\n',u(n).name, u(n).MAD_diff_median(k), u(n).num_files(k));
%         end
%     end
% end



%% Histogram of MOS for Time Scale MEAN
% figure('Position',[1680-500 200 500 250])
% h = histogram2([a.TSM],[a.mean_MOS_norm],[50 50],'FaceColor','flat');
% h.DisplayStyle = 'tile';
% view(2)
% colormap(flipud(gray));
% c = colorbar;
% c.Label.String = 'Count';
% title('Mean OS At Time Scales For All Methods')
% xlabel('Time Scale')
% ylabel('MOS')
% set(gca,...
%     'FontSize', 12, ...
%     'FontName', 'Times');
% print('Plots/PDF/MOS_2D_Histogram_mean', '-dpdf');
% print('Plots/EPSC/MOS_2D_Histogram_mean', '-depsc');
% print('Plots/PNG/MOS_2D_Histogram_mean', '-dpng');


%% Histogram of MOS for Time Scale MEDIAN
% figure('Position',[1680-500 200 500 250])
% h = histogram2([a.TSM],[a.median_MOS_norm],[50 50],'FaceColor','flat');
% h.DisplayStyle = 'tile';
% view(2)
% colormap(flipud(gray));
% c = colorbar;
% c.Label.String = 'Count';
% title('Median OS At Time Scales For All Methods')
% xlabel('Time Scale')
% ylabel('MOS')
% set(gca,...
%     'FontSize', 12, ...
%     'FontName', 'Times');
% print('Plots/PDF/MOS_2D_Histogram_median', '-dpdf');
% print('Plots/EPSC/MOS_2D_Histogram_median', '-depsc');
% print('Plots/PNG/MOS_2D_Histogram_median', '-dpng');





%% Histogram for Mean OS vs Std(MOS)


% figure('Position',[1680-500 200 500 250])
% h = histogram2([a.mean_MOS_norm],[a.std_MOS_norm],[50 50],'FaceColor','flat');
% h.DisplayStyle = 'tile';
% view(2)
% colormap(gray);
% c = colorbar;
% c.Label.String = 'Count';
% title('Mean Opinion Score vs Std(MOS)')
% xlabel('MOS')
% ylabel('Standard Deviation')
%
% set(gca,...
%     'FontSize', 12, ...
%     'FontName', 'Times');
% print('Plots/PDF/MeanOS_vs_Std', '-dpdf');
% print('Plots/EPSC/MeanOS_vs_Std', '-depsc');
% print('Plots/PNG/MeanOS_vs_Std', '-dpng');

%% Histogram for MedianOS vs Std(MOS)
% figure('Position',[1680-500 200 500 250])
% h = histogram2([a.median_MOS_norm],[a.std_MOS_norm],[50 50],'FaceColor','flat');
% h.DisplayStyle = 'tile';
% view(2)
% colormap(gray);
% c = colorbar;
% c.Label.String = 'Count';
% title('Median Opinion Score vs Std(MOS)')
% xlabel('MOS')
% ylabel('Standard Deviation')
%
% set(gca,...
%     'FontSize', 12, ...
%     'FontName', 'Times');
% print('Plots/PDF/MedianOS_vs_Std', '-dpdf');
% print('Plots/EPSC/MedianOS_vs_Std', '-depsc');
% print('Plots/PNG/MedianOS_vs_Std', '-dpng');



%%  ----- Plot the number of responses still required. -------

% resp = [a.num_responses];
% one = length(resp(resp==1));
% two = length(resp(resp==2));
% three = length(resp(resp==3));
% four = length(resp(resp==4));
% five = length(resp(resp==5));
% six = length(resp(resp==6));
%
% remaining = one*6 + two*5 + three*4 + four*3 + five*2 + six;
%
% % Total_ratings = sum(resp);
% % y = [Total_ratings-remaining remaining];
% % figure('Position',[1680-500 200 1200 628])
% % % labels = {[num2str(round((Total_ratings-remaining)/60)) ' Sets Complete' ], [num2str(round(remaining/60)) ' To Go' ]};
% % labels = {'Complete', num2str(round(remaining/60))};
% % p = pie(y, labels);
% % colormap([ 0.4 0.67 0.19 ;
% %     0.85 0.2 0.1])
% % for iHandle = 2:2:2*numel(labels)
% %     p(iHandle).Position = 0.7*p(iHandle).Position;
% % end
%
% Total_ratings = sum(resp);
% y = [Total_ratings 1];
% figure('Position',[1680-500 200 1200 628])
% % labels = {[num2str(round((Total_ratings-remaining)/60)) ' Sets Complete' ], [num2str(round(remaining/60)) ' To Go' ]};
% labels = {'Complete', ''};
% p = pie(y, labels);
% colormap([ 0.4 0.67 0.19 ;
%     0.85 0.2 0.1])
% for iHandle = 2:2:2*numel(labels)
%     p(iHandle).Position = 0.7*p(iHandle).Position;
% end
% set(gca,...
%     'FontSize', 12, ...
%     'FontName', 'Times');
% print('Plots/PDF/Ratings_to_go', '-dpdf');


%% ---- Compare results for expert and non-expert listeners ----

% MMADs = [];
% experts = [];
%
% for n = 1:length(u)
%     MMADs = [MMADs, u(n).mean_abs_diff_norm_mean];
%     experts = [experts u(n).expert*ones(size(u(n).mean_abs_diff_norm_mean))];
% end

% figure('Position',[1680-500 200 500 250])
% h = histogram2(experts,MMADs,[2 30],'FaceColor','flat');
% % h.ShowEmptyBins = 'On';
% axis([-0.5 1.5 0 1.1*max(MMADs)])
% % h.DisplayStyle = 'tile';
% view(2)
% colormap(gray);
% c = colorbar;
% c.Label.String = 'Count';
% title('MAD per session for Expert and Non-Expert Listeners')
% xticks([0.25, 0.75])
% xticklabels({'Non-Expert','Expert'});
% ylabel('MMAD');
% set(gca,...
%     'FontSize', 12, ...
%     'FontName', 'Times');
% print('Plots/PDF/Expert_vs_Non_Expert_2D_Hist', '-dpdf');
% print('Plots/EPSC/Expert_vs_Non_Expert_2D_Hist', '-depsc');
% print('Plots/PNG/Expert_vs_Non_Expert_2D_Hist', '-dpng');

% figure('Position',[1680-500 200 600 400])
% expert_MMADs = MMADs(experts==1);
% non_expert_MMADs = MMADs(experts==0);
% h1 = histogram(expert_MMADs);
% hold on
% h2 = histogram(non_expert_MMADs);
% hold off
% h1.Normalization = 'probability';
% h1.BinWidth = 0.02;
% h2.Normalization = 'probability';
% h2.BinWidth = 0.02;
% title('MMAD Relative Probability for Expert and Non-Expert Listeners')
% xlabel('MMAD')
% ylabel('Probability')
% legend('Expert','Non-Expert');
% set(gca,...
%     'FontSize', 12, ...
%     'FontName', 'Times');
% print('Plots/PDF/Expert_vs_Non_Expert_Hist', '-dpdf');
% print('Plots/EPSC/Expert_vs_Non_Expert_Hist', '-depsc');
% print('Plots/PNG/Expert_vs_Non_Expert_Hist', '-dpng');
fprintf('Compare Expert and Non-Expert Listeners\n')
MMADs = [];
experts = [];

for n = 1:length(u)
    MMADs = [MMADs, u(n).mean_abs_diff_norm_mean];
    experts = [experts u(n).expert*ones(size(u(n).mean_abs_diff_norm_mean))];
end
expert_MMADs = MMADs(experts==1);
non_expert_MMADs = MMADs(experts==0);
bins = 50;
EDGES = linspace(min([expert_MMADs non_expert_MMADs]),max([expert_MMADs non_expert_MMADs]),bins);
EDGES_PLOT = (EDGES(1:end-1)+EDGES(2:end))/2;
[Expert_Count, ~] = histcounts(expert_MMADs,EDGES,'Normalization','probability');
[Non_Expert_Count, ~] = histcounts(non_expert_MMADs,EDGES,'Normalization','probability');

figure('Position',[1680-500 200 500 250])
plot(EDGES_PLOT, Expert_Count,'k-')
hold on
plot(EDGES_PLOT, Non_Expert_Count,'k:')
hold off
% title('MAD for Expert and Non-Expert Listeners')
xlabel('$\bar{X}_s$','interpreter','latex')
ylabel('Normalised Probability')
axis([min(EDGES),max(EDGES),0,1.1*max([Expert_Count Non_Expert_Count])])
legend('Expert','Non-Expert','location','northeast');
set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');

print('Plots/PDF/Expert_vs_Non_Expert_Line', '-dpdf');
print('Plots/EPSC/Expert_vs_Non_Expert_Line', '-depsc');
print('Plots/PNG/Expert_vs_Non_Expert_Line', '-dpng');

% RMSE version
MMADs = [];
experts = [];

for n = 1:length(u)
    MMADs = [MMADs, u(n).RMSE_norm];
    experts = [experts u(n).expert*ones(size(u(n).RMSE_norm))];
end
expert_MMADs = MMADs(experts==1);
non_expert_MMADs = MMADs(experts==0);
bins = 50;
EDGES = linspace(min([expert_MMADs non_expert_MMADs]),max([expert_MMADs non_expert_MMADs]),bins);
EDGES_PLOT = (EDGES(1:end-1)+EDGES(2:end))/2;
[Expert_Count, ~] = histcounts(expert_MMADs,EDGES,'Normalization','probability');
[Non_Expert_Count, ~] = histcounts(non_expert_MMADs,EDGES,'Normalization','probability');

figure('Position',[1680-500 200 500 250])
plot(EDGES_PLOT, Expert_Count,'k-')
hold on
plot(EDGES_PLOT, Non_Expert_Count,'k:')
hold off
% title('MAD for Expert and Non-Expert Listeners')
xlabel('$\mathcal{L}$','interpreter','latex')
ylabel('Normalised Probability')
axis([min(EDGES),max(EDGES),0,1.1*max([Expert_Count Non_Expert_Count])])
legend('Expert','Non-Expert','location','northeast');
set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');

print('Plots/PDF/RMSE_Expert_vs_Non_Expert_Line', '-dpdf');
print('Plots/EPSC/RMSE_Expert_vs_Non_Expert_Line', '-depsc');
print('Plots/PNG/RMSE_Expert_vs_Non_Expert_Line', '-dpng');






g1=ones(1,size(expert_MMADs,2));
g2=2*ones(1,size(non_expert_MMADs,2));
x=[expert_MMADs(:);non_expert_MMADs(:)]' ;
g=[g1,g2];
[p1,ANOVATAB,STATS]=anova1(x,g,'on');
[H,P]=ttest2(expert_MMADs,non_expert_MMADs);
fid = fopen('log_Anon.txt','a');
fprintf(fid,'Expert vs Non-Expert\n')
fprintf(fid,'Reject Null hypothesis of different means at alpha=0.05: %d\n',H);
fprintf(fid,'p-value: %g\n',P);
fclose(fid);
print('Plots/PDF/Expert_Boxplot', '-dpdf');
print('Plots/EPSC/Expert_Boxplot', '-depsc');
print('Plots/PNG/Expert_Boxplot', '-dpng');







%% ---- Compare results for Participants hearing responses ----

% MMADs_h = [];
% hearing = [];
%
% for n = 1:length(u)
%     MMADs_h = [MMADs_h, u(n).mean_abs_diff_norm_mean];
%     hearing = [hearing u(n).hearing*ones(size(u(n).mean_abs_diff_norm_mean))];
% end

% figure('Position',[1680-500 200 500 250])
% h = histogram2(hearing,MMADs_h,[2 30],'FaceColor','flat');
% % h.ShowEmptyBins = 'On';
% axis([-0.5 1.5 0 1.1*max(MMADs_h)])
% % h.DisplayStyle = 'tile';
% view(2)
% colormap(gray);
% c = colorbar;
% c.Label.String = 'Count';
% title('MAD per session for Hearing')
% xticks([0.25, 0.75])
% xticklabels({'Known Issues','No Issues'});
% ylabel('MMAD');
% set(gca,...
%     'FontSize', 12, ...
%     'FontName', 'Times');
% print('Plots/PDF/Hearing_2D_Hist', '-dpdf');
% print('Plots/EPSC/Hearing_2D_Hist', '-depsc');
% print('Plots/PNG/Hearing_2D_Hist', '-dpng');
%
%
% figure('Position',[1680-500 200 600 400])
% good_hearing_MMADs = MMADs(hearing==1);
% bad_hearing_MMADs = MMADs(hearing==0);
% h1 = histogram(good_hearing_MMADs);
% hold on
% h2 = histogram(bad_hearing_MMADs);
% hold off
% h1.Normalization = 'probability';
% h1.BinWidth = 0.02;
% h2.Normalization = 'probability';
% h2.BinWidth = 0.02;
% title('MMAD Relative Probability for Hearing Responses')
% xlabel('MMAD')
% ylabel('Probability')
% legend('None','Any');
% set(gca,...
%     'FontSize', 12, ...
%     'FontName', 'Times');
% print('Plots/PDF/Hearing_Hist', '-dpdf');
% print('Plots/EPSC/Hearing_Hist', '-depsc');
% print('Plots/PNG/Hearing_Hist', '-dpng');

fprintf('Compare Hearing\n')
MMADs_h = [];
hearing = [];

for n = 1:length(u)
    MMADs_h = [MMADs_h, u(n).mean_abs_diff_norm_mean];
    hearing = [hearing u(n).hearing*ones(size(u(n).mean_abs_diff_norm_mean))];
end
good_hearing_MMADs = MMADs(hearing==1);
bad_hearing_MMADs = MMADs(hearing==0);
bins = 50;
EDGES = linspace(min([good_hearing_MMADs bad_hearing_MMADs]),max([good_hearing_MMADs bad_hearing_MMADs]),bins);
EDGES_PLOT = (EDGES(1:end-1)+EDGES(2:end))/2;

[Good_Hearing_Count, ~] = histcounts(good_hearing_MMADs,EDGES,'Normalization','probability');
[Bad_Hearing_Count, ~] = histcounts(bad_hearing_MMADs,EDGES,'Normalization','probability');

figure('Position',[1680-500 200 500 250])
% plot(EDGES_PLOT, Good_Hearing_Count*100,'k-')
% hold on
% plot(EDGES_PLOT, Bad_Hearing_Count*100,'k:')
% hold off

plot(EDGES_PLOT, Good_Hearing_Count,'k-')
hold on
plot(EDGES_PLOT, Bad_Hearing_Count,'k:')
hold off
% title('MAD for Hearing')
xlabel('$\bar{X}_s$','interpreter','latex')
ylabel('Normalised Probability')
legend('None','Any','Location','northeast');
axis([min(EDGES),max(EDGES),0,1.1*max([Good_Hearing_Count Bad_Hearing_Count])])
set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');
print('Plots/PDF/Hearing_Line', '-dpdf');
print('Plots/EPSC/Hearing_Line', '-depsc');
print('Plots/PNG/Hearing_Line', '-dpng');

%RMSE Version
MMADs_h = [];
hearing = [];

for n = 1:length(u)
    MMADs_h = [MMADs_h, u(n).RMSE_norm];
    hearing = [hearing u(n).hearing*ones(size(u(n).RMSE_norm))];
end
good_hearing_MMADs = MMADs(hearing==1);
bad_hearing_MMADs = MMADs(hearing==0);
bins = 50;
EDGES = linspace(min([good_hearing_MMADs bad_hearing_MMADs]),max([good_hearing_MMADs bad_hearing_MMADs]),bins);
EDGES_PLOT = (EDGES(1:end-1)+EDGES(2:end))/2;

[Good_Hearing_Count, ~] = histcounts(good_hearing_MMADs,EDGES,'Normalization','probability');
[Bad_Hearing_Count, ~] = histcounts(bad_hearing_MMADs,EDGES,'Normalization','probability');

figure('Position',[1680-500 200 500 250])
% plot(EDGES_PLOT, Good_Hearing_Count*100,'k-')
% hold on
% plot(EDGES_PLOT, Bad_Hearing_Count*100,'k:')
% hold off

plot(EDGES_PLOT, Good_Hearing_Count,'k-')
hold on
plot(EDGES_PLOT, Bad_Hearing_Count,'k:')
hold off
% title('MAD for Hearing')
xlabel('$\mathcal{L}$','interpreter','latex')
ylabel('Normalised Probability')
legend('None','Any','Location','northeast');
axis([min(EDGES),max(EDGES),0,1.1*max([Good_Hearing_Count Bad_Hearing_Count])])
set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');
print('Plots/PDF/RMSE_Hearing_Line', '-dpdf');
print('Plots/EPSC/RMSE_Hearing_Line', '-depsc');
print('Plots/PNG/RMSE_Hearing_Line', '-dpng');


g1=ones(1,size(good_hearing_MMADs,2));
g2=2*ones(1,size(bad_hearing_MMADs,2));
x=[good_hearing_MMADs(:);bad_hearing_MMADs(:)]' ;
g=[g1,g2];
[p1,ANOVATAB,STATS]=anova1(x,g);
[H,P]=ttest2(good_hearing_MMADs,bad_hearing_MMADs);
fid = fopen('log_Anon.txt','a');
fprintf(fid,'Hearing\n')
fprintf(fid,'Reject Null hypothesis of different means at alpha=0.05: %d\n',H);
fprintf(fid,'p-value: %g\n',P);
fclose(fid);


print('Plots/PDF/Hearing_Boxplot', '-dpdf');
print('Plots/EPSC/Hearing_Boxplot', '-depsc');
print('Plots/PNG/Hearing_Boxplot', '-dpng');



%% ----------------------- 2D Histogram of all responses minus MEAN -------------------------------

% figure('Position',[0 0 800 500])
% All = [a.mean_MOS_norm];
% [~,I] = sort(All);
% max_len = 1;
% for n = 1:length(I)
%     len = length(a(n).mean_MOS_norm);
%     if len>max_len
%         max_len = len;
%     end
% end
% All_results = zeros(length(I),max_len);
% x = [];
% y = [];
% for n = 1:length(All)
%     x = [x ; n*ones(length(a(I(n)).MOS_norm),1)];
%     y = [y a(I(n)).MOS_norm - a(I(n)).mean_MOS_norm];
% end
%
% h = histogram2(x,y',[100 100],'FaceColor','flat');
% % h.ShowEmptyBins = 'On';
% h.DisplayStyle = 'tile';
%
% % ax = gca;
% % ax.GridColor = [0.4 0.4 0.4];
% % ax.GridLineStyle = '--';
% % ax.GridAlpha = 0.5;
% % ax.XGrid = 'off';
% % ax.YGrid = 'on';
% % ax.Layer = 'top';
% view(2)
% % colormap(gray);
% c = colorbar;
% c.Label.String = 'Count';
% % c.Label.String = 'Probability';
% title_text = sprintf('All Opinion Scores Minus Mean Ordered by Ascending Mean (%d Ratings)',size([a.MOS],2));
% title(title_text)
%
% % yticks(1:5);
% % yticklabels({'Bad', 'Poor', 'Fair', 'Good', 'Excellent'})
%
% xlabel('File')
% ylabel('Opinion Score - Mean Opinion Score')
% % axis('tight')
% axis([0 length(a) -5 5])
% % hold on
% % p = plot3(1:5280, [a(I).mean_MOS_norm],(1+max(max(h.BinCounts)))*ones(1,5280),'r');
% % p.LineWidth = 2;
% % hold off
%
%
% % hold on
% % for n=1:5280
% %     plot3(n, median(a(I(n)).MOS_norm), max(max(h.BinCounts)),'c.');
% % %     p.LineWidth = 2;
% % end
% % hold off
% set(gca,...
%     'FontSize', 12, ...
%     'FontName', 'Times');
% print('Plots/PDF/All_Results_Hist_Minus_Mean', '-dpdf');
% print('Plots/EPSC/All_Results_Hist_Minus_Mean', '-depsc');
% print('Plots/PNG/All_Results_Hist_Minus_Mean', '-dpng');

%% ----------------------- 2D Histogram of all responses minus MEDIAN -------------------------------

% figure('Position',[0 0 800 500])
% All = [a.median_MOS_norm];
% [~,I] = sort(All);
% max_len = 1;
% for n = 1:length(I)
%     len = length(a(n).MOS_norm);
%     if len>max_len
%         max_len = len;
%     end
% end
% All_results = zeros(length(I),max_len);
% x = [];
% y = [];
% for n = 1:length(All)
%     x = [x ; n*ones(length(a(I(n)).MOS_norm),1)];
%     y = [y a(I(n)).MOS_norm-a(I(n)).median_MOS_norm];
% end
%
% h = histogram2(x,y',[100 100],'FaceColor','flat');
% % h.ShowEmptyBins = 'On';
% h.DisplayStyle = 'tile';
%
% % ax = gca;
% % ax.GridColor = [0.4 0.4 0.4];
% % ax.GridLineStyle = '--';
% % ax.GridAlpha = 0.5;
% % ax.XGrid = 'off';
% % ax.YGrid = 'on';
% % ax.Layer = 'top';
% view(2)
% % colormap(gray);
% c = colorbar;
% c.Label.String = 'Count';
% % c.Label.String = 'Probability';
% title_text = sprintf('All Opinion Scores Minus Median Ordered by Ascending Median (%d Ratings)',size([a.MOS],2));
% title(title_text)
%
% % yticks(1:5);
% % yticklabels({'Bad', 'Poor', 'Fair', 'Good', 'Excellent'})
%
% xlabel('File')
% ylabel('Opinion Score - Median Opinion Score')
% % axis('tight')
% axis([0 length(a) -5 5])
% % hold on
% % p = plot3(1:5280, [a(I).median_MOS_norm],(1+max(max(h.BinCounts)))*ones(1,5280),'r');
% % p.LineWidth = 2;
% % hold off
%
%
% % hold on
% % for n=1:5280
% %     plot3(n, median(a(I(n)).MOS_norm), max(max(h.BinCounts)),'c.');
% % %     p.LineWidth = 2;
% % end
% % hold off
% set(gca,...
%     'FontSize', 12, ...
%     'FontName', 'Times');
% print('Plots/PDF/All_Results_Hist_Minus_Median', '-dpdf');
% print('Plots/EPSC/All_Results_Hist_Minus_Median', '-depsc');
% print('Plots/PNG/All_Results_Hist_Minus_Median', '-dpng');



%% ----------  How to remove outliers or unreliable data? ----------

%Consider the Mean Absolute difference to the Mean or Median.
% bins = 50;
% EDGES = linspace(0,max([[u.mean_abs_diff_norm_mean] [u.mean_abs_diff_norm_median] [u.mean_abs_diff_mean] [u.mean_abs_diff_median]]),bins);
% EDGES_PLOT = (EDGES(1:end-1)+EDGES(2:end))/2;
%
% [N_MAD_NMEAN, ~] = histcounts([u.mean_abs_diff_norm_mean],EDGES);
% [N_MAD_NMEDIAN, ~] = histcounts([u.mean_abs_diff_norm_median],EDGES);
% [N_MAD_MEAN, ~] = histcounts([u.mean_abs_diff_mean],EDGES);
% [N_MAD_MEDIAN, ~] = histcounts([u.mean_abs_diff_median],EDGES);
%
% figure('Position',[1680-500 200 500 250])
% plot(EDGES_PLOT, N_MAD_MEAN, 'k-')
% hold on
% plot(EDGES_PLOT, N_MAD_MEDIAN, 'k:')
% plot(EDGES_PLOT, N_MAD_NMEAN, 'k- .')
% plot(EDGES_PLOT, N_MAD_NMEDIAN, 'k:.')
% hold off
% % title('MAD Values')
% xlabel('$\bar{X}_s$','interpreter','latex')
% ylabel('Count')
% legend('Mean', 'Median', 'Norm Mean', 'Norm Median','location','best');
% set(gca,...
%     'FontSize', 12, ...
%     'FontName', 'Times');
% print('Plots/PDF/MAD_Mean_vs_Median_Pre_Post_Norm', '-dpdf');
% print('Plots/EPSC/MAD_Mean_vs_Median_Pre_Post_Norm', '-depsc');
% print('Plots/PNG/MAD_Mean_vs_Median_Pre_Post_Norm', '-dpng');

fprintf('Compare MAD to Mean and Median\n')
%Consider the Mean Absolute difference to the Mean or Median.
bins = 50;
EDGES = linspace(min([[u.mean_abs_diff_norm_mean] [u.mean_abs_diff_norm_median] [u.mean_abs_diff_mean] [u.mean_abs_diff_median]]),...
    max([[u.mean_abs_diff_norm_mean] [u.mean_abs_diff_norm_median] [u.mean_abs_diff_mean] [u.mean_abs_diff_median]]),bins);
EDGES_PLOT = (EDGES(1:end-1)+EDGES(2:end))/2;

[N_MAD_NMEAN, ~] = histcounts([u.mean_abs_diff_norm_mean],EDGES,'Normalization','probability');
[N_MAD_MEAN, ~] = histcounts([u.mean_abs_diff_mean],EDGES,'Normalization','probability');

figure('Position',[1680-500 200 500 250])
plot(EDGES_PLOT, N_MAD_MEAN, 'k:')
hold on
plot(EDGES_PLOT, N_MAD_NMEAN, 'k-')
hold off
% title('MAD Values')
xlabel('$\bar{X}_s$','interpreter','latex')
ylabel('Normalised Probability')
axis([min(EDGES),max(EDGES),0,1.1*max([N_MAD_NMEAN N_MAD_MEAN])])
legend('Raw', 'Normalised','location','northeast');
set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');
print('Plots/PDF/MAD_Mean_Pre_Post_Norm', '-dpdf');
print('Plots/EPSC/MAD_Mean_Pre_Post_Norm', '-depsc');
print('Plots/PNG/MAD_Mean_Pre_Post_Norm', '-dpng');


% RMSE version
bins = 50;
EDGES = linspace(min([ [u.RMSE_norm] [u.RMSE]]),...
                 max([[u.RMSE_norm] [u.RMSE]]),bins);
EDGES_PLOT = (EDGES(1:end-1)+EDGES(2:end))/2;

[N_MAD_NMEAN, ~] = histcounts([u.RMSE_norm],EDGES,'Normalization','probability');
[N_MAD_MEAN, ~] = histcounts([u.RMSE],EDGES,'Normalization','probability');

figure('Position',[1680-500 200 500 250])
plot(EDGES_PLOT, N_MAD_MEAN, 'k:')
hold on
plot(EDGES_PLOT, N_MAD_NMEAN, 'k-')
hold off
% title('MAD Values')
xlabel('$\mathcal{L}$','interpreter','latex')
ylabel('Normalised Probability')
axis([min(EDGES),max(EDGES),0,1.1*max([N_MAD_MEAN N_MAD_NMEAN])])
legend('Raw', 'Normalised','location','northeast');
set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');
print('Plots/PDF/RMSE_Pre_Post_Norm', '-dpdf');
print('Plots/EPSC/RMSE_Pre_Post_Norm', '-depsc');
print('Plots/PNG/RMSE_Pre_Post_Norm', '-dpng');

%Consider the difference in MAD before and after Normalisation

% for n = 1:length(u)
%     for k = 1:length(u(n).mean_abs_diff_mean)
%         u(n).MAD_mean_diff(k) = u(n).mean_abs_diff_mean(k)-u(n).mean_abs_diff_norm_mean(k);
%         u(n).MAD_median_diff(k) = u(n).mean_abs_diff_median(k)-u(n).mean_abs_diff_norm_median(k);
%     end
% end
%
% bins = 50;
% EDGES = linspace(min([[u.MAD_mean_diff] [u.MAD_median_diff]]),max([[u.MAD_mean_diff] [u.MAD_median_diff]]),bins);
% EDGES_PLOT = (EDGES(1:end-1)+EDGES(2:end))/2;
%
% [N_MAD_NMEAN_DIFF, ~] = histcounts([u.MAD_mean_diff],EDGES);
% [N_MAD_NMEDIAN_DIFF, ~] = histcounts([u.MAD_median_diff],EDGES);
%
% figure('Position',[1680-500 200 500 250])
% plot(EDGES_PLOT, N_MAD_NMEAN_DIFF, 'b-x')
% hold on
% plot(EDGES_PLOT, N_MAD_NMEDIAN_DIFF, 'r-x')
% hold off
% title('MAD Difference Values')
% xlabel('MAD Pre/Post Normalisation (Pre minus Post)')
% ylabel('Count')
% legend('Mean','Median');
% set(gca,...
%     'FontSize', 12, ...
%     'FontName', 'Times');
% print('Plots/PDF/MAD_Mean_vs_Median', '-dpdf');
% print('Plots/EPSC/MAD_Mean_vs_Median', '-depsc');
% print('Plots/PNG/MAD_Mean_vs_Median', '-dpng');


%% --------------------- Plot the STD of MAD compared to old STD MAD ---------------------
%MEAN

% figure('Position',[1680-500 200 500 250])
% for n = 1:length(u)
%     u(n).MAD_diff_std_mean = u(n).std_abs_diff_mean - u(n).std_abs_diff_norm_mean;
% end
% plot(abs([u.MAD_diff_std_mean]),'k.')
% title('Comparison of STD of MAD values (Old-New Means)')
%
% figure('Position',[1680-500 200 500 250])
% plot([u.num_files],abs([u.MAD_diff_std_mean]),'k.')
% title('STD MAD Means difference Pre and Post Normalisation')
% xlabel('Number of Files in Session')
% ylabel('Difference')
% set(gca,...
%     'FontSize', 12, ...
%     'FontName', 'Times');
% print('Plots/PDF/STD_MAD_difference_pre_post_norm_mean', '-dpdf');
% print('Plots/EPSC/STD_difference_pre_post_norm_mean', '-depsc');
% print('Plots/PNG/STD_difference_pre_post_norm_mean', '-dpng');

% diff_threshold = 0.5;
% for n = 1:length(u)
%     for k = 1:length(u(n).MAD_diff)
%         if u(n).MAD_diff(k) > diff_threshold
%             fprintf('%s with mean MAD difference of %.3f with %d files\n',u(n).name, u(n).MAD_diff(k), u(n).num_files(k));
%         end
%     end
% end

%MEDIAN

% figure('Position',[1680-500 200 500 250])
% for n = 1:length(u)
%     u(n).MAD_diff_std_median = u(n).std_abs_diff_median - u(n).std_abs_diff_norm_median;
% end
% plot(abs([u.MAD_diff_std_median]),'k.')
% title('Comparison of STD of MAD values (Old-New Medians)')
%
% figure('Position',[1680-500 200 500 250])
% plot([u.num_files],abs([u.MAD_diff_std_median]),'k.')
% % title('STD MAD Medians difference Pre and Post Normalisation')
% xlabel('Number of Files in Session')
% ylabel('$\bar{X}_s$ (Median)','interpreter','latex')
% set(gca,...
%     'FontSize', 12, ...
%     'FontName', 'Times');
% print('Plots/PDF/STD_MAD_difference_pre_post_norm_median', '-dpdf');
% print('Plots/EPSC/STD_MAD_difference_pre_post_norm_median', '-depsc');
% print('Plots/PNG/STD_MAD_difference_pre_post_norm_median', '-dpng');

% diff_threshold = 0.5;
% for n = 1:length(u)
%     for k = 1:length(u(n).MAD_diff_median)
%         if u(n).MAD_diff_median(k) > diff_threshold
%             fprintf('%s with median MAD difference of %.3f with %d files\n',u(n).name, u(n).MAD_diff_median(k), u(n).num_files(k));
%         end
%     end
% end

% figure('Position',[1680-500 200 500 250])
% plot([u.mean_abs_diff_mean],[u.std_abs_diff_mean],'k.')
% % title('Mean Absolute Difference to Mean vs Std Dev of MADmean');
% xlabel('$\bar{X}_s$ (Pre-Norm)','interpreter','latex');
% ylabel('$\sigma_s$ (Pre-Norm)','interpreter','latex');
% set(gca,...
%     'FontSize', 12, ...
%     'FontName', 'Times');
% print('Plots/PDF/MADmean_vs_STD_pre_Norm', '-dpdf');
% print('Plots/EPSC/MADmean_vs_STD_pre_Norm', '-depsc');
% print('Plots/PNG/MADmean_vs_STD_pre_Norm', '-dpng');
%
%
% figure('Position',[1680-500 200 500 250])
% plot([u.MMAD_norm_mean_new],[u.MSTD_mean_new],'k.')
% % title('Mean Absolute Difference to Normalised Mean vs Std Dev of MADmean');
% xlabel('$\bar{X}_s$ (Post-Norm)','interpreter','latex');
% ylabel('$\sigma_s$ (Post-Norm)','interpreter','latex');
% set(gca,...
%     'FontSize', 12, ...
%     'FontName', 'Times');
% print('Plots/PDF/MADmean_vs_STD_post_Norm', '-dpdf');
% print('Plots/EPSC/MADmean_vs_STD_post_Norm', '-depsc');
% print('Plots/PNG/MADmean_vs_STD_post_Norm', '-dpng');


%% Plots for the Fuzzy TSM files
fprintf('FuzzyPV Plots\n')
figure('Position',[1680-500 200 500 250])
hold on
for n = 5281:5360
    plot(a(n).TSM*ones(size(a(n).MOS)),a(n).MOS,'k.')
    plot(a(n).TSM,a(n).mean_MOS,'rx')
end
hold off
% title('Mean OS At Time Scales For FuzzyPV')
xlabel('Time Scale')
ylabel('MOS')
axis( [ min([a(5281:5360).TSM]) max([a(5281:5360).TSM]) min([a(5281:5360).MOS_norm]) max([a(5281:5360).MOS_norm]) ] )
set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');
print('Plots/PDF/MOS_FuzzyPV_Mean', '-dpdf');
print('Plots/EPSC/MOS_FuzzyPV_Mean', '-depsc');
print('Plots/PNG/MOS_FuzzyPV_Mean', '-dpng');

figure('Position',[1680-500 200 500 250])
hold on
for n = 5281:5360
    plot(a(n).TSM*ones(size(a(n).MOS_norm)),a(n).MOS_norm,'k.')
    plot(a(n).TSM,a(n).mean_MOS_norm,'rx')
end
hold off
% title('Normalised MeanOS At Time Scales For FuzzyPV')
xlabel('Time Scale')
ylabel('MOS')
axis( [ min([a(5281:5360).TSM]) max([a(5281:5360).TSM]) min([a(5281:5360).MOS_norm]) max([a(5281:5360).MOS_norm]) ] )

set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');
print('Plots/PDF/MOS_FuzzyPV_Mean_Norm', '-dpdf');
print('Plots/EPSC/MOS_FuzzyPV_Mean_Norm', '-depsc');
print('Plots/PNG/MOS_FuzzyPV_Mean_Norm', '-dpng');



figure('Position',[1680-500 200 700 500])
for n = 5281:5360
    subplot(221)
    hold on
    plot(a(n).TSM*ones(size(a(n).MOS_norm)),a(n).MOS_norm,'k.')
    plot(a(n).TSM,a(n).mean_MOS_norm,'rx')
    hold off
    switch a(n).cat
%         case 'Complex'
%             subplot(221)
%             hold on
%             plot(a(n).TSM*ones(size(a(n).MOS_norm)),a(n).MOS_norm,'k.')
%             plot(a(n).TSM,a(n).mean_MOS_norm,'rx')
%             hold off
        case 'Music'
            subplot(222)
            hold on
            plot(a(n).TSM*ones(size(a(n).MOS_norm)),a(n).MOS_norm,'k.')
            plot(a(n).TSM,a(n).mean_MOS_norm,'rx')
            hold off
        case 'Solo'
            subplot(223)
            hold on
            plot(a(n).TSM*ones(size(a(n).MOS_norm)),a(n).MOS_norm,'k.')
            plot(a(n).TSM,a(n).mean_MOS_norm,'rx')
            hold off
        case 'Voice'
            subplot(224)
            hold on
            plot(a(n).TSM*ones(size(a(n).MOS_norm)),a(n).MOS_norm,'k.')
            plot(a(n).TSM,a(n).mean_MOS_norm,'rx')
            hold off
    end
end
subplot(221)
title('All Files')
xlabel('Time Scale')
ylabel('Mean OS')
axis( [ min([a(5281:5360).TSM]) max([a(5281:5360).TSM]) min([a(5281:5360).MOS_norm]) max([a(5281:5360).MOS_norm]) ] )
set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');

subplot(222)
title('Music Files')
xlabel('Time-Scale Ratio (\beta)')
ylabel('Mean OS')
set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');

subplot(223)
title('Solo Files')
xlabel('Time-Scale Ratio (\beta)')
ylabel('Mean OS')
set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');

subplot(224)
title('Voice Files')
xlabel('Time-Scale Ratio (\beta)')
ylabel('Mean OS')
set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');

% suptitle('FuzzyPV')

print('Plots/PDF/FuzzyPV', '-dpdf');
print('Plots/EPSC/FuzzyPV', '-depsc');
print('Plots/PNG/FuzzyPV', '-dpng');


%% Plots for the NMF TSM files
fprintf('NMFTSM plots\n')
figure('Position',[1680-500 200 500 250])
hold on
for n = 5441:5520
    plot(a(n).TSM*ones(size(a(n).MOS)),a(n).MOS,'k.')
    plot(a(n).TSM,a(n).mean_MOS,'rx')
end
hold off
% title('Mean OS At Time Scales For FuzzyPV')
xlabel('Time Scale')
ylabel('MOS')
axis( [ min([a(5441:5520).TSM]) max([a(5441:5520).TSM]) min([a(5441:5520).MOS_norm]) max([a(5441:5520).MOS_norm]) ] )
set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');
print('Plots/PDF/MOS_NMFTSM_Mean', '-dpdf');
print('Plots/EPSC/MOS_NMFTSM_Mean', '-depsc');
print('Plots/PNG/MOS_NMFTSM_Mean', '-dpng');

figure('Position',[1680-500 200 500 250])
hold on
for n = 5441:5520
    plot(a(n).TSM*ones(size(a(n).MOS_norm)),a(n).MOS_norm,'k.')
    plot(a(n).TSM,a(n).mean_MOS_norm,'rx')
end
hold off
% title('Normalised MeanOS At Time Scales For FuzzyPV')
xlabel('Time Scale')
ylabel('MOS')
axis( [ min([a(5441:5520).TSM]) max([a(5441:5520).TSM]) min([a(5441:5520).MOS_norm]) max([a(5441:5520).MOS_norm]) ] )

set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');
print('Plots/PDF/MOS_NMFTSM_Mean_Norm', '-dpdf');
print('Plots/EPSC/MOS_NMFTSM_Mean_Norm', '-depsc');
print('Plots/PNG/MOS_NMFTSM_Mean_Norm', '-dpng');



figure('Position',[1680-500 200 700 500])
for n = 5441:5520
    subplot(221)
    hold on
    plot(a(n).TSM*ones(size(a(n).MOS_norm)),a(n).MOS_norm,'k.')
    plot(a(n).TSM,a(n).mean_MOS_norm,'rx')
    hold off
    switch a(n).cat
%         case 'Complex'
%             subplot(221)
%             hold on
%             plot(a(n).TSM*ones(size(a(n).MOS_norm)),a(n).MOS_norm,'k.')
%             plot(a(n).TSM,a(n).mean_MOS_norm,'rx')
%             hold off
        case 'Music'
            subplot(222)
            hold on
            plot(a(n).TSM*ones(size(a(n).MOS_norm)),a(n).MOS_norm,'k.')
            plot(a(n).TSM,a(n).mean_MOS_norm,'rx')
            hold off
        case 'Solo'
            subplot(223)
            hold on
            plot(a(n).TSM*ones(size(a(n).MOS_norm)),a(n).MOS_norm,'k.')
            plot(a(n).TSM,a(n).mean_MOS_norm,'rx')
            hold off
        case 'Voice'
            subplot(224)
            hold on
            plot(a(n).TSM*ones(size(a(n).MOS_norm)),a(n).MOS_norm,'k.')
            plot(a(n).TSM,a(n).mean_MOS_norm,'rx')
            hold off
    end
end
subplot(221)
title('All Files')
xlabel('Time Scale')
ylabel('Mean OS')
axis( [ min([a(5441:5520).TSM]) max([a(5441:5520).TSM]) min([a(5441:5520).MOS_norm]) max([a(5441:5520).MOS_norm]) ] )
set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');
% subplot(221)
% title('Complex Files')
% xlabel('Time-Scale Ratio (\beta)')
% ylabel('Mean OS')
% set(gca,...
%     'FontSize', 12, ...
%     'FontName', 'Times');

subplot(222)
title('Music Files')
xlabel('Time-Scale Ratio (\beta)')
ylabel('Mean OS')
set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');

subplot(223)
title('Solo Files')
xlabel('Time-Scale Ratio (\beta)')
ylabel('Mean OS')
set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');

subplot(224)
title('Voice Files')
xlabel('Time-Scale Ratio (\beta)')
ylabel('Mean OS')
set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');

% suptitle('FuzzyPV')

print('Plots/PDF/NMFTSM', '-dpdf');
print('Plots/EPSC/NMFTSM', '-depsc');
print('Plots/PNG/NMFTSM', '-dpng');


%% Plots for the Elastique TSM files
fprintf('Elastique Plotting\n')
figure('Position',[1680-500 200 500 250])
hold on
for n = 5361:5440
    plot(a(n).TSM*ones(size(a(n).MOS)),a(n).MOS,'k.')
    plot(a(n).TSM,a(n).mean_MOS,'rx')
end
hold off
% title('Mean OS At Time Scales For FuzzyPV')
xlabel('Time Scale')
ylabel('MOS')
axis( [ min([a(5361:5440).TSM]) max([a(5361:5440).TSM]) min([a(5361:5440).MOS_norm]) max([a(5361:5440).MOS_norm]) ] )
set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');
print('Plots/PDF/MOS_Elastique_Mean', '-dpdf');
print('Plots/EPSC/MOS_Elastique_Mean', '-depsc');
print('Plots/PNG/MOS_Elastique_Mean', '-dpng');

figure('Position',[1680-500 200 500 250])
hold on
for n = 5361:5440
    plot(a(n).TSM*ones(size(a(n).MOS_norm)),a(n).MOS_norm,'k.')
    plot(a(n).TSM,a(n).mean_MOS_norm,'rx')
end
hold off
% title('Normalised MeanOS At Time Scales For FuzzyPV')
xlabel('Time Scale')
ylabel('MOS')
axis( [ min([a(5361:5440).TSM]) max([a(5361:5440).TSM]) min([a(5361:5440).MOS_norm]) max([a(5361:5440).MOS_norm]) ] )

set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');
print('Plots/PDF/MOS_Elastique_Mean_Norm', '-dpdf');
print('Plots/EPSC/MOS_Elastique_Mean_Norm', '-depsc');
print('Plots/PNG/MOS_Elastique_Mean_Norm', '-dpng');



figure('Position',[1680-500 200 700 500])
for n = 5361:5440
    subplot(221)
    hold on
    plot(a(n).TSM*ones(size(a(n).MOS_norm)),a(n).MOS_norm,'k.')
    plot(a(n).TSM,a(n).mean_MOS_norm,'rx')
    hold off
    switch a(n).cat
%         case 'Complex'
%             subplot(221)
%             hold on
%             plot(a(n).TSM*ones(size(a(n).MOS_norm)),a(n).MOS_norm,'k.')
%             plot(a(n).TSM,a(n).mean_MOS_norm,'rx')
%             hold off
        case 'Music'
            subplot(222)
            hold on
            plot(a(n).TSM*ones(size(a(n).MOS_norm)),a(n).MOS_norm,'k.')
            plot(a(n).TSM,a(n).mean_MOS_norm,'rx')
            hold off
        case 'Solo'
            subplot(223)
            hold on
            plot(a(n).TSM*ones(size(a(n).MOS_norm)),a(n).MOS_norm,'k.')
            plot(a(n).TSM,a(n).mean_MOS_norm,'rx')
            hold off
        case 'Voice'
            subplot(224)
            hold on
            plot(a(n).TSM*ones(size(a(n).MOS_norm)),a(n).MOS_norm,'k.')
            plot(a(n).TSM,a(n).mean_MOS_norm,'rx')
            hold off
    end
end
subplot(221)
title('All Files')
xlabel('Time Scale')
ylabel('Mean OS')
axis( [ min([a(5361:5440).TSM]) max([a(5361:5440).TSM]) min([a(5361:5440).MOS_norm]) max([a(5361:5440).MOS_norm]) ] )
set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');
% subplot(221)
% title('Complex Files')
% xlabel('Time-Scale Ratio (\beta)')
% ylabel('Mean OS')
% set(gca,...
%     'FontSize', 12, ...
%     'FontName', 'Times');

subplot(222)
title('Music Files')
xlabel('Time-Scale Ratio (\beta)')
ylabel('Mean OS')
set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');

subplot(223)
title('Solo Files')
xlabel('Time-Scale Ratio (\beta)')
ylabel('Mean OS')
set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');

subplot(224)
title('Voice Files')
xlabel('Time-Scale Ratio (\beta)')
ylabel('Mean OS')
set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');

% suptitle('FuzzyPV')

print('Plots/PDF/Elastique', '-dpdf');
print('Plots/EPSC/Elastique', '-depsc');
print('Plots/PNG/Elastique', '-dpng');

%% -------------------------  MEDIAN RESULTS TO REMOVE -------------------------

%Calculate the outlier values
fprintf('Plotting outliers\n')
fprintf('Pre-Normalisation\n')
fprintf('Mean STD of File Ratings = %g\n',mean([a.std_MOS]));
fid = fopen('log_Anon.txt','a');
fprintf(fid,'\nMedian Outliers\n');
fprintf(fid,'Raw Outliers\n');
session_MAD = [u.mean_abs_diff_mean];
% TF = isoutlier(session_MAD,'mean');
% TF = isoutlier(session_MAD,'grubbs');
TF = isoutlier(session_MAD);
mean_outlier_MAD = session_MAD(TF);

% session_STD = [u.std_abs_diff_mean];
% % TF = isoutlier(session_STD,'mean');
% % TF = isoutlier(session_STD,'grubbs');
% TF = isoutlier(session_STD);
% mean_outlier_STD = session_STD(TF);
%
% session_MD = [u.mean_diff];
% % TF = isoutlier(session_MD,'mean');
% % TF = isoutlier(session_MD,'grubbs');
% TF = isoutlier(session_MD);
% mean_outlier_MD = session_MD(TF);

session_PCC = [u.pearson_corr_mean];
% TF = isoutlier(session_PCC,'mean');
% TF = isoutlier(session_PCC,'grubbs');
TF = isoutlier(session_PCC);
mean_outlier_PCC = session_PCC(TF);

figure('Position',[1680-500 200 500 250])
hold on
for n = 1:length(u)
    for k = 1:length(u(n).mean_diff)
        plot(u(n).pearson_corr_mean(k), u(n).mean_abs_diff_mean(k),'k.')
        if sum(u(n).pearson_corr_mean(k)==mean_outlier_PCC)>0
            plot(u(n).pearson_corr_mean(k),u(n).mean_abs_diff_mean(k),'b+')
            fprintf('%s with filename %s is a PCC outlier (%g) with %d files\n',u(n).name, u(n).filename(9:end), u(n).pearson_corr_mean(k), u(n).num_files(k))
            fprintf(fid,'%s with filename %s is a PCC outlier (%g) with %d files\n',u(n).name, u(n).filename(9:end), u(n).pearson_corr_mean(k), u(n).num_files(k));
        end
        if sum(u(n).mean_abs_diff_mean(k)==mean_outlier_MAD)>0
            plot(u(n).pearson_corr_mean(k),u(n).mean_abs_diff_mean(k),'rx')
            fprintf('%s with filename %s is a MAD outlier (%g) with %d files\n',u(n).name, u(n).filename(9:end), u(n).mean_abs_diff_mean(k), u(n).num_files(k))
            fprintf(fid,'%s with filename %s is a MAD outlier (%g) with %d files\n',u(n).name, u(n).filename(9:end), u(n).mean_abs_diff_mean(k), u(n).num_files(k));

        end
    end
end
hold off
title('PCC vs MAD for Raw MeanOS')
% title('MD vs MAD vs STD-AD to Mean (MAD and STD-AD 1st outlier sets) Removed Pre-Normalisation')
% title('MD vs MAD vs STD-AD to Mean (MAD and STD-AD 2nd outlier sets) Removed Pre-Normalisation')
xlabel('$\rho$','Interpreter','latex')
ylabel('$\bar{X}_s$','interpreter','latex','Rotation',0)
axis([0.8*min([u.pearson_corr_mean]) 1.05*max([u.pearson_corr_mean]) ...
      0 1.05*max([u.mean_abs_diff_mean])])
text(0.2,.2,'MAD Outliers','Color','r')
text(0.2,.1,'PCC Outliers','Color','b')

set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');

print('Plots/PDF/Outliers_Raw_Median', '-dpdf');
print('Plots/EPSC/Outliers_Raw_Median', '-depsc');
print('Plots/PNG/Outliers_Raw_Median', '-dpng');



%Plot the normalised values with the original outliers

fprintf('Post-Normalisation\n')
fprintf('Mean STD of Normalised File Ratings = %g\n',mean([a.std_MOS_norm]));
session_MAD_norm = [u.mean_abs_diff_norm_mean];
TF = isoutlier(session_MAD_norm);
mean_outlier_MAD_norm = session_MAD_norm(TF);

% session_STD_norm = [u.std_abs_diff_norm_mean];
% TF = isoutlier(session_STD_norm);
% mean_outlier_STD_norm = session_STD_norm(TF);
%
% session_MD_norm = [u.mean_diff_norm];
% TF = isoutlier(session_MD_norm);
% mean_outlier_MD_norm = session_MD_norm(TF);

session_PCC = [u.pearson_corr_MeanOS_norm];
TF = isoutlier(session_PCC);
mean_norm_outlier_PCC = session_PCC(TF);


figure('Position',[1680-500 200 500 250])
hold on
for n = 1:length(u)
    for k = 1:length(u(n).mean_diff)
        plot(u(n).pearson_corr_MeanOS_norm(k),u(n).mean_abs_diff_norm_mean(k),'k.')
        if sum(u(n).pearson_corr_MeanOS_norm(k)==mean_norm_outlier_PCC)>0
            plot(u(n).pearson_corr_MeanOS_norm(k),u(n).mean_abs_diff_norm_mean(k),'b+')
            %             fprintf('%s with key %s is a MD outlier with %d files\n',u(n).name, u(n).key, u(n).num_files(k))
        end
        if sum(u(n).mean_abs_diff_mean(k)==mean_outlier_MAD)>0
            plot(u(n).pearson_corr_MeanOS_norm(k),u(n).mean_abs_diff_norm_mean(k),'rx')
            %             fprintf('%s with key %s is a MAD outlier with %d files\n',u(n).name, u(n).key, u(n).num_files(k))
        end
    end
end
hold off
title('PCC vs MAD for Norm. MeanOS (Raw outliers)')
% title('MD vs MAD vs STD-AD to Mean (MAD and STD-AD 1st outlier sets) Removed Post-Normalisation (Pre-norm outliers marked)')
% title('MD vs MAD vs STD-AD to Mean (MAD and STD-AD 2nd outlier sets) Removed Post-Normalisation (Pre-norm outliers marked)')

xlabel('$\rho$','Interpreter','latex')
ylabel('$\bar{X}_s$','interpreter','latex','Rotation',0)

axis([0.8*min([u.pearson_corr_MeanOS_norm]) 1.05*max([u.pearson_corr_MeanOS_norm]) ...
      0 1.05*max([u.mean_abs_diff_norm_mean]) ])
text(0.2,0.2,'MAD Outliers','Color','r')
text(0.2,0.1,'PCC Outliers','Color','b')

set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');

print('Plots/PDF/Outliers_Raw_Norm_Results_Median', '-dpdf');
print('Plots/EPSC/Outliers_Raw_Norm_Results_Median', '-depsc');
print('Plots/PNG/Outliers_Raw_Norm_Results_Median', '-dpng');




% POST NORMALISATION COMPARISON

fprintf(fid,'\nNorm Outliers\n');

figure('Position',[1680-500 200 500 250])
hold on
for n = 1:length(u)
    for k = 1:length(u(n).mean_diff_norm)
        plot(u(n).pearson_corr_MeanOS_norm(k),u(n).mean_abs_diff_norm_mean(k),'k.')
        if sum(u(n).pearson_corr_MeanOS_norm(k)==mean_norm_outlier_PCC)>0
            plot(u(n).pearson_corr_MeanOS_norm(k),u(n).mean_abs_diff_norm_mean(k),'b+')
            fprintf('%s with filename %s is a PCC outlier (%g) with %d files\n', u(n).name, u(n).filename(9:end), u(n).pearson_corr_MeanOS_norm(k), u(n).num_files(k))
            fprintf(fid,'%s with filename %s is a PCC outlier (%g) with %d files\n', u(n).name, u(n).filename(9:end), u(n).pearson_corr_MeanOS_norm(k), u(n).num_files(k));
        end
        if sum(u(n).mean_abs_diff_norm_mean(k)==mean_outlier_MAD_norm)>0
            plot(u(n).pearson_corr_MeanOS_norm(k),u(n).mean_abs_diff_norm_mean(k),'rx')
            fprintf('%s with filename %s is a MAD outlier (%g) with %d files\n', u(n).name,u(n).filename(9:end), u(n).mean_abs_diff_norm_mean(k), u(n).num_files(k))
            fprintf(fid,'%s with filename %s is a MAD outlier (%g) with %d files\n', u(n).name,u(n).filename(9:end), u(n).mean_abs_diff_norm_mean(k), u(n).num_files(k));
        end
    end
end
hold off

title('PCC vs MAD for Norm. MeanOS')
% title('MD vs MAD vs STD-AD to Mean (MAD and STD-AD 1st outlier sets) Removed Post-Normalisation')
% title('MD vs MAD vs STD-AD to Mean (MAD and STD-AD 2nd outlier sets) Removed Post-Normalisation')
xlabel('$\rho$','Interpreter','latex')
ylabel('$\bar{X}_s$','interpreter','latex','Rotation',0)
axis([0.8*min([u.pearson_corr_MeanOS_norm]) 1.05*max([u.pearson_corr_MeanOS_norm]) ...
      0 1.05*max([u.mean_abs_diff_norm_mean])])
text(0.2,0.2,'MAD Outliers','Color','r')
text(0.2,0.1,'PCC Outliers','Color','b')

set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');


print('Plots/PDF/Outliers_Norm_Median', '-dpdf');
print('Plots/EPSC/Outliers_Norm_Median', '-depsc');
print('Plots/PNG/Outliers_Norm_Median', '-dpng');

fclose(fid);


%% -------------------------  GRUBBS RESULTS TO REMOVE -------------------------

%Calculate the outlier values

fprintf('Pre-Normalisation\n')
fprintf('Mean STD of File Ratings = %g\n',mean([a.std_MOS]));
fid = fopen('log_Anon.txt','a');
fprintf(fid,'\nGrubbs Outliers\n');
fprintf(fid,'Raw Outliers\n');
session_MAD = [u.mean_abs_diff_mean];
% TF = isoutlier(session_MAD,'mean');
TF = isoutlier(session_MAD,'grubbs');
% TF = isoutlier(session_MAD);
mean_outlier_MAD = session_MAD(TF);

% session_STD = [u.std_abs_diff_mean];
% % TF = isoutlier(session_STD,'mean');
% % TF = isoutlier(session_STD,'grubbs');
% TF = isoutlier(session_STD);
% mean_outlier_STD = session_STD(TF);
%
% session_MD = [u.mean_diff];
% % TF = isoutlier(session_MD,'mean');
% % TF = isoutlier(session_MD,'grubbs');
% TF = isoutlier(session_MD);
% mean_outlier_MD = session_MD(TF);

session_PCC = [u.pearson_corr_mean];
% TF = isoutlier(session_PCC,'mean');
TF = isoutlier(session_PCC,'grubbs');
% TF = isoutlier(session_PCC);
mean_outlier_PCC = session_PCC(TF);

figure('Position',[1680-500 200 500 250])
hold on
for n = 1:length(u)
    for k = 1:length(u(n).mean_diff)
        plot(u(n).pearson_corr_mean(k), u(n).mean_abs_diff_mean(k),'k.')
        if sum(u(n).pearson_corr_mean(k)==mean_outlier_PCC)>0
            plot(u(n).pearson_corr_mean(k),u(n).mean_abs_diff_mean(k),'b+')
            fprintf('%s with filename %s is a PCC outlier (%g) with %d files\n',u(n).name, u(n).filename(9:end), u(n).pearson_corr_mean(k), u(n).num_files(k))
            fprintf(fid,'%s with filename %s is a PCC outlier (%g) with %d files\n',u(n).name, u(n).filename(9:end), u(n).pearson_corr_mean(k), u(n).num_files(k));
        end
        if sum(u(n).mean_abs_diff_mean(k)==mean_outlier_MAD)>0
            plot(u(n).pearson_corr_mean(k),u(n).mean_abs_diff_mean(k),'rx')
            fprintf('%s with filename %s is a MAD outlier (%g) with %d files\n',u(n).name, u(n).filename(9:end), u(n).mean_abs_diff_mean(k), u(n).num_files(k))
            fprintf(fid,'%s with filename %s is a MAD outlier (%g) with %d files\n',u(n).name, u(n).filename(9:end), u(n).mean_abs_diff_mean(k), u(n).num_files(k));

        end
    end
end
hold off
title('PCC vs MAD for Raw MeanOS')
% title('MD vs MAD vs STD-AD to Mean (MAD and STD-AD 1st outlier sets) Removed Pre-Normalisation')
% title('MD vs MAD vs STD-AD to Mean (MAD and STD-AD 2nd outlier sets) Removed Pre-Normalisation')
xlabel('$\rho$','Interpreter','latex')
ylabel('$\bar{X}_s$','interpreter','latex','Rotation',0)
axis([0.8*min([u.pearson_corr_mean]) 1.05*max([u.pearson_corr_mean]) ...
      0 1.05*max([u.mean_abs_diff_mean])])
text(0.2,.2,'MAD Outliers','Color','r')
text(0.2,.1,'PCC Outliers','Color','b')

set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');

print('Plots/PDF/Outliers_Raw_Grubbs', '-dpdf');
print('Plots/EPSC/Outliers_Raw_Grubbs', '-depsc');
print('Plots/PNG/Outliers_Raw_Grubbs', '-dpng');



%Plot the normalised values with the original outliers

fprintf('Post-Normalisation\n')
fprintf('Mean STD of Normalised File Ratings = %g\n',mean([a.std_MOS_norm]));
session_MAD_norm = [u.mean_abs_diff_norm_mean];
TF = isoutlier(session_MAD_norm,'grubbs');
mean_outlier_MAD_norm = session_MAD_norm(TF);

% session_STD_norm = [u.std_abs_diff_norm_mean];
% TF = isoutlier(session_STD_norm);
% mean_outlier_STD_norm = session_STD_norm(TF);
%
% session_MD_norm = [u.mean_diff_norm];
% TF = isoutlier(session_MD_norm);
% mean_outlier_MD_norm = session_MD_norm(TF);

session_PCC = [u.pearson_corr_MeanOS_norm];
TF = isoutlier(session_PCC,'grubbs');
mean_norm_outlier_PCC = session_PCC(TF);


figure('Position',[1680-500 200 500 250])
hold on
for n = 1:length(u)
    for k = 1:length(u(n).mean_diff)
        plot(u(n).pearson_corr_MeanOS_norm(k),u(n).mean_abs_diff_norm_mean(k),'k.')
        if sum(u(n).pearson_corr_MeanOS_norm(k)==mean_norm_outlier_PCC)>0
            plot(u(n).pearson_corr_MeanOS_norm(k),u(n).mean_abs_diff_norm_mean(k),'b+')
            %             fprintf('%s with key %s is a MD outlier with %d files\n',u(n).name, u(n).key, u(n).num_files(k))
        end
        if sum(u(n).mean_abs_diff_mean(k)==mean_outlier_MAD)>0
            plot(u(n).pearson_corr_MeanOS_norm(k),u(n).mean_abs_diff_norm_mean(k),'rx')
            %             fprintf('%s with key %s is a MAD outlier with %d files\n',u(n).name, u(n).key, u(n).num_files(k))
        end
    end
end
hold off
title('PCC vs MAD for Norm. MeanOS (Raw outliers)')
% title('MD vs MAD vs STD-AD to Mean (MAD and STD-AD 1st outlier sets) Removed Post-Normalisation (Pre-norm outliers marked)')
% title('MD vs MAD vs STD-AD to Mean (MAD and STD-AD 2nd outlier sets) Removed Post-Normalisation (Pre-norm outliers marked)')

xlabel('$\rho$','Interpreter','latex')
ylabel('$\bar{X}_s$','interpreter','latex','Rotation',0)

axis([0.8*min([u.pearson_corr_MeanOS_norm]) 1.05*max([u.pearson_corr_MeanOS_norm]) ...
      0 1.05*max([u.mean_abs_diff_norm_mean]) ])
text(0.2,0.2,'MAD Outliers','Color','r')
text(0.2,0.1,'PCC Outliers','Color','b')

set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');

print('Plots/PDF/Outliers_Raw_Norm_Results_Grubbs', '-dpdf');
print('Plots/EPSC/Outliers_Raw_Norm_Results_Grubbs', '-depsc');
print('Plots/PNG/Outliers_Raw_Norm_Results_Grubbs', '-dpng');




% POST NORMALISATION COMPARISON


fprintf(fid,'\nNorm Outliers\n');
figure('Position',[1680-500 200 500 250])
hold on
for n = 1:length(u)
    for k = 1:length(u(n).mean_diff_norm)
        plot(u(n).pearson_corr_MeanOS_norm(k),u(n).mean_abs_diff_norm_mean(k),'k.')
        if sum(u(n).pearson_corr_MeanOS_norm(k)==mean_norm_outlier_PCC)>0
            plot(u(n).pearson_corr_MeanOS_norm(k),u(n).mean_abs_diff_norm_mean(k),'b+')
            fprintf('%s with filename %s is a PCC outlier (%g) with %d files\n', u(n).name, u(n).filename(12:end), u(n).pearson_corr_MeanOS_norm(k), u(n).num_files(k))
            fprintf(fid,'%s with filename %s is a PCC outlier (%g) with %d files\n', u(n).name, u(n).filename(12:end), u(n).pearson_corr_MeanOS_norm(k), u(n).num_files(k));
        end
        if sum(u(n).mean_abs_diff_norm_mean(k)==mean_outlier_MAD_norm)>0
            plot(u(n).pearson_corr_MeanOS_norm(k),u(n).mean_abs_diff_norm_mean(k),'rx')
            fprintf('%s with filename %s is a MAD outlier (%g) with %d files\n', u(n).name,u(n).filename(12:end), u(n).mean_abs_diff_norm_mean(k), u(n).num_files(k))
            fprintf(fid,'%s with filename %s is a MAD outlier (%g) with %d files\n', u(n).name,u(n).filename(12:end), u(n).mean_abs_diff_norm_mean(k), u(n).num_files(k));
        end
    end
end
hold off

title('PCC vs MAD for Norm. MeanOS')
% title('MD vs MAD vs STD-AD to Mean (MAD and STD-AD 1st outlier sets) Removed Post-Normalisation')
% title('MD vs MAD vs STD-AD to Mean (MAD and STD-AD 2nd outlier sets) Removed Post-Normalisation')
xlabel('$\rho$','Interpreter','latex')
ylabel('$\bar{X}_s$','interpreter','latex','Rotation',0)
axis([0.8*min([u.pearson_corr_MeanOS_norm]) 1.05*max([u.pearson_corr_MeanOS_norm]) ...
      0 1.05*max([u.mean_abs_diff_norm_mean])])
text(0.2,0.2,'MAD Outliers','Color','r')
text(0.2,0.1,'PCC Outliers','Color','b')

set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');


print('Plots/PDF/Outliers_Norm_Grubbs', '-dpdf');
print('Plots/EPSC/Outliers_Norm_Grubbs', '-depsc');
print('Plots/PNG/Outliers_Norm_Grubbs', '-dpng');

fclose(fid);


%% -------------------------  MEDIAN OUTLIERS RMSE RESULTS TO REMOVE -------------------------

%Calculate the outlier values

fprintf('Pre-Normalisation\n')
fprintf('Mean STD of File Ratings = %g\n',mean([a.std_MOS]));
fid = fopen('log_Anon.txt','a');
fprintf(fid,'\n RMSE Median Outliers\n');
fprintf(fid,'Raw Outliers\n');
session_RMSE = [u.RMSE];
% TF = isoutlier(session_MAD,'mean');
TF = isoutlier(session_RMSE,'median');
% TF = isoutlier(session_MAD);
mean_outlier_RMSE = session_RMSE(TF);

% session_STD = [u.std_abs_diff_mean];
% % TF = isoutlier(session_STD,'mean');
% % TF = isoutlier(session_STD,'grubbs');
% TF = isoutlier(session_STD);
% mean_outlier_STD = session_STD(TF);
%
% session_MD = [u.mean_diff];
% % TF = isoutlier(session_MD,'mean');
% % TF = isoutlier(session_MD,'grubbs');
% TF = isoutlier(session_MD);
% mean_outlier_MD = session_MD(TF);

session_PCC = [u.pearson_corr_mean];
% TF = isoutlier(session_PCC,'mean');
TF = isoutlier(session_PCC,'median');
% TF = isoutlier(session_PCC);
mean_outlier_PCC = session_PCC(TF);

figure('Position',[0 0 500 250])
hold on
for n = 1:length(u)
    for k = 1:length(u(n).mean_diff)
        plot(u(n).pearson_corr_mean(k), u(n).RMSE(k),'k.')
        if sum(u(n).pearson_corr_mean(k)==mean_outlier_PCC)>0
            plot(u(n).pearson_corr_mean(k),u(n).RMSE(k),'b+')
            fprintf('%s with filename %s is a PCC outlier (%g) with %d files\n',u(n).name, u(n).filename(9:end), u(n).pearson_corr_mean(k), u(n).num_files(k))
            fprintf(fid,'%s with filename %s is a PCC outlier (%g) with %d files\n',u(n).name, u(n).filename(9:end), u(n).pearson_corr_mean(k), u(n).num_files(k));
        end
        if sum(u(n).RMSE(k)==mean_outlier_RMSE)>0
            plot(u(n).pearson_corr_mean(k),u(n).RMSE(k),'rx')
            fprintf('%s with filename %s is an RMSE outlier (%g) with %d files\n',u(n).name, u(n).filename(9:end), u(n).RMSE(k), u(n).num_files(k))
            fprintf(fid,'%s with filename %s is an RMSE outlier (%g) with %d files\n',u(n).name, u(n).filename(9:end), u(n).RMSE(k), u(n).num_files(k));

        end
    end
end
hold off
title('PCC vs RMSE for Raw MeanOS')
% title('MD vs MAD vs STD-AD to Mean (MAD and STD-AD 1st outlier sets) Removed Pre-Normalisation')
% title('MD vs MAD vs STD-AD to Mean (MAD and STD-AD 2nd outlier sets) Removed Pre-Normalisation')
xlabel('$\rho$','Interpreter','latex')
ylabel('$\mathcal{L}$','interpreter','latex','Rotation',0)
axis([0.8*min([u.pearson_corr_mean]) 1.05*max([u.pearson_corr_mean]) ...
      0 1.05*max([u.RMSE])])
text(0.2,.2,'RMSE Outliers','Color','r')
text(0.2,.1,'PCC Outliers','Color','b')

set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');

print('Plots/PDF/Outliers_Raw_RMSE_Median', '-dpdf');
print('Plots/EPSC/Outliers_Raw_RMSE_Median', '-depsc');
print('Plots/PNG/Outliers_Raw_RMSE_Median', '-dpng');



%Plot the normalised values with the original outliers

fprintf('Post-Normalisation\n')
fprintf('Mean STD of Normalised File Ratings = %g\n',mean([a.std_MOS]));
session_RMSE_norm = [u.RMSE_norm];
TF = isoutlier(session_RMSE_norm,'median');
mean_outlier_RMSE_norm = session_RMSE_norm(TF);

% session_STD_norm = [u.std_abs_diff_norm_mean];
% TF = isoutlier(session_STD_norm);
% mean_outlier_STD_norm = session_STD_norm(TF);
%
% session_MD_norm = [u.mean_diff_norm];
% TF = isoutlier(session_MD_norm);
% mean_outlier_MD_norm = session_MD_norm(TF);

session_PCC = [u.pearson_corr_MeanOS_norm];
TF = isoutlier(session_PCC,'median');
mean_norm_outlier_PCC = session_PCC(TF);


figure('Position',[0 0 500 250])
hold on
for n = 1:length(u)
    for k = 1:length(u(n).RMSE_norm)
        plot(u(n).pearson_corr_MeanOS_norm(k),u(n).RMSE_norm(k),'k.')
        if sum(u(n).pearson_corr_MeanOS_norm(k)==mean_norm_outlier_PCC)>0
            plot(u(n).pearson_corr_MeanOS_norm(k),u(n).RMSE_norm(k),'b+')
            %             fprintf('%s with key %s is a MD outlier with %d files\n',u(n).name, u(n).key, u(n).num_files(k))
        end
        if sum(u(n).RMSE(k)==mean_outlier_RMSE)>0
            plot(u(n).pearson_corr_MeanOS_norm(k),u(n).RMSE_norm(k),'rx')
            %             fprintf('%s with key %s is a MAD outlier with %d files\n',u(n).name, u(n).key, u(n).num_files(k))
        end
    end
end
hold off
title('PCC vs RMSE for Norm. MeanOS (Raw outliers)')
% title('MD vs MAD vs STD-AD to Mean (MAD and STD-AD 1st outlier sets) Removed Post-Normalisation (Pre-norm outliers marked)')
% title('MD vs MAD vs STD-AD to Mean (MAD and STD-AD 2nd outlier sets) Removed Post-Normalisation (Pre-norm outliers marked)')

xlabel('$\rho$','Interpreter','latex')
ylabel('$\mathcal{L}$','interpreter','latex','Rotation',0)

axis([0.8*min([u.pearson_corr_MeanOS_norm]) 1.05*max([u.pearson_corr_MeanOS_norm]) ...
      0 1.05*max([u.RMSE_norm]) ])
text(0.2,0.2,'RMSE Outliers','Color','r')
text(0.2,0.1,'PCC Outliers','Color','b')

set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');

print('Plots/PDF/Outliers_Raw_Norm_Results_RMSE_Median', '-dpdf');
print('Plots/EPSC/Outliers_Raw_Norm_Results_RMSE_Median', '-depsc');
print('Plots/PNG/Outliers_Raw_Norm_Results_RMSE_Median', '-dpng');




% POST NORMALISATION COMPARISON


fprintf(fid,'\nNorm Outliers\n');
figure('Position',[0 0 500 250])
hold on
for n = 1:length(u)
    for k = 1:length(u(n).RMSE_norm)
        plot(u(n).pearson_corr_MeanOS_norm(k),u(n).RMSE_norm(k),'k.')
        if sum(u(n).pearson_corr_MeanOS_norm(k)==mean_norm_outlier_PCC)>0
            plot(u(n).pearson_corr_MeanOS_norm(k),u(n).RMSE_norm(k),'b+')
            fprintf('%s with filename %s is a PCC outlier (%g) with %d files\n', u(n).name, u(n).filename(12:end), u(n).pearson_corr_MeanOS_norm(k), u(n).num_files(k))
            fprintf(fid,'%s with filename %s is a PCC outlier (%g) with %d files\n', u(n).name, u(n).filename(12:end), u(n).pearson_corr_MeanOS_norm(k), u(n).num_files(k));
        end
        if sum(u(n).RMSE_norm(k)==mean_outlier_RMSE_norm)>0
            plot(u(n).pearson_corr_MeanOS_norm(k),u(n).RMSE_norm(k),'rx')
            fprintf('%s with filename %s is an RMSE outlier (%g) with %d files\n', u(n).name,u(n).filename(12:end), u(n).RMSE_norm(k), u(n).num_files(k))
            fprintf(fid,'%s with filename %s is an RMSE outlier (%g) with %d files\n', u(n).name,u(n).filename(12:end), u(n).RMSE_norm(k), u(n).num_files(k));
        end
    end
end
hold off

title('PCC vs RMSE for Norm. MeanOS')
% title('MD vs MAD vs STD-AD to Mean (MAD and STD-AD 1st outlier sets) Removed Post-Normalisation')
% title('MD vs MAD vs STD-AD to Mean (MAD and STD-AD 2nd outlier sets) Removed Post-Normalisation')
xlabel('$\rho$','Interpreter','latex')
ylabel('$\mathcal{L}$','interpreter','latex','Rotation',0)
axis([0.8*min([u.pearson_corr_MeanOS_norm]) 1.05*max([u.pearson_corr_MeanOS_norm]) ...
      0 1.05*max([u.RMSE_norm])])
text(0.2,0.2,'RMSE Outliers','Color','r')
text(0.2,0.1,'PCC Outliers','Color','b')

set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');


print('Plots/PDF/Outliers_Norm_RMSE_Median', '-dpdf');
print('Plots/EPSC/Outliers_Norm_RMSE_Median', '-depsc');
print('Plots/PNG/Outliers_Norm_RMSE_Median', '-dpng');

fclose(fid);

%% -------------------------  Grubbs OUTLIERS RMSE RESULTS TO REMOVE -------------------------

%Calculate the outlier values

fprintf('Pre-Normalisation\n')
fprintf('Mean STD of File Ratings = %g\n',mean([a.std_MOS]));
fid = fopen('log_Anon.txt','a');
fprintf(fid,'\nRMSE Grubbs Outliers\n');
fprintf(fid,'Raw Outliers\n');
session_RMSE = [u.RMSE];
% TF = isoutlier(session_MAD,'mean');
TF = isoutlier(session_RMSE,'grubbs');
% TF = isoutlier(session_MAD);
mean_outlier_RMSE = session_RMSE(TF);

% session_STD = [u.std_abs_diff_mean];
% % TF = isoutlier(session_STD,'mean');
% % TF = isoutlier(session_STD,'grubbs');
% TF = isoutlier(session_STD);
% mean_outlier_STD = session_STD(TF);
%
% session_MD = [u.mean_diff];
% % TF = isoutlier(session_MD,'mean');
% % TF = isoutlier(session_MD,'grubbs');
% TF = isoutlier(session_MD);
% mean_outlier_MD = session_MD(TF);

session_PCC = [u.pearson_corr_mean];
% TF = isoutlier(session_PCC,'mean');
TF = isoutlier(session_PCC,'grubbs');
% TF = isoutlier(session_PCC);
mean_outlier_PCC = session_PCC(TF);

figure('Position',[0 0 500 250])
hold on
for n = 1:length(u)
    for k = 1:length(u(n).mean_diff)
        plot(u(n).pearson_corr_mean(k), u(n).RMSE(k),'k.')
        if sum(u(n).pearson_corr_mean(k)==mean_outlier_PCC)>0
            plot(u(n).pearson_corr_mean(k),u(n).RMSE(k),'b+')
            fprintf('%s with filename %s is a PCC outlier (%g) with %d files\n',u(n).name, u(n).filename(9:end), u(n).pearson_corr_mean(k), u(n).num_files(k))
            fprintf(fid,'%s with filename %s is a PCC outlier (%g) with %d files\n',u(n).name, u(n).filename(9:end), u(n).pearson_corr_mean(k), u(n).num_files(k));
        end
        if sum(u(n).RMSE(k)==mean_outlier_RMSE)>0
            plot(u(n).pearson_corr_mean(k),u(n).RMSE(k),'rx')
            fprintf('%s with filename %s is an RMSE outlier (%g) with %d files\n',u(n).name, u(n).filename(9:end), u(n).RMSE(k), u(n).num_files(k))
            fprintf(fid,'%s with filename %s is an RMSE outlier (%g) with %d files\n',u(n).name, u(n).filename(9:end), u(n).RMSE(k), u(n).num_files(k));

        end
    end
end
hold off
title('PCC vs RMSE for Raw MeanOS')
% title('MD vs MAD vs STD-AD to Mean (MAD and STD-AD 1st outlier sets) Removed Pre-Normalisation')
% title('MD vs MAD vs STD-AD to Mean (MAD and STD-AD 2nd outlier sets) Removed Pre-Normalisation')
xlabel('$\rho$','Interpreter','latex')
ylabel('$\mathcal{L}$','interpreter','latex','Rotation',0)
axis([0.8*min([u.pearson_corr_mean]) 1.05*max([u.pearson_corr_mean]) ...
      0 1.05*max([u.RMSE])])
text(0.2,.2,'RMSE Outliers','Color','r')
text(0.2,.1,'PCC Outliers','Color','b')

set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');

print('Plots/PDF/Outliers_Raw_RMSE_grubbs', '-dpdf');
print('Plots/EPSC/Outliers_Raw_RMSE_grubbs', '-depsc');
print('Plots/PNG/Outliers_Raw_RMSE_grubbs', '-dpng');



%Plot the normalised values with the original outliers

fprintf('Post-Normalisation\n')
fprintf('Mean STD of Normalised File Ratings = %g\n',mean([a.std_MOS]));
session_RMSE_norm = [u.RMSE_norm];
TF = isoutlier(session_RMSE_norm,'grubbs');
mean_outlier_RMSE_norm = session_RMSE_norm(TF);

% session_STD_norm = [u.std_abs_diff_norm_mean];
% TF = isoutlier(session_STD_norm);
% mean_outlier_STD_norm = session_STD_norm(TF);
%
% session_MD_norm = [u.mean_diff_norm];
% TF = isoutlier(session_MD_norm);
% mean_outlier_MD_norm = session_MD_norm(TF);

session_PCC = [u.pearson_corr_MeanOS_norm];
TF = isoutlier(session_PCC,'grubbs');
mean_norm_outlier_PCC = session_PCC(TF);


figure('Position',[0 0 500 250])
hold on
for n = 1:length(u)
    for k = 1:length(u(n).RMSE_norm)
        plot(u(n).pearson_corr_MeanOS_norm(k),u(n).RMSE_norm(k),'k.')
        if sum(u(n).pearson_corr_MeanOS_norm(k)==mean_norm_outlier_PCC)>0
            plot(u(n).pearson_corr_MeanOS_norm(k),u(n).RMSE_norm(k),'b+')
            %             fprintf('%s with key %s is a MD outlier with %d files\n',u(n).name, u(n).key, u(n).num_files(k))
        end
        if sum(u(n).RMSE(k)==mean_outlier_RMSE)>0
            plot(u(n).pearson_corr_MeanOS_norm(k),u(n).RMSE_norm(k),'rx')
            %             fprintf('%s with key %s is a MAD outlier with %d files\n',u(n).name, u(n).key, u(n).num_files(k))
        end
    end
end
hold off
title('PCC vs RMSE for Norm. MeanOS (Raw outliers)')
% title('MD vs MAD vs STD-AD to Mean (MAD and STD-AD 1st outlier sets) Removed Post-Normalisation (Pre-norm outliers marked)')
% title('MD vs MAD vs STD-AD to Mean (MAD and STD-AD 2nd outlier sets) Removed Post-Normalisation (Pre-norm outliers marked)')

xlabel('$\rho$','Interpreter','latex')
ylabel('$\mathcal{L}$','interpreter','latex','Rotation',0)

axis([0.8*min([u.pearson_corr_MeanOS_norm]) 1.05*max([u.pearson_corr_MeanOS_norm]) ...
      0 1.05*max([u.RMSE_norm]) ])
text(0.2,0.2,'RMSE Outliers','Color','r')
text(0.2,0.1,'PCC Outliers','Color','b')

set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');

print('Plots/PDF/Outliers_Raw_Norm_Results_RMSE_grubbs', '-dpdf');
print('Plots/EPSC/Outliers_Raw_Norm_Results_RMSE_grubbs', '-depsc');
print('Plots/PNG/Outliers_Raw_Norm_Results_RMSE_grubbs', '-dpng');




% POST NORMALISATION COMPARISON


fprintf(fid,'\nNorm Outliers\n');
figure('Position',[0 0 500 250])
hold on
for n = 1:length(u)
    for k = 1:length(u(n).RMSE_norm)
        plot(u(n).pearson_corr_MeanOS_norm(k),u(n).RMSE_norm(k),'k.')
        if sum(u(n).pearson_corr_MeanOS_norm(k)==mean_norm_outlier_PCC)>0
            plot(u(n).pearson_corr_MeanOS_norm(k),u(n).RMSE_norm(k),'b+')
            fprintf('%s with filename %s is a PCC outlier (%g) with %d files\n', u(n).name, u(n).filename(12:end), u(n).pearson_corr_MeanOS_norm(k), u(n).num_files(k))
            fprintf(fid,'%s with filename %s is a PCC outlier (%g) with %d files\n', u(n).name, u(n).filename(12:end), u(n).pearson_corr_MeanOS_norm(k), u(n).num_files(k));
        end
        if sum(u(n).RMSE_norm(k)==mean_outlier_RMSE_norm)>0
            plot(u(n).pearson_corr_MeanOS_norm(k),u(n).RMSE_norm(k),'rx')
            fprintf('%s with filename %s is an RMSE outlier (%g) with %d files\n', u(n).name,u(n).filename(12:end), u(n).RMSE_norm(k), u(n).num_files(k))
            fprintf(fid,'%s with filename %s is an RMSE outlier (%g) with %d files\n', u(n).name,u(n).filename(12:end), u(n).RMSE_norm(k), u(n).num_files(k));
        end
    end
end
hold off

title('PCC vs RMSE for Norm. MeanOS')
% title('MD vs MAD vs STD-AD to Mean (MAD and STD-AD 1st outlier sets) Removed Post-Normalisation')
% title('MD vs MAD vs STD-AD to Mean (MAD and STD-AD 2nd outlier sets) Removed Post-Normalisation')
xlabel('$\rho$','Interpreter','latex')
ylabel('$\mathcal{L}$','interpreter','latex','Rotation',0)
axis([0.8*min([u.pearson_corr_MeanOS_norm]) 1.05*max([u.pearson_corr_MeanOS_norm]) ...
      0 1.05*max([u.RMSE_norm])])
text(0.2,0.2,'RMSE Outliers','Color','r')
text(0.2,0.1,'PCC Outliers','Color','b')

set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');


print('Plots/PDF/Outliers_Norm_RMSE_grubbs', '-dpdf');
print('Plots/EPSC/Outliers_Norm_RMSE_grubbs', '-depsc');
print('Plots/PNG/Outliers_Norm_RMSE_grubbs', '-dpng');

fclose(fid);


%% ----------------Convert old response to STD plot to histogram-------------


% open 'OLD/Anova Plots/Results_v6/Responses v STD.fig'
% D=get(gca,'Children');
% XData=get(D,'XData');
% YData=get(D,'YData');
%
% %only take the values for files that have more than 1 rating
% x_val = XData(XData~=1);
% y_val = YData(XData~=1);
%
%
% h = histogram2(x_val,y_val,[50 50],'FaceColor','flat');
% h.ShowEmptyBins = 'On';
% h.DisplayStyle = 'tile';
%
% view(2)
% colormap(flipud(gray));
% c = colorbar;
% c.Label.String = 'Count';
% title_text = sprintf('Standard Deviation of Opinion Scores for Number of File Ratings');




%% -------------------- Compare MATLAB to WAET results -------------------
fprintf('Compare Lab and Remote Testing\n')
Offline_MMADs = [u(1:15).mean_abs_diff_norm_mean];
Online_MMADs = [u(16:end).mean_abs_diff_norm_mean];

bins = 50;
EDGES = linspace(min([Offline_MMADs Online_MMADs]), max([Offline_MMADs Online_MMADs]),bins);
EDGES_PLOT = (EDGES(1:end-1)+EDGES(2:end))/2;

[Offline_MMADs_Count, ~] = histcounts(Offline_MMADs,EDGES,'Normalization','probability');
[Online_MMADs_Count, ~] = histcounts(Online_MMADs,EDGES,'Normalization','probability');

figure('Position',[1680-500 200 500 250])
plot(EDGES_PLOT, Online_MMADs_Count,'k-')
hold on
plot(EDGES_PLOT, Offline_MMADs_Count,'k:')
hold off
% T=title('MAD for Online vs Offline Testing');
xlabel('$\bar{X}_s$','interpreter','latex')
ylabel('Normalised Probability')
axis([min(EDGES),max(EDGES),0,1.1*max([Offline_MMADs_Count Online_MMADs_Count])])
legend('Remote','Laboratory','Location','northeast');
set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');

print('Plots/PDF/Offline_Online_Line', '-dpdf');
print('Plots/EPSC/Offline_Online_Line', '-depsc');
print('Plots/PNG/Offline_Online_Line', '-dpng');

% RMSE version
Offline_MMADs = [u(1:65).RMSE_norm];
Online_MMADs = [u(66:end).RMSE_norm];

bins = 50;
EDGES = linspace(min([Offline_MMADs Online_MMADs]), max([Offline_MMADs Online_MMADs]),bins);
EDGES_PLOT = (EDGES(1:end-1)+EDGES(2:end))/2;

[Offline_MMADs_Count, ~] = histcounts(Offline_MMADs,EDGES,'Normalization','probability');
[Online_MMADs_Count, ~] = histcounts(Online_MMADs,EDGES,'Normalization','probability');

figure('Position',[1680-500 200 500 250])
plot(EDGES_PLOT, Online_MMADs_Count,'k-')
hold on
plot(EDGES_PLOT, Offline_MMADs_Count,'k:')
hold off
% T=title('MAD for Online vs Offline Testing');
xlabel('$\mathcal{L}$','interpreter','latex')
ylabel('Normalised Probability')
axis([min(EDGES),max(EDGES),0,1.1*max([Offline_MMADs_Count Online_MMADs_Count])])
legend('Remote','Laboratory','Location','northeast');
set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');

print('Plots/PDF/RMSE_Offline_Online_Line', '-dpdf');
print('Plots/EPSC/RMSE_Offline_Online_Line', '-depsc');
print('Plots/PNG/RMSE_Offline_Online_Line', '-dpng');

g1=ones(1,size(Offline_MMADs,2));
g2=2*ones(1,size(Online_MMADs,2));
x=[Offline_MMADs(:);Online_MMADs(:)]' ;
g=[g1,g2];
[p1,ANOVATAB,STATS]=anova1(x,g);
[H,P]=ttest2(Offline_MMADs,Online_MMADs);
fid = fopen('log_Anon.txt','a');
fprintf(fid,'Offline vs Online\n');
fprintf(fid,'Reject Null hypothesis of different means at alpha=0.05: %d\n',H);
fprintf(fid,'p-value: %g\n',P);
fclose(fid);

print('Plots/PDF/Offline_Online_Boxplot', '-dpdf');
print('Plots/EPSC/Offline_Online_Boxplot', '-depsc');
print('Plots/PNG/Offline_Online_Boxplot', '-dpng');


%Compare the duration of sessions to number of files in each session.
fprintf('Session Duration\n')
figure('Position',[1680-500 200 500 250])
h = histogram2([u.total_time_min],[u.num_files],[100 100],'FaceColor','flat');
h.ShowEmptyBins = 'Off';
h.DisplayStyle = 'tile';
h.EdgeAlpha = 0;

ax = gca;
ax.GridColor = [0.4 0.4 0.4];
ax.GridLineStyle = '--';
ax.GridAlpha = 0.5;
ax.XGrid = 'off';
ax.YGrid = 'on';
ax.Layer = 'top';
view(2)
% colormap(flipud(gray));
c = colorbar;
c.Label.String = 'Count';
xlabel('Total Time (Minutes)')
ylabel('Files in Session')

set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');

print('Plots/PDF/Session_Time_Offline_Online', '-dpdf');
print('Plots/EPSC/Session_Time_Offline_Online', '-depsc');
print('Plots/PNG/Session_Time_Offline_Online', '-dpng');

fprintf('Mean time per file for Offline Sets = %g\n',mean([u(1:15).total_time_sec]./[u(1:15).num_files]))
fprintf('Mean time per file for Online Sets = %g\n',mean([u(16:end).total_time_sec]./[u(16:end).num_files]))




%Compare MeanOS to MedianOS
fprintf('Compare MeanOS and MedianOS\n')
figure('Position',[1680-500 200 500 250])
h = histogram2([a.mean_MOS_norm],[a.median_MOS_norm],[100 100],'FaceColor','flat');
h.ShowEmptyBins = 'Off';
h.DisplayStyle = 'tile';

ax = gca;
ax.GridColor = [0.4 0.4 0.4];
ax.GridLineStyle = '--';
ax.GridAlpha = 0.5;
ax.XGrid = 'off';
ax.YGrid = 'on';
ax.Layer = 'top';
view(2)
% colormap(gray);
c = colorbar;
c.Label.String = 'Count';
title('Normalised MeanOS vs Normalised MedianOS')
xlabel('Mean Opinion Score')
ylabel('Median Opinion Score')

set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');

print('Plots/PDF/Mean_vs_Median_Normalised', '-dpdf');
print('Plots/EPSC/Mean_vs_Median_Normalised', '-depsc');
print('Plots/PNG/Mean_vs_Median_Normalised', '-dpng');

figure('Position',[1680-500 200 500 250])
h = histogram2([a.mean_MOS],[a.median_MOS],[100 100],'FaceColor','flat');
h.ShowEmptyBins = 'Off';
h.DisplayStyle = 'tile';

ax = gca;
ax.GridColor = [0.4 0.4 0.4];
ax.GridLineStyle = '--';
ax.GridAlpha = 0.5;
ax.XGrid = 'off';
ax.YGrid = 'on';
ax.Layer = 'top';
view(2)
% colormap(gray);
c = colorbar;
c.Label.String = 'Count';
title('MeanOS vs MedianOS')
xlabel('Mean Opinion Score')
ylabel('Median Opinion Score')

set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');

print('Plots/PDF/Mean_vs_Median_Pre_Normalisation', '-dpdf');
print('Plots/EPSC/Mean_vs_Median_Pre_Normalisation', '-depsc');
print('Plots/PNG/Mean_vs_Median_Pre_Normalisation', '-dpng');


%% ------- Histogram of Correlation between Subjective and MeanOS ----------
fprintf('Correlation Subjective to MeanOS\n')
fprintf('Plotting Histogram of Correlation of results to MeanOS\n')
figure('Position',[1680-500 200 500 250])
h = histogram([u.pearson_corr_mean],'BinWidth',0.05);
h.EdgeColor = 'k';
h.FaceColor = [ 1,1,1];
% title('Mean Absolute Difference Per Session')
xlabel('$\rho$','Interpreter','latex')
ylabel('Count')

set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');

print('Plots/PDF/Correlation_with_MeanOS', '-dpdf');
print('Plots/EPSC/Correlation_with_MeanOS', '-depsc');
print('Plots/PNG/Correlation_with_MeanOS', '-dpng');

fprintf('Plotting Histogram of Correlation of results to MedianOS\n')
figure('Position',[1680-500 200 500 250])
h = histogram([u.pearson_corr_median],'BinWidth',0.05);
h.EdgeColor = 'k';
h.FaceColor = [ 1,1,1];
% title('Mean Absolute Difference Per Session')
xlabel('$\rho$','Interpreter','latex')
ylabel('Count')

set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');

print('Plots/PDF/Correlation_with_MedianOS', '-dpdf');
print('Plots/EPSC/Correlation_with_MedianOS', '-depsc');
print('Plots/PNG/Correlation_with_MedianOS', '-dpng');

% -------------------------Once normalised-------------------------
fprintf('Plotting Histogram of Correlation of results to MeanOS\n')
figure('Position',[1680-500 200 500 250])
h = histogram([u.pearson_corr_MeanOS_norm],'BinWidth',0.05);
h.EdgeColor = 'k';
h.FaceColor = [ 1,1,1];
% title('Mean Absolute Difference Per Session')
xlabel('$\rho$','Interpreter','latex')
ylabel('Count')

set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');

print('Plots/PDF/Correlation_with_Norm_MeanOS', '-dpdf');
print('Plots/EPSC/Correlation_with_Norm_MeanOS', '-depsc');
print('Plots/PNG/Correlation_with_Norm_MeanOS', '-dpng');

fprintf('Plotting Histogram of Correlation of results to MedianOS\n')
figure('Position',[1680-500 200 500 250])
h = histogram([u.pearson_corr_MedianOS_norm],'BinWidth',0.05);
h.EdgeColor = 'k';
h.FaceColor = [ 1,1,1];
% title('Mean Absolute Difference Per Session')
xlabel('$\rho$','Interpreter','latex')
ylabel('Count')

set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');

print('Plots/PDF/Correlation_with_Norm_MedianOS', '-dpdf');
print('Plots/EPSC/Correlation_with_Norm_MedianOS', '-depsc');
print('Plots/PNG/Correlation_with_Norm_MedianOS', '-dpng');


%% ---------- Histogram2 of PCC with MAD ---------------
fprintf('PCC and MAD\n')
figure('Position',[1680-500 200 500 250])
h = histogram2([u.mean_abs_diff_mean],[u.pearson_corr_mean],'BinWidth',[0.025 0.025],'FaceColor','flat');
h.DisplayStyle = 'tile';
h.EdgeAlpha = 0;
view(2)
% colormap(gray);
c = colorbar;
c.Label.String = 'Count';
grid off
% axis([6, 18, 0, 1.1*max([u.mean_abs_diff_mean])])
% title('Number of Responses vs Std(Opinion Scores)')
xlabel('$\bar{X}_s$','Interpreter','latex')
ylabel('$\rho$','Interpreter','latex')

set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');

print('Plots/PDF/MAD_vs_PCC', '-dpdf');
print('Plots/EPSC/MAD_vs_PCC', '-depsc');
print('Plots/PNG/MAD_vs_PCC', '-dpng');





%% ----------------------- Print out stats for paper ----------------------
fprintf('Printing Out Stats\n')
fid = fopen('log_Anon.txt','a');
fprintf(fid,'The overall number of ratings given is %d\n', sum([a.num_responses]));
fprintf('Overall number of ratings without FuzzyPV, NMF and Elastique = %d\n',sum([a(1:5280).num_responses]));
fprintf('Overall number of ratings with FuzzyPV, NMF and Elastique = %d\n',sum([a.num_responses]));
fprintf('Revisit the next 2 lines after outliers have been removed.\n')
fprintf(fid,'The number of files removed due to outlier sessions = %d\n',42529-sum([a.num_responses]));


fprintf(fid,'Number of ratings in MATLAB = %d\n',sum([u(1:63).num_files])+sum([u(65:68).num_files]));

fprintf(fid,'Number of sessions = %d\n', length([u.num_files]));

fprintf(fid,'Manually count the number of individual participants\n');

expert_ratings = 0;
for n = 1:length(u)
    if u(n).expert > 0
        for k = 1:length(u(n).num_files)
            expert_ratings = expert_ratings + u(n).num_files(k);
        end
    end
end
fprintf(fid,'Number of expert ratings = %d\n',expert_ratings);
fprintf(fid,'Percentage of expert ratings = %g\n', 100*expert_ratings/sum([a.num_responses]));






fprintf(fid,'\n\n\nLatex output for file types\n');
input.data = zeros(6,4);
input.data(:,1) = mean(Music.type_mean);
input.data(:,2) = mean(Solo.type_mean);
input.data(:,3) = mean(Voice.type_mean);
% input.data(:,5) = mean(input.data(:,1:4),2);
% input.data(:,6) = sum([size(Complex.type_mean,1)*input.data(:,1) ,...
%                             size(Music.type_mean,1)*input.data(:,2) ,...
%                             size(Solo.type_mean,1)*input.data(:,3) ,...
%                             size(Voice.type_mean,1)*input.data(:,4)],2)/88;
input.data(:,4) = mean(anova2_data);
input.tablePositioning = 'ht';
input.tableColLabels = {'Music','Solo','Voice','Overall'};%, 'Weighted Overall','Overall RAW'};
input.tableRowLabels = {'PV','IPL','WSOLA','FESOLA','HPTSM','uTVS'};
input.dataFormat = {'%.3f'};
input.tableBorders = 1;
input.tableCaption = 'Means and Standard Deviation of Mean Opinion Scores';
input.tableLabel = 'MOS_Results';
latex_output = JASAlatexTable(input,fid);


fprintf(fid,'\nMusic file difficulty\n');
%Order of difficulty for Music files
music_file_means = mean(Music.type_mean,2);
% wsola_music_file_means = Music.type_mean(:,3);
[~,I] = sort(music_file_means);
for n = 1:length(I)
    fprintf(fid,'%g, %s\n',music_file_means(I(n)),filelist(I(n)).location);
end

fprintf(fid,'Solo file difficulty\n');
%Order of difficulty for Solo files
solo_file_means = mean(Solo.type_mean,2);
[~,I] = sort(solo_file_means);
for n = 1:length(I)
    fprintf(fid,'%g, %s\n',solo_file_means(I(n)),filelist(I(n)+47).location);
end
fprintf(fid,'Voice file difficulty\n');
%Order of difficulty for Voice files
voice_file_means = mean(Voice.type_mean,2);
% wsola_voice_file_means = Voice.type_mean(:,4);
[~,I] = sort(voice_file_means);
for n = 1:length(I)
    fprintf(fid,'%g, %s\n',voice_file_means(I(n)),filelist(I(n)+78).location);
end

fclose(fid);
beep
pause(1)
beep
pause(1)
beep
