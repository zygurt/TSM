function [processing_time] = Generate_Training_Features(match_method)
%[processing_time] = Generate_Features(match_method)
%   Calculate MOVs for Time Scaled signals.
%   Choose the method of matching signal lengths
%   match_method = 'Framing_Ref'
%   match_method = 'Framing_Test'
%   match_method = 'Interpolate_fd_up'
%   match_method = 'Interpolate_fd_down'
%   match_method = 'Interpolate_to_ref'
%   match_method = 'Interpolate_to_test'
if nargin <1
    match_method = 'Interpolate_to_test';
end

% close all
% clear all
% clc

global debug_var
debug_var = 0 ;

tic
addpath(genpath('Functions/')); %OMOQ functions
addpath(genpath('../Functions/')); %Additional Functions

addpath(genpath('../Subjective_Testing/Sets/'));
addpath(genpath('../Subjective_Testing/Source/'));

load('../Subjective_Testing/TSM_MOS_Scores.mat');
% load('../Subjective_Testing/TSM_MOS_Scores_29-Aug-2019.mat');
% load('../Subjective_Testing/TSM_MOS_Scores_22-Nov-2019.mat');
% load('../Subjective_Testing/TSM_MOS_Scores_06-Feb-2020.mat');

data_in = data;

log_name = sprintf('Logs/%s_Feature_log.txt',match_method);
f = fopen(log_name,'a');
c_date = clock;
fprintf(f,'Starting Processing at %s%s%s',num2str(c_date(1)),num2str(c_date(2),'%02d'),num2str(c_date(3),'%02d'));
fclose(f);
N = length(data_in);

OMOV = {'MeanOS', 'MedianOS', ...
    'MeanOS_RAW', 'MedianOS_RAW', ...
    'TSM', ...
    'WinModDiff1B', 'AvgModDiff1B', 'AvgModDiff2B', ...
    'RmsNoiseLoudB', ...
    'BandwidthRefB', 'BandwidthTestB', 'BandwidthTestB_new', ...
    'TotalNMRB', ...
    'RelDistFramesB', ...
    'MFPDB', 'ADBB', ...
    'EHSB', ...
    'RmsModDiffA', 'RmsNoiseLoudAsymA', 'AvgLinDistA', 'SegmentalNMRB', ...
    'DM', 'SER', ...
    'peak_delta', 'transient_ratio', 'hpsep_transient_ratio', ...
    'MPhNW', 'SPhNW', ...
    'MPhMW', 'SPhMW', ...
    'SSMAD','SSMD'};



MOVs = zeros(N,size(OMOV,2));


for n = 1: length(data_in)
    side_data(n).TSM = str2double(data_in(n).TSM)/100;
    side_data(n).MeanOS = data_in(n).MeanOS;
    side_data(n).MedianOS = data_in(n).MedianOS;
    % side_data(n).StdOS = data_in(n).std;
    side_data(n).MeanOS_RAW = data_in(n).MeanOS_RAW;
    side_data(n).MedianOS_RAW = data_in(n).MedianOS_RAW;
    % side_data(n).StdOS_RAW = data_in(n).std_RAW;
end

% % load('MOVs_20191121Framing.mat')
% % t = 5250:size(MOVs,1);
% % t = 346;
% % %
% for n = 1:size(data_in,2)
%     ref = [data_in(n).ref_loc data_in(n).ref_name];
%     test = [data_in(n).test_loc data_in(n).test_name];
%     fprintf('%d: %s: \n',n, test);
% 
% %     side_data.TSM = str2double(data_in(t(n)).TSM)/100;
% %     side_data.MeanOS = data_in(n).MeanOS;
% %     side_data.MedianOS = data_in(n).MedianOS;
% %     side_data.StdOS = data_in(n).std;
% %     side_data.MeanOS_RAW = data_in(n).MeanOS_RAW;
% %     side_data.MedianOS_RAW = data_in(n).MedianOS_RAW;
% %     side_data.StdOS_RAW = data_in(n).std_RAW;
% 
% 
%     [MOVs(n,:),~] = OMOQ(ref, test, side_data(n), match_method);
% %     [MOVs(t(n),:),~] = OMOQ('Audio/Ref/My_Song_9.wav', 'Audio/Test/My_Song_9_72bpm.wav', 0.72, 0, 0, 0, match_method);
% %     fprintf('%g\n',MOVs(t(n),5));
% 
% end
% 
% c_date = clock;
% sname = sprintf('TSMMOVs_%s%s%s%s.mat',num2str(c_date(1)),num2str(c_date(2),'%02d'),num2str(c_date(3),'%02d'),match_method);
% save(sname,'MOVs','OMOV');
% processing_time = toc;
% fprintf('Processing Complete.  Time taken = %.3f hours\n',processing_time/3600);
% fprintf(f, 'Processing Complete.  Time taken = %.3f hours\n',processing_time/3600);


%% ------  Parallel processing -------




parfor n = 1:250
    ref = [data_in(n).ref_loc data_in(n).ref_name];
    test = [data_in(n).test_loc data_in(n).test_name];
    %fprintf(f,'%d: %s\n',n,test);
    try
        a = load(['./Features/' match_method '/' data_in(n).test_name(1:end-4) '.mat']);
        MOVs(n,:) = [a.OMOV];
    catch
        [MOVs(n,:),~] = OMOQ(ref, test, side_data(n), match_method);
    end
end
c_date = clock;
sname = sprintf('MOVs_%s%s%s%s.mat',num2str(c_date(1)),num2str(c_date(2),'%02d'),num2str(c_date(3),'%02d'),match_method);
save(sname,'MOVs','OMOV');
fprintf('\n250 Completed\n');
f = fopen(log_name,'a');
fprintf(f,'\n250 Completed\n');
fclose(f);


parfor n = 251:500
    ref = [data_in(n).ref_loc data_in(n).ref_name];
    test = [data_in(n).test_loc data_in(n).test_name];
    %fprintf(f,'%d: %s\n',n,test);
    try
        a = load(['./Features/' match_method '/' data_in(n).test_name(1:end-4) '.mat']);
        MOVs(n,:) = [a.OMOV];
    catch
        [MOVs(n,:),~] = OMOQ(ref, test, side_data(n), match_method);
    end
end
c_date = clock;
sname = sprintf('MOVs_%s%s%s%s.mat',num2str(c_date(1)),num2str(c_date(2),'%02d'),num2str(c_date(3),'%02d'),match_method);
save(sname,'MOVs','OMOV');
fprintf('\n500 Completed\n');
f = fopen(log_name,'a');
fprintf(f,'\n500 Completed\n');
fclose(f);

parfor (n = 501:750)
    ref = [data_in(n).ref_loc data_in(n).ref_name];
    test = [data_in(n).test_loc data_in(n).test_name];
    %fprintf(f,'%d: %s\n',n,test);
    try
        a = load(['./Features/' match_method '/' data_in(n).test_name(1:end-4) '.mat']);
        MOVs(n,:) = [a.OMOV];
    catch
        [MOVs(n,:),~] = OMOQ(ref, test, side_data(n), match_method);
    end
end
c_date = clock;
sname = sprintf('MOVs_%s%s%s%s.mat',num2str(c_date(1)),num2str(c_date(2),'%02d'),num2str(c_date(3),'%02d'),match_method);
save(sname,'MOVs','OMOV');
fprintf('\n750 Completed\n');
f = fopen(log_name,'a');
fprintf(f,'\n750 Completed\n');
fclose(f);

parfor (n = 751:1000)
    ref = [data_in(n).ref_loc data_in(n).ref_name];
    test = [data_in(n).test_loc data_in(n).test_name];
    %fprintf(f,'%d: %s\n',n,test);
    try
        a = load(['./Features/' match_method '/' data_in(n).test_name(1:end-4) '.mat']);
        MOVs(n,:) = [a.OMOV];
    catch
        [MOVs(n,:),~] = OMOQ(ref, test, side_data(n), match_method);
    end
end
c_date = clock;
sname = sprintf('MOVs_%s%s%s%s.mat',num2str(c_date(1)),num2str(c_date(2),'%02d'),num2str(c_date(3),'%02d'),match_method);
save(sname,'MOVs','OMOV');
fprintf('\n1000 Completed\n');
f = fopen(log_name,'a');
fprintf(f,'\n1000 Completed\n');
fclose(f);

parfor (n = 1001:1250)
    ref = [data_in(n).ref_loc data_in(n).ref_name];
    test = [data_in(n).test_loc data_in(n).test_name];
    %fprintf(f,'%d: %s\n',n,test);
    try
        a = load(['./Features/' match_method '/' data_in(n).test_name(1:end-4) '.mat']);
        MOVs(n,:) = [a.OMOV];
    catch
        [MOVs(n,:),~] = OMOQ(ref, test, side_data(n), match_method);
    end
end
c_date = clock;
sname = sprintf('MOVs_%s%s%s%s.mat',num2str(c_date(1)),num2str(c_date(2),'%02d'),num2str(c_date(3),'%02d'),match_method);
save(sname,'MOVs','OMOV');
fprintf('\n1250 Completed\n');
f = fopen(log_name,'a');
fprintf(f,'\n1250 Completed\n');
fclose(f);
%
parfor (n = 1251:1500)
    ref = [data_in(n).ref_loc data_in(n).ref_name];
    test = [data_in(n).test_loc data_in(n).test_name];
    %fprintf(f,'%d: %s\n',n,test);
    try
        a = load(['./Features/' match_method '/' data_in(n).test_name(1:end-4) '.mat']);
        MOVs(n,:) = [a.OMOV];
    catch
        [MOVs(n,:),~] = OMOQ(ref, test, side_data(n), match_method);
    end
end
c_date = clock;
sname = sprintf('MOVs_%s%s%s%s.mat',num2str(c_date(1)),num2str(c_date(2),'%02d'),num2str(c_date(3),'%02d'),match_method);
save(sname,'MOVs','OMOV');
fprintf('\n1500 Completed\n');
f = fopen(log_name,'a');
fprintf(f,'\n1500 Completed\n');
fclose(f);

parfor (n = 1501:1750)
    ref = [data_in(n).ref_loc data_in(n).ref_name];
    test = [data_in(n).test_loc data_in(n).test_name];
    %fprintf(f,'%d: %s\n',n,test);
    try
        a = load(['./Features/' match_method '/' data_in(n).test_name(1:end-4) '.mat']);
        MOVs(n,:) = [a.OMOV];
    catch
        [MOVs(n,:),~] = OMOQ(ref, test, side_data(n), match_method);
    end
end
c_date = clock;
sname = sprintf('MOVs_%s%s%s%s.mat',num2str(c_date(1)),num2str(c_date(2),'%02d'),num2str(c_date(3),'%02d'),match_method);
save(sname,'MOVs','OMOV');
fprintf('\n1750 Completed\n');
f = fopen(log_name,'a');
fprintf(f,'\n1750 Completed\n');
fclose(f);

parfor (n = 1751:2000)
    ref = [data_in(n).ref_loc data_in(n).ref_name];
    test = [data_in(n).test_loc data_in(n).test_name];
    %fprintf(f,'%d: %s\n',n,test);
    try
        a = load(['./Features/' match_method '/' data_in(n).test_name(1:end-4) '.mat']);
        MOVs(n,:) = [a.OMOV];
    catch
        [MOVs(n,:),~] = OMOQ(ref, test, side_data(n), match_method);
    end
end
c_date = clock;
sname = sprintf('MOVs_%s%s%s%s.mat',num2str(c_date(1)),num2str(c_date(2),'%02d'),num2str(c_date(3),'%02d'),match_method);
save(sname,'MOVs','OMOV');
fprintf('\n2000 Completed\n');
f = fopen(log_name,'a');
fprintf(f,'\n2000 Completed\n');
fclose(f);

parfor (n = 2001:2250)
    ref = [data_in(n).ref_loc data_in(n).ref_name];
    test = [data_in(n).test_loc data_in(n).test_name];
    %fprintf(f,'%d: %s\n',n,test);
    try
        a = load(['./Features/' match_method '/' data_in(n).test_name(1:end-4) '.mat']);
        MOVs(n,:) = [a.OMOV];
    catch
        [MOVs(n,:),~] = OMOQ(ref, test, side_data(n), match_method);
    end
end
c_date = clock;
sname = sprintf('MOVs_%s%s%s%s.mat',num2str(c_date(1)),num2str(c_date(2),'%02d'),num2str(c_date(3),'%02d'),match_method);
save(sname,'MOVs','OMOV');
fprintf('\n2250 Completed\n');
f = fopen(log_name,'a');
fprintf(f,'\n2250 Completed\n');
fclose(f);

parfor (n = 2251:2500)
    ref = [data_in(n).ref_loc data_in(n).ref_name];
    test = [data_in(n).test_loc data_in(n).test_name];
    %fprintf(f,'%d: %s\n',n,test);
    try
        a = load(['./Features/' match_method '/' data_in(n).test_name(1:end-4) '.mat']);
        MOVs(n,:) = [a.OMOV];
    catch
        [MOVs(n,:),~] = OMOQ(ref, test, side_data(n), match_method);
    end
end
c_date = clock;
sname = sprintf('MOVs_%s%s%s%s.mat',num2str(c_date(1)),num2str(c_date(2),'%02d'),num2str(c_date(3),'%02d'),match_method);
save(sname,'MOVs','OMOV');
fprintf('\n2500 Completed\n');
f = fopen(log_name,'a');
fprintf(f,'\n2500 Completed\n');
fclose(f);

parfor (n = 2501:2750)
    ref = [data_in(n).ref_loc data_in(n).ref_name];
    test = [data_in(n).test_loc data_in(n).test_name];
    %fprintf(f,'%d: %s\n',n,test);
    try
        a = load(['./Features/' match_method '/' data_in(n).test_name(1:end-4) '.mat']);
        MOVs(n,:) = [a.OMOV];
    catch
        [MOVs(n,:),~] = OMOQ(ref, test, side_data(n), match_method);
    end
end
c_date = clock;
sname = sprintf('MOVs_%s%s%s%s.mat',num2str(c_date(1)),num2str(c_date(2),'%02d'),num2str(c_date(3),'%02d'),match_method);
save(sname,'MOVs','OMOV');
fprintf('\n2750 Completed\n');
f = fopen(log_name,'a');
fprintf(f,'\n2750 Completed\n');
fclose(f);

parfor (n = 2751:3000)
    ref = [data_in(n).ref_loc data_in(n).ref_name];
    test = [data_in(n).test_loc data_in(n).test_name];
    %fprintf(f,'%d: %s\n',n,test);
    try
        a = load(['./Features/' match_method '/' data_in(n).test_name(1:end-4) '.mat']);
        MOVs(n,:) = [a.OMOV];
    catch
        [MOVs(n,:),~] = OMOQ(ref, test, side_data(n), match_method);
    end
end
c_date = clock;
sname = sprintf('MOVs_%s%s%s%s.mat',num2str(c_date(1)),num2str(c_date(2),'%02d'),num2str(c_date(3),'%02d'),match_method);
save(sname,'MOVs','OMOV');
fprintf('\n3000 Completed\n');
f = fopen(log_name,'a');
fprintf(f,'\n3000 Completed\n');
fclose(f);

parfor (n = 3001:3250)
    ref = [data_in(n).ref_loc data_in(n).ref_name];
    test = [data_in(n).test_loc data_in(n).test_name];
    %fprintf(f,'%d: %s\n',n,test);
    try
        a = load(['./Features/' match_method '/' data_in(n).test_name(1:end-4) '.mat']);
        MOVs(n,:) = [a.OMOV];
    catch
        [MOVs(n,:),~] = OMOQ(ref, test, side_data(n), match_method);
    end
end
c_date = clock;
sname = sprintf('MOVs_%s%s%s%s.mat',num2str(c_date(1)),num2str(c_date(2),'%02d'),num2str(c_date(3),'%02d'),match_method);
save(sname,'MOVs','OMOV');
fprintf('\n3250 Completed\n');
f = fopen(log_name,'a');
fprintf(f,'\n3250 Completed\n');
fclose(f);

parfor (n = 3251:3500)
    ref = [data_in(n).ref_loc data_in(n).ref_name];
    test = [data_in(n).test_loc data_in(n).test_name];
    %fprintf(f,'%d: %s\n',n,test);
    try
        a = load(['./Features/' match_method '/' data_in(n).test_name(1:end-4) '.mat']);
        MOVs(n,:) = [a.OMOV];
    catch
        [MOVs(n,:),~] = OMOQ(ref, test, side_data(n), match_method);
    end
end
c_date = clock;
sname = sprintf('MOVs_%s%s%s%s.mat',num2str(c_date(1)),num2str(c_date(2),'%02d'),num2str(c_date(3),'%02d'),match_method);
save(sname,'MOVs','OMOV');
fprintf('\n3500 Completed\n');
f = fopen(log_name,'a');
fprintf(f,'\n3500 Completed\n');
fclose(f);

parfor (n = 3501:3750)
    ref = [data_in(n).ref_loc data_in(n).ref_name];
    test = [data_in(n).test_loc data_in(n).test_name];
    %fprintf(f,'%d: %s\n',n,test);
    try
        a = load(['./Features/' match_method '/' data_in(n).test_name(1:end-4) '.mat']);
        MOVs(n,:) = [a.OMOV];
    catch
        [MOVs(n,:),~] = OMOQ(ref, test, side_data(n), match_method);
    end
end
c_date = clock;
sname = sprintf('MOVs_%s%s%s%s.mat',num2str(c_date(1)),num2str(c_date(2),'%02d'),num2str(c_date(3),'%02d'),match_method);
save(sname,'MOVs','OMOV');
fprintf('\n3750 Completed\n');
f = fopen(log_name,'a');
fprintf(f,'\n3750 Completed\n');
fclose(f);

parfor (n = 3751:4000)
    ref = [data_in(n).ref_loc data_in(n).ref_name];
    test = [data_in(n).test_loc data_in(n).test_name];
    %fprintf(f,'%d: %s\n',n,test);
    try
        a = load(['./Features/' match_method '/' data_in(n).test_name(1:end-4) '.mat']);
        MOVs(n,:) = [a.OMOV];
    catch
        [MOVs(n,:),~] = OMOQ(ref, test, side_data(n), match_method);
    end
end
c_date = clock;
sname = sprintf('MOVs_%s%s%s%s.mat',num2str(c_date(1)),num2str(c_date(2),'%02d'),num2str(c_date(3),'%02d'),match_method);
save(sname,'MOVs','OMOV');
fprintf('\n4000 Completed\n');
f = fopen(log_name,'a');
fprintf(f,'\n4000 Completed\n');
fclose(f);

parfor n = 4001:4250
    ref = [data_in(n).ref_loc data_in(n).ref_name];
    test = [data_in(n).test_loc data_in(n).test_name];
    %fprintf(f,'%d: %s\n',n,test);
    try
        a = load(['./Features/' match_method '/' data_in(n).test_name(1:end-4) '.mat']);
        MOVs(n,:) = [a.OMOV];
    catch
        [MOVs(n,:),~] = OMOQ(ref, test, side_data(n), match_method);
    end
end
c_date = clock;
sname = sprintf('MOVs_%s%s%s%s.mat',num2str(c_date(1)),num2str(c_date(2),'%02d'),num2str(c_date(3),'%02d'),match_method);
save(sname,'MOVs','OMOV');
fprintf('\n4250 Completed\n');
f = fopen(log_name,'a');
fprintf(f,'\n4250 Completed\n');
fclose(f);

parfor (n = 4251:4500)
    ref = [data_in(n).ref_loc data_in(n).ref_name];
    test = [data_in(n).test_loc data_in(n).test_name];
    %fprintf(f,'%d: %s\n',n,test);
    try
        a = load(['./Features/' match_method '/' data_in(n).test_name(1:end-4) '.mat']);
        MOVs(n,:) = [a.OMOV];
    catch
        [MOVs(n,:),~] = OMOQ(ref, test, side_data(n), match_method);
    end
end
c_date = clock;
sname = sprintf('MOVs_%s%s%s%s.mat',num2str(c_date(1)),num2str(c_date(2),'%02d'),num2str(c_date(3),'%02d'),match_method);
save(sname,'MOVs','OMOV');
fprintf('\n4500 Completed\n');
f = fopen(log_name,'a');
fprintf(f,'\n4500 Completed\n');
fclose(f);

parfor (n = 4501:4750)
    ref = [data_in(n).ref_loc data_in(n).ref_name];
    test = [data_in(n).test_loc data_in(n).test_name];
    %fprintf(f,'%d: %s\n',n,test);
    try
        a = load(['./Features/' match_method '/' data_in(n).test_name(1:end-4) '.mat']);
        MOVs(n,:) = [a.OMOV];
    catch
        [MOVs(n,:),~] = OMOQ(ref, test, side_data(n), match_method);
    end
end
c_date = clock;
sname = sprintf('MOVs_%s%s%s%s.mat',num2str(c_date(1)),num2str(c_date(2),'%02d'),num2str(c_date(3),'%02d'),match_method);
save(sname,'MOVs','OMOV');
fprintf('\n4750 Completed\n');
f = fopen(log_name,'a');
fprintf(f,'\n4750 Completed\n');
fclose(f);

parfor (n = 4751:5000)
    ref = [data_in(n).ref_loc data_in(n).ref_name];
    test = [data_in(n).test_loc data_in(n).test_name];
    %     fprintf('\n%d',n);
    try
        a = load(['./Features/' match_method '/' data_in(n).test_name(1:end-4) '.mat']);
        MOVs(n,:) = [a.OMOV];
    catch
        [MOVs(n,:),~] = OMOQ(ref, test, side_data(n), match_method);
    end
end
c_date = clock;
sname = sprintf('MOVs_%s%s%s%s.mat',num2str(c_date(1)),num2str(c_date(2),'%02d'),num2str(c_date(3),'%02d'),match_method);
save(sname,'MOVs','OMOV');
fprintf('\n5000 Completed\n');
f = fopen(log_name,'a');
fprintf(f,'\n5000 Completed\n');
fclose(f);

parfor (n = 5001:5250)
    ref = [data_in(n).ref_loc data_in(n).ref_name];
    test = [data_in(n).test_loc data_in(n).test_name];
    %fprintf(f,'%d: %s\n',n,test);
    try
        a = load(['./Features/' match_method '/' data_in(n).test_name(1:end-4) '.mat']);
        MOVs(n,:) = [a.OMOV];
    catch
        [MOVs(n,:),~] = OMOQ(ref, test, side_data(n), match_method);
    end
end
c_date = clock;
sname = sprintf('MOVs_%s%s%s%s.mat',num2str(c_date(1)),num2str(c_date(2),'%02d'),num2str(c_date(3),'%02d'),match_method);
save(sname,'MOVs','OMOV');
fprintf('\n5250 Completed\n');
f = fopen(log_name,'a');
fprintf(f,'\n5250 Completed\n');
fclose(f);

parfor (n = 5251:N)
    ref = [data_in(n).ref_loc data_in(n).ref_name];
    test = [data_in(n).test_loc data_in(n).test_name];
    %     fprintf('%d: %s\n',n,test);
    try
        a = load(['./Features/' match_method '/' data_in(n).test_name(1:end-4) '.mat']);
        MOVs(n,:) = [a.OMOV];
    catch
        [MOVs(n,:),~] = OMOQ(ref, test, side_data(n), match_method);
    end
end
c_date = clock;
sname = sprintf('MOVs_%s%s%s%s.mat',num2str(c_date(1)),num2str(c_date(2),'%02d'),num2str(c_date(3),'%02d'),match_method);
save(sname,'MOVs','OMOV');


processing_time = toc;
fprintf('Processing Complete.  Time taken = %g hours\n',processing_time/3600);
f = fopen(log_name,'a');
fprintf(f,'Processing Complete.  Time taken = %g hours\n',processing_time/3600);
fclose(f);

% parpool CLOSE
end
