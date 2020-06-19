function [processing_time] = Generate_Source_Features(match_method)
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
Test_File_Path = '../Subjective_Testing/Source/Subjective/';
Ref_File_Path = '../Subjective_Testing/Source/Subjective/';
%Initial load to get the number of files
temp = rec_filelist(Test_File_Path);
N = size(temp,1);
%Create the table
filelist = table(cell(N,1),cell(N,1),zeros(N,1),cell(N,1));%,zeros(N,1),zeros(N,1),zeros(N,1));
filelist.Properties.VariableNames = {'test_file','ref_file','TSM_per','method'};%,'MeanOS','MedianOS','STD'};

filelist.test_file = rec_filelist(Test_File_Path);
ref_filelist = rec_filelist(Ref_File_Path);

%Find the matching Source audio file
for n = 1:size(filelist,1)
    match = 0;
    q = 1;
    test_name = split(filelist.test_file(n),'/');
    test_name = char(test_name(end));
    while ~match
        source = split(ref_filelist(q),'/');
        source = char(source(end));
        match = startsWith(test_name,source(1:end-4));
%         if(test_name(length(source(1:end-3)))~= '_')
%             match = 0;
%         end
        if ~match
            q = q+1;
        end
    end
    filelist.ref_file(n) = ref_filelist(q);
    TSM = split(test_name,'_');
%     filelist.TSM_per(n) = str2double(TSM(end-1,1));
    filelist.TSM_per(n) = 100;
%     methods = split(filelist.test_file(n),'/');
%     filelist.method(n) = methods(end-1);
    filelist.method(n) = {'Source'};
    %     filelist.MeanOS(n) = 0;
    %     filelist.MedianOS(n) = 0;
    %     filelist.STD(n) = 0;
end


log_name = sprintf('Logs/%s_Feature_log.txt',match_method);


%Set OMOV for parallel processing
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




%% ------  Parallel processing -------
% N = height(filelist); %Calculated earlier

for n = 1:N
    side_data(n).TSM = filelist.TSM_per(n)/100;
    side_data(n).MeanOS = 5;
    side_data(n).MedianOS = 5;
    % side_data(n).StdOS = 1;
    side_data(n).MeanOS_RAW = 5;
    side_data(n).MedianOS_RAW = 5;
    % side_data(n).StdOS_RAW = 1;
end

% K = 10; %Number of groups to split processing into
K = 4;
nsize = N/K;
MOVs = zeros(N,size(OMOV,2));
%M = load('MOVs_20200130Interpolate_to_test.mat');
%MOVs = M.MOVs;

for k = 1:K
    fprintf('k = %d\n',k)
    parfor n = (k-1)*nsize+1:k*nsize
        test = char(filelist.test_file(n));
        ref = char(filelist.ref_file(n));
        f = fopen(log_name,'a');
        fprintf(f,'%d: %s\n',n,test);
        fprintf('%d: %s\n',n,test);
        fclose(f);
        [MOVs(n,:), ~] = OMOQ(ref, test, side_data(n), match_method);
    end
    c_date = clock;
    sname = sprintf('MOVs_Source_%s%s%s%s.mat',num2str(c_date(1)),num2str(c_date(2),'%02d'),num2str(c_date(3),'%02d'),match_method);

    save(sname,'MOVs','OMOV','-v7');
    f = fopen(log_name,'a');
    fprintf('\n%d/%d Completed\n',k*nsize,N);
    fprintf(f,'\n%d/%d Completed\n',k*nsize,N);
    fclose(f);

end
c_date = clock;
sname = sprintf('MOVs_Source_%s%s%s%s.mat',num2str(c_date(1)),num2str(c_date(2),'%02d'),num2str(c_date(3),'%02d'),match_method);
save(sname,'MOVs','OMOV','-v7');


MOV_Table = array2table(MOVs,'VariableNames',OMOV);
filelist = [filelist MOV_Table];
sname = sprintf('MOVs_Source_Table_%s%s%s%s.csv',num2str(c_date(1)),num2str(c_date(2),'%02d'),num2str(c_date(3),'%02d'),match_method);
writetable(filelist,sname)

processing_time = toc;
f = fopen(log_name,'a');
fprintf('Processing Complete.  Time taken = %g hours\n',processing_time/3600);
fprintf(f,'Processing Complete.  Time taken = %g hours\n',processing_time/3600);
fclose(f);

% parpool CLOSE
end
