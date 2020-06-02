%Create MOS output file
close all
clear all
clc

% load('Combined_Results.mat')
% load('Results_v7.mat')
% load('Results_v8_MAD_STD_2nd_outliers_removed.mat');
% load('Results.mat')
load('Plotting_Data_Anon_No_Outliers.mat')
load Full_Source_filelist.mat
% load Sets_filelist.mat;
for n = 1:length(filelist)
    file_folder_name = split(filelist(n).location,'/');
    source_file(n).folder = ['Source/',char(file_folder_name(2)),'/'];
    source_file(n).name = char(file_folder_name(3));
end

%Init the array of structs
data(length(a)).test_loc = [];
data(length(a)).test_name = [];
data(length(a)).ref_loc = [];
data(length(a)).ref_name = [];
data(length(a)).method = [];
data(length(a)).TSM = [];
data(length(a)).MeanOS = [];
data(length(a)).MedianOS = [];
data(length(a)).std = [];
data(length(a)).MeanOS_RAW = [];
data(length(a)).MedianOS_RAW = [];
data(length(a)).std_RAW = [];
file_count = 1;

for n = 1:length(a)
    f = split(a(n).name,'/');
    data(file_count).test_loc = [char(f(1)) '/' char(f(2)) '/'];
    
    data(file_count).MeanOS = a(n).mean_MOS_norm;
    data(file_count).std = a(n).std_MOS_norm;
    data(file_count).MedianOS = a(n).median_MOS_norm;
    data(file_count).MeanOS_RAW = a(n).mean_MOS;
    data(file_count).std_RAW = a(n).std_MOS;
    data(file_count).MedianOS_RAW = median(a(n).MOS);
    
    TSM = split(f{3},'_');
    data(file_count).method = char(TSM(end-2));
    
    data(file_count).TSM = char(TSM(end-1));
    data(file_count).test_name = char(f(3));
    %Need to figure out how to add a 0 in the name before the TSM ratio
    
    
%     names{file_count} = data(file_count).test_name;
    
    
    %Find the matching Source audio file
    match = 0;
    q = 1;
    while ~match
        source = source_file(q).name;
        match = startsWith(data(file_count).test_name,source(1:end-4));
        if(data(file_count).test_name(length(source(1:end-3)))~= '_')
            match = 0;
        end
        if ~match
            q = q+1;
        end
    end
    
    data(file_count).ref_loc = source_file(q).folder;
    data(file_count).ref_name = source_file(q).name;
    
    
    
    file_count = file_count+1;
end
save_name = sprintf('TSM_MOS_Scores_%s.mat',date);
save(save_name,'data');
save('TSM_MOS_Scores.mat','data');

% T  = struct2table(data);

struct2csv(data,'TSM_MOS_Scores.csv')
