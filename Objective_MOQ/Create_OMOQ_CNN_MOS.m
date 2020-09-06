%Create CNN OMOQ Data

%Create a file called OMOQ_CNN_MOS.mat using -v7
%Contains 3 arrays
    %MeanOS (double array)
    %MedianOS (double array)
    %Name (cell array)
close all
clear all
clc

load('../Subjective_Testing/TSM_MOS_Scores.mat');
    
MeanOS = [data.MeanOS];
MedianOS = [data.MedianOS];
for n = 1:size(data,2)
    Name{1,n} = data(n).test_name;
end

save('Features/OMOQ_CNN_MOS_Col.mat','MeanOS','MedianOS','Name','-v7')