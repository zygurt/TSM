%Update MOS scores in features.

close all
clear all
clc

load('Features/New_Peak_Delta/MOVs_Final_Anchor_Ref.mat')
load('../Subjective_Testing/Results_RMSE_no_outliers.mat')
% load('../Subjective_Testing/Results_v4_no_MOS_limiting.mat')

% This section updates the MOS
for n = 1:length(a)
    MOVs(n,1) = a(n).mean_MOS_norm;
    MOVs(n,2) = a(n).median_MOS_norm;
    MOVs(n,3) = a(n).mean_MOS;
    MOVs(n,4) = median(a(n).MOS);
end

save('Features/RMSE_Outliers/MOVs_Final_Anchor_Ref_20200416.mat','OMOV','MOVs','-v7');
save('../ML/data/RMSE_Outliers/MOVs_Final_Anchor_Ref_20200416.mat','OMOV','MOVs','-v7');

% load('Features/New_Peak_Delta/MOVs_Final_To_Test_Source.mat')
% load('../Subjective_Testing/Results_RMSE_no_outliers.mat')
% % load('../Subjective_Testing/Results_v4_no_MOS_limiting.mat')
% 
% % This section updates the MOS
% for n = 1:length(a)
%     MOVs(n+88,1) = a(n).mean_MOS_norm;
%     MOVs(n+88,2) = a(n).median_MOS_norm;
%     MOVs(n+88,3) = a(n).mean_MOS;
%     MOVs(n+88,4) = median(a(n).MOS);
% end
% 
% save('Features/RMSE_Outliers/MOVs_Final_To_Test_Source_20200416.mat','OMOV','MOVs','-v7');


% %This section creates the mat file of names, MeanOS and MedianOS
% for n = 1:length(a)
%     name = a(n).name;
%     temp = split(name,'/');
%     if n < 5281
%         % Train
%         Name{n,1} = strcat('train/',temp{3});
%     else
%         % Test
%         Name{n,1} = strcat('test/',temp{3});
%     end
% end
% MeanOS = [a.mean_MOS_norm];
% MedianOS = [a.median_MOS_norm];
% 
% save('SMOQ_MOS.mat','Name','MeanOS','MedianOS','-v7');
% 
% C = {};
% 
% for n = 1:length(a)
%     name = a(n).name;
%     temp = split(name,'/');
%     if n < 5281
%         % Train
%         C{n,1} = strcat('train/',temp{3});
%     else
%         % Test
%         C{n,1} = strcat('test/',temp{3});
%     end
%     C{n,2} = a(n).mean_MOS_norm;
%     C{n,3} = a(n).median_MOS_norm;
% end
% 
% 
% fileID = fopen('SMOQ_MOS.csv','w');
% formatSpec = '%s,%1.5f,%1.5f\n';
% fprintf(fileID,'Name,MeanOS,MedianOS\n');
% [nrows,ncols] = size(C);
% for row = 1:nrows
%     fprintf(fileID,formatSpec,C{row,:});
% end
% fclose(fileID);
