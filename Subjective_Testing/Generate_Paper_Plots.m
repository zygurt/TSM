%Generate Plots
close all
clear all
clc

%%
addpath('../Functions');
addpath('../../External');
addpath('Functions');
load('Results_Anon_All.mat')
% load('Results_v8_MAD_STD_1st_outliers_removed.mat')
% load('Results_v8_MAD_STD_2nd_outliers_removed.mat')
load('Full_Source_filelist.mat');
recalculate = 0;
fid = fopen('log_Anon.txt','a');
fprintf(fid,'\n\n%s\n',date);
fclose(fid);
TSM = [38, 44, 53, 65, 78, 82, 99, 138, 166, 192];
TSM_methods = {'PV','IPL','WSOLA','FESOLA','HPTSM','uTVS'};
grey_lines= {'k-o', 'k-+', 'k-*', 'k.-', 'k-x', 'k-s', 'k--d', 'k--^', 'k--v'};

%% ------------------------- OUTLIERS RMSE RESULTS TO REMOVE BASED ON MEDIAN CRITERIA -------------------------

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
% title('PCC vs RMSE for Raw MeanOS')
% title('MD vs MAD vs STD-AD to Mean (MAD and STD-AD 1st outlier sets) Removed Pre-Normalisation')
% title('MD vs MAD vs STD-AD to Mean (MAD and STD-AD 2nd outlier sets) Removed Pre-Normalisation')
xlabel('$\rho$','Interpreter','latex')
ylabel('$\mathcal{L}$','interpreter','latex','Rotation',0,'Position',[0.035 0.5*1.05*max([u.RMSE])-0.08 -1])
axis([0.8*min([u.pearson_corr_mean]) 1.05*max([u.pearson_corr_mean]) ...
      0 1.05*max([u.RMSE])])
% text(0.2,.2,'RMSE Outliers','Color','r')
% text(0.2,.1,'PCC Outliers','Color','b')

set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');

print('Plots/PDF/Outliers_Raw_RMSE_Median', '-dpdf');
print('Plots/EPSC/Outliers_Raw_RMSE_Median', '-depsc');
print('Plots/PNG/Outliers_Raw_RMSE_Median', '-dpng');



% %Plot the normalised values with the original outliers
% 
% fprintf('Post-Normalisation\n')
% fprintf('Mean STD of Normalised File Ratings = %g\n',mean([a.std_MOS]));
% session_RMSE_norm = [u.RMSE_norm];
% TF = isoutlier(session_RMSE_norm,'median');
% mean_outlier_RMSE_norm = session_RMSE_norm(TF);
% 
% % session_STD_norm = [u.std_abs_diff_norm_mean];
% % TF = isoutlier(session_STD_norm);
% % mean_outlier_STD_norm = session_STD_norm(TF);
% %
% % session_MD_norm = [u.mean_diff_norm];
% % TF = isoutlier(session_MD_norm);
% % mean_outlier_MD_norm = session_MD_norm(TF);
% 
% session_PCC = [u.pearson_corr_MeanOS_norm];
% TF = isoutlier(session_PCC,'median');
% mean_norm_outlier_PCC = session_PCC(TF);
% 
% 
% figure('Position',[0 0 500 250])
% hold on
% for n = 1:length(u)
%     for k = 1:length(u(n).RMSE_norm)
%         plot(u(n).pearson_corr_MeanOS_norm(k),u(n).RMSE_norm(k),'k.')
%         if sum(u(n).pearson_corr_MeanOS_norm(k)==mean_norm_outlier_PCC)>0
%             plot(u(n).pearson_corr_MeanOS_norm(k),u(n).RMSE_norm(k),'b+')
%             %             fprintf('%s with key %s is a MD outlier with %d files\n',u(n).name, u(n).key, u(n).num_files(k))
%         end
%         if sum(u(n).RMSE(k)==mean_outlier_RMSE)>0
%             plot(u(n).pearson_corr_MeanOS_norm(k),u(n).RMSE_norm(k),'rx')
%             %             fprintf('%s with key %s is a MAD outlier with %d files\n',u(n).name, u(n).key, u(n).num_files(k))
%         end
%     end
% end
% hold off
% title('PCC vs RMSE for Norm. MeanOS (Raw outliers)')
% % title('MD vs MAD vs STD-AD to Mean (MAD and STD-AD 1st outlier sets) Removed Post-Normalisation (Pre-norm outliers marked)')
% % title('MD vs MAD vs STD-AD to Mean (MAD and STD-AD 2nd outlier sets) Removed Post-Normalisation (Pre-norm outliers marked)')
% 
% xlabel('$\rho$','Interpreter','latex')
% ylabel('$\mathcal{L}$','interpreter','latex','Rotation',0,'Position',[0.09 mean(get(gca,'YLim')) -1])
% 
% axis([0.8*min([u.pearson_corr_MeanOS_norm]) 1.05*max([u.pearson_corr_MeanOS_norm]) ...
%       0 1.05*max([u.RMSE_norm]) ])
% % text(0.2,0.2,'RMSE Outliers','Color','r')
% % text(0.2,0.1,'PCC Outliers','Color','b')
% 
% set(gca,...
%     'FontSize', 12, ...
%     'FontName', 'Times');
% 
% print('Plots/PDF/Outliers_Raw_Norm_Results_RMSE_Median', '-dpdf');
% print('Plots/EPSC/Outliers_Raw_Norm_Results_RMSE_Median', '-depsc');
% print('Plots/PNG/Outliers_Raw_Norm_Results_RMSE_Median', '-dpng');
% 
% 
% 
% 
% % POST NORMALISATION COMPARISON
% 
% 
% fprintf(fid,'\nNorm Outliers\n');
% figure('Position',[0 0 500 250])
% hold on
% for n = 1:length(u)
%     for k = 1:length(u(n).RMSE_norm)
%         plot(u(n).pearson_corr_MeanOS_norm(k),u(n).RMSE_norm(k),'k.')
%         if sum(u(n).pearson_corr_MeanOS_norm(k)==mean_norm_outlier_PCC)>0
%             plot(u(n).pearson_corr_MeanOS_norm(k),u(n).RMSE_norm(k),'b+')
%             fprintf('%s with filename %s is a PCC outlier (%g) with %d files\n', u(n).name, u(n).filename(12:end), u(n).pearson_corr_MeanOS_norm(k), u(n).num_files(k))
%             fprintf(fid,'%s with filename %s is a PCC outlier (%g) with %d files\n', u(n).name, u(n).filename(12:end), u(n).pearson_corr_MeanOS_norm(k), u(n).num_files(k));
%         end
%         if sum(u(n).RMSE_norm(k)==mean_outlier_RMSE_norm)>0
%             plot(u(n).pearson_corr_MeanOS_norm(k),u(n).RMSE_norm(k),'rx')
%             fprintf('%s with filename %s is an RMSE outlier (%g) with %d files\n', u(n).name,u(n).filename(12:end), u(n).RMSE_norm(k), u(n).num_files(k))
%             fprintf(fid,'%s with filename %s is an RMSE outlier (%g) with %d files\n', u(n).name,u(n).filename(12:end), u(n).RMSE_norm(k), u(n).num_files(k));
%         end
%     end
% end
% hold off
% 
% title('PCC vs RMSE for Norm. MeanOS')
% % title('MD vs MAD vs STD-AD to Mean (MAD and STD-AD 1st outlier sets) Removed Post-Normalisation')
% % title('MD vs MAD vs STD-AD to Mean (MAD and STD-AD 2nd outlier sets) Removed Post-Normalisation')
% xlabel('$\rho$','Interpreter','latex')
% ylabel('$\mathcal{L}$','interpreter','latex','Rotation',0,'Position',[0.09 mean(get(gca,'YLim')) -1])
% axis([0.8*min([u.pearson_corr_MeanOS_norm]) 1.05*max([u.pearson_corr_MeanOS_norm]) ...
%       0 1.05*max([u.RMSE_norm])])
% % text(0.2,0.2,'RMSE Outliers','Color','r')
% % text(0.2,0.1,'PCC Outliers','Color','b')
% 
% set(gca,...
%     'FontSize', 12, ...
%     'FontName', 'Times');
% 
% 
% print('Plots/PDF/Outliers_Norm_RMSE_Median', '-dpdf');
% print('Plots/EPSC/Outliers_Norm_RMSE_Median', '-depsc');
% print('Plots/PNG/Outliers_Norm_RMSE_Median', '-dpng');
% 
% fclose(fid);


% After Calculating Outliers load the results without them

load('Results_Anon_No_Outliers.mat')

load('Plotting_Data_Anon_No_Outliers.mat')


%% ---- Compare results for expert and non-expert listeners ----
fprintf('Compare Expert and Non-Expert Listeners\n')
% RMSE version
MMADs = [];
PCCs = [];
experts = [];

for n = 1:length(u)
%     if u(n).RMSE_norm >0.2
        MMADs = [MMADs, u(n).RMSE];
        PCCs = [PCCs, u(n).pearson_corr_mean]; %pearson_corr_mean pearson_corr_MeanOS_norm
        experts = [experts u(n).expert*ones(size(u(n).RMSE))];
%     end
end
expert_MMADs = MMADs(experts==1);
non_expert_MMADs = MMADs(experts==0);
expert_PCCs = PCCs(experts==1);
non_expert_PCCs = PCCs(experts==0);
vars.alpha = 0.05;
vars.theta = 0.05;
vars.percent_flag = 1;
vars.equal_var_flag = 0;
vars.plot_flag = 1;
vars.title_name = 'RMSE ($\mathcal{L}$)';
figure('Position',[1020 669 589 106])
subplot(121)
[H,pL,pU,d,t] = my_TOST(expert_MMADs, non_expert_MMADs, vars);
fprintf('Pre-Normalisation\n')
fid = fopen('log_Anon.txt','a');
fprintf(fid,'\n\nTOST RMSE Pre Normalisation Expert vs Non-Expert\n');
if H==1
    fprintf(fid,'Can claim equivalence based on %g%% confidence interval for equivalence.\n',(1-vars.alpha)*100);
    fprintf(fid,'Largest p value: %g\n',max(abs(pL),abs(pU)));
else
    fprintf(fid, 'Cannot claim equivalence based on %g%% confidence interval for equivalence.\n',(1-vars.alpha)*100);
    fprintf(fid, 'Largest p value: %g\n',max(abs(pL),abs(pU)));
end
if vars.percent_flag
    fprintf(fid,'theta = %g%% gives equiv of %g\n',vars.theta*100,mean(expert_MMADs)*vars.theta);
    fprintf(fid,'theta for Edge CI = %g\n',t);
else
    fprintf(fid,'theta = %g\n',vars.theta);
    fprintf(fid,'theta for Edge CI = %g\n',t);
end
fprintf(fid,'Cohen''s d = %g\n',d);
fclose(fid);

subplot(122)
vars.title_name = 'PCC ($\rho$)';
[H,pL,pU,d,t] = my_TOST(expert_PCCs, non_expert_PCCs, vars);
fid = fopen('log_Anon.txt','a');
fprintf(fid,'TOST PCC Pre Normalisation Expert vs Non-Expert\n');
if H==1
    fprintf(fid,'Can claim equivalence based on %g%% confidence interval for equivalence.\n',(1-vars.alpha)*100);
    fprintf(fid,'Largest p value: %g\n',max(abs(pL),abs(pU)));
else
    fprintf(fid, 'Cannot claim equivalence based on %g%% confidence interval for equivalence.\n',(1-vars.alpha)*100);
    fprintf(fid, 'Largest p value: %g\n',max(abs(pL),abs(pU)));
end
if vars.percent_flag
    fprintf(fid,'theta = %g%% gives equiv of %g\n',vars.theta*100,mean(expert_PCCs)*vars.theta);
    fprintf(fid,'theta for Edge CI = %g\n',t);
else
    fprintf(fid,'theta = %g\n',vars.theta);
end
fprintf(fid,'Cohen''s d = %g\n',d);
fclose(fid);

for n = 1:2
    subplot(1,2,n)
    set(gca,...
        'FontSize', 12, ...
        'FontName', 'Times');
end

print('Plots/EPSC/CI_Expert', '-depsc');
print('Plots/PNG/CI_Expert', '-dpng');
figure

%Post normalisation
MMADs = [];
PCCs = [];
experts = [];

for n = 1:length(u)
%     if u(n).RMSE_norm >0.2
        MMADs = [MMADs, u(n).RMSE_norm];
        PCCs = [PCCs, u(n).pearson_corr_MeanOS_norm]; %pearson_corr_mean pearson_corr_MeanOS_norm
        experts = [experts u(n).expert*ones(size(u(n).RMSE_norm))];
%     end
end
expert_MMADs = MMADs(experts==1);
non_expert_MMADs = MMADs(experts==0);
expert_PCCs = PCCs(experts==1);
non_expert_PCCs = PCCs(experts==0);
vars.alpha = 0.05;
vars.theta = 0.05;
vars.percent_flag = 1;
vars.equal_var_flag = 0;
vars.plot_flag = 1;
vars.title_name = 'RMSE Norm';
% subplot(223)
[H,pL,pU,d,t] = my_TOST(expert_MMADs, non_expert_MMADs, vars);
fprintf('Post-Normalisation\n')
fid = fopen('log_Anon.txt','a');
fprintf(fid,'TOST RMSE Post Normalisation Expert vs Non-Expert\n');
if H==1
    fprintf(fid,'Can claim equivalence based on %g%% confidence interval for equivalence.\n',(1-vars.alpha)*100);
    fprintf(fid,'Largest p value: %g\n',max(abs(pL),abs(pU)));
else
    fprintf(fid, 'Cannot claim equivalence based on %g%% confidence interval for equivalence.\n',(1-vars.alpha)*100);
    fprintf(fid, 'Largest p value: %g\n',max(abs(pL),abs(pU)));
end
if vars.percent_flag
    fprintf(fid,'theta = %g%% gives equiv of %g\n',vars.theta*100,mean(expert_MMADs)*vars.theta);
    fprintf(fid,'theta for Edge CI = %g\n',t);
else
    fprintf(fid,'theta = %g\n',vars.theta);
end
fprintf(fid,'Cohen''s d = %g\n',d);
fclose(fid);
% subplot(224)
vars.title_name = 'PCC Norm';
[H,pL,pU,d,t] = my_TOST(expert_PCCs, non_expert_PCCs, vars);
fid = fopen('log_Anon.txt','a');
fprintf(fid,'TOST PCC Post Normalisation Expert vs Non-Expert\n');
if H==1
    fprintf(fid,'Can claim equivalence based on %g%% confidence interval for equivalence.\n',(1-vars.alpha)*100);
    fprintf(fid,'Largest p value: %g\n',max(abs(pL),abs(pU)));
else
    fprintf(fid, 'Cannot claim equivalence based on %g%% confidence interval for equivalence.\n',(1-vars.alpha)*100);
    fprintf(fid, 'Largest p value: %g\n',max(abs(pL),abs(pU)));
end
if vars.percent_flag
    fprintf(fid,'theta = %g%% gives equiv of %g\n',vars.theta*100,mean(expert_PCCs)*vars.theta);
    fprintf(fid,'theta for Edge CI = %g\n',t);
else
    fprintf(fid,'theta = %g\n',vars.theta);
end
fprintf(fid,'Cohen''s d = %g\n',d);
fprintf(fid,'\n');
fclose(fid);






%% ---- Compare results for Participants hearing responses ----

fprintf('Compare Hearing\n')
%RMSE Version

MMADs_h = [];
PCCs = [];
hearing = [];

for n = 1:length(u)
    MMADs_h = [MMADs_h, u(n).RMSE];
    PCCs = [PCCs, u(n).pearson_corr_mean];
    hearing = [hearing u(n).hearing*ones(size(u(n).RMSE))];
end
good_hearing_MMADs = MMADs_h(hearing==1);
bad_hearing_MMADs = MMADs_h(hearing==0);
good_hearing_PCCs = PCCs(hearing==1);
bad_hearing_PCCs = PCCs(hearing==0);

vars.alpha = 0.05;
vars.theta = 0.05;
vars.percent_flag = 1;
vars.equal_var_flag = 0;
vars.plot_flag = 1;
vars.title_name = 'RMSE ($\mathcal{L}$)';
figure('Position',[1020 669 589 106])
subplot(121)
[H,pL,pU,d,t] = my_TOST(good_hearing_MMADs, bad_hearing_MMADs, vars);
fprintf('Pre-Normalisation\n')
fid = fopen('log_Anon.txt','a');
fprintf(fid,'TOST RMSE Pre Normalisation Hearing\n');
if H==1
    fprintf(fid,'Can claim equivalence based on %g%% confidence interval for equivalence.\n',(1-vars.alpha)*100);
    fprintf(fid,'Largest p value: %g\n',max(abs(pL),abs(pU)));
else
    fprintf(fid, 'Cannot claim equivalence based on %g%% confidence interval for equivalence.\n',(1-vars.alpha)*100);
    fprintf(fid, 'Largest p value: %g\n',max(abs(pL),abs(pU)));
end
if vars.percent_flag
    fprintf(fid,'theta = %g%% gives equiv of %g\n',vars.theta*100,mean(good_hearing_MMADs)*vars.theta);
    fprintf(fid,'theta for Edge CI = %g\n',t);
else
    fprintf(fid,'theta = %g\n',vars.theta);
end
fprintf(fid,'Cohen''s d = %g\n',d);
fclose(fid);

[H_PCC,P_PCC]=ttest2(good_hearing_PCCs,bad_hearing_PCCs,'Vartype','unequal');


vars.title_name = 'PCC ($\rho$)';
subplot(122)
[H,pL,pU,d,t] = my_TOST(good_hearing_PCCs, bad_hearing_PCCs, vars);
fid = fopen('log_Anon.txt','a');
fprintf(fid,'TOST PCC Pre Normalisation Hearing\n');
if H==1
    fprintf(fid,'Can claim equivalence based on %g%% confidence interval for equivalence.\n',(1-vars.alpha)*100);
    fprintf(fid,'Largest p value: %g\n',max(abs(pL),abs(pU)));
else
    fprintf(fid, 'Cannot claim equivalence based on %g%% confidence interval for equivalence.\n',(1-vars.alpha)*100);
    fprintf(fid, 'Largest p value: %g\n',max(abs(pL),abs(pU)));
end
if vars.percent_flag
    fprintf(fid,'theta = %g%% gives equiv of %g\n',vars.theta*100,mean(good_hearing_PCCs)*vars.theta);
    fprintf(fid,'theta for Edge CI = %g\n',t);
else
    fprintf(fid,'theta = %g\n',vars.theta);
end
fprintf(fid,'Cohen''s d = %g\n',d);
fclose(fid);

for n = 1:2
    subplot(1,2,n)
    set(gca,...
        'FontSize', 12, ...
        'FontName', 'Times');
end
print('Plots/EPSC/CI_Hearing', '-depsc');
print('Plots/PNG/CI_Hearing', '-dpng');
figure

MMADs_h = [];
PCCs = [];
hearing = [];

for n = 1:length(u)
    MMADs_h = [MMADs_h, u(n).RMSE_norm];
    PCCs = [PCCs, u(n).pearson_corr_MeanOS_norm];
    hearing = [hearing u(n).hearing*ones(size(u(n).RMSE_norm))];
end
good_hearing_MMADs = MMADs_h(hearing==1);
bad_hearing_MMADs = MMADs_h(hearing==0);
good_hearing_PCCs = PCCs(hearing==1);
bad_hearing_PCCs = PCCs(hearing==0);

vars.alpha = 0.05;
vars.theta = 0.05;
vars.percent_flag = 1;
vars.equal_var_flag = 0;
vars.plot_flag = 1;
vars.title_name = 'RMSE Norm';
% subplot(223)
[H,pL,pU,d,t] = my_TOST(good_hearing_MMADs, bad_hearing_MMADs, vars);
fprintf('Post-Normalisation\n')
fid = fopen('log_Anon.txt','a');
fprintf(fid,'TOST RMSE Post Normalisation Hearing\n');
if H==1
    fprintf(fid,'Can claim equivalence based on %g%% confidence interval for equivalence.\n',(1-vars.alpha)*100);
    fprintf(fid,'Largest p value: %g\n',max(abs(pL),abs(pU)));
else
    fprintf(fid, 'Cannot claim equivalence based on %g%% confidence interval for equivalence.\n',(1-vars.alpha)*100);
    fprintf(fid, 'Largest p value: %g\n',max(abs(pL),abs(pU)));
end
if vars.percent_flag
    fprintf(fid,'theta = %g%% gives equiv of %g\n',vars.theta*100,mean(good_hearing_MMADs)*vars.theta);
    fprintf(fid,'theta for Edge CI = %g\n',t);
else
    fprintf(fid,'theta = %g\n',vars.theta);
end
fprintf(fid,'Cohen''s d = %g\n',d);
fclose(fid);
vars.title_name = 'PCC Norm';
% subplot(224)
[H,pL,pU,d,t] = my_TOST(good_hearing_PCCs, bad_hearing_PCCs, vars);

fid = fopen('log_Anon.txt','a');
fprintf(fid,'TOST PCC Post Normalisation Hearing\n');
if H==1
    fprintf(fid,'Can claim equivalence based on %g%% confidence interval for equivalence.\n',(1-vars.alpha)*100);
    fprintf(fid,'Largest p value: %g\n',max(abs(pL),abs(pU)));
else
    fprintf(fid, 'Cannot claim equivalence based on %g%% confidence interval for equivalence.\n',(1-vars.alpha)*100);
    fprintf(fid, 'Largest p value: %g\n',max(abs(pL),abs(pU)));
end
if vars.percent_flag
    fprintf(fid,'theta = %g%% gives equiv of %g\n',vars.theta*100,mean(good_hearing_PCCs)*vars.theta);
    fprintf(fid,'theta for Edge CI = %g\n',t);
else
    fprintf(fid,'theta = %g\n',vars.theta);
end
fprintf(fid,'Cohen''s d = %g\n',d);
fprintf(fid,'\n');
fclose(fid);






%% -------------------- Compare MATLAB to WAET results -------------------
fprintf('Compare Lab and Remote Testing\n')
% RMSE version

Offline_MMADs = [u(1:65).RMSE];
Online_MMADs = [u(66:end).RMSE];
Offline_PCCs = [u(1:65).pearson_corr_mean];
Online_PCCs = [u(66:end).pearson_corr_mean];

vars.alpha = 0.05;
vars.theta = 0.05;
vars.percent_flag = 1;
vars.equal_var_flag = 0;
vars.plot_flag = 1;
vars.title_name = 'RMSE ($\mathcal{L}$)';
figure('Position',[1020 669 589 106])
subplot(121)
[H,pL,pU,d,t] = my_TOST(Offline_MMADs, Online_MMADs, vars);
fprintf('Pre-Normalisation\n')
fid = fopen('log_Anon.txt','a');
fprintf(fid,'TOST RMSE Pre Normalisation Lab vs Remote\n');
if H==1
    fprintf(fid,'Can claim equivalence based on %g%% confidence interval for equivalence.\n',(1-vars.alpha)*100);
    fprintf(fid,'Largest p value: %g\n',max(abs(pL),abs(pU)));
else
    fprintf(fid, 'Cannot claim equivalence based on %g%% confidence interval for equivalence.\n',(1-vars.alpha)*100);
    fprintf(fid, 'Largest p value: %g\n',max(abs(pL),abs(pU)));
end
if vars.percent_flag
    fprintf(fid,'theta = %g%% gives equiv of %g\n',vars.theta*100,mean(Offline_MMADs)*vars.theta);
    fprintf(fid,'theta for Edge CI = %g\n',t);
else
    fprintf(fid,'theta = %g\n',vars.theta);
end
fprintf(fid,'Cohen''s d = %g\n',d);
fclose(fid);

[H_PCC,P_PCC]=ttest2(Offline_PCCs,Online_PCCs,'Vartype','unequal');

vars.title_name = 'PCC ($\rho$)';
subplot(122)
[H,pL,pU,d,t] = my_TOST(Offline_PCCs, Online_PCCs, vars);
fid = fopen('log_Anon.txt','a');
fprintf(fid,'TOST PCC Pre Normalisation Lab vs Remote\n');
if H==1
    fprintf(fid,'Can claim equivalence based on %g%% confidence interval for equivalence.\n',(1-vars.alpha)*100);
    fprintf(fid,'Largest p value: %g\n',max(abs(pL),abs(pU)));
else
    fprintf(fid, 'Cannot claim equivalence based on %g%% confidence interval for equivalence.\n',(1-vars.alpha)*100);
    fprintf(fid, 'Largest p value: %g\n',max(abs(pL),abs(pU)));
end
if vars.percent_flag
    fprintf(fid,'theta = %g%% gives equiv of %g\n',vars.theta*100,mean(Offline_PCCs)*vars.theta);
    fprintf(fid,'theta for Edge CI = %g\n',t);
else
    fprintf(fid,'theta = %g\n',vars.theta);
end
fprintf(fid,'Cohen''s d = %g\n',d);
fclose(fid);

for n = 1:2
    subplot(1,2,n)
    set(gca,...
        'FontSize', 12, ...
        'FontName', 'Times');
end

print('Plots/EPSC/CI_Remote', '-depsc');
print('Plots/PNG/CI_Remote', '-dpng');
figure

Offline_MMADs = [u(1:65).RMSE_norm];
Online_MMADs = [u(66:end).RMSE_norm];
Offline_PCCs = [u(1:65).pearson_corr_MeanOS_norm];
Online_PCCs = [u(66:end).pearson_corr_MeanOS_norm];

vars.alpha = 0.05;
vars.theta = 0.05;
vars.percent_flag = 1;
vars.equal_var_flag = 0;
vars.plot_flag = 1;
vars.title_name = 'RMSE Norm';
% subplot(223)
[H,pL,pU,d,t] = my_TOST(Offline_MMADs, Online_MMADs, vars);

fprintf('Post-Normalisation\n')
fid = fopen('log_Anon.txt','a');
fprintf(fid,'TOST RMSE Post Normalisation Lab vs Remote\n');
if H==1
    fprintf(fid,'Can claim equivalence based on %g%% confidence interval for equivalence.\n',(1-vars.alpha)*100);
    fprintf(fid,'Largest p value: %g\n',max(abs(pL),abs(pU)));
else
    fprintf(fid, 'Cannot claim equivalence based on %g%% confidence interval for equivalence.\n',(1-vars.alpha)*100);
    fprintf(fid, 'Largest p value: %g\n',max(abs(pL),abs(pU)));
end
if vars.percent_flag
    fprintf(fid,'theta = %g%% gives equiv of %g\n',vars.theta*100,mean(Offline_MMADs)*vars.theta);
    fprintf(fid,'theta for Edge CI = %g\n',t);
else
    fprintf(fid,'theta = %g\n',vars.theta);
end
fprintf(fid,'Cohen''s d = %g\n',d);
fclose(fid);

vars.title_name = 'PCC Norm';
% subplot(224)
[H,pL,pU,d,t] = my_TOST(Offline_PCCs, Online_PCCs, vars);
fid = fopen('log_Anon.txt','a');
fprintf(fid,'TOST PCC Post Normalisation Lab vs Remote\n');
if H==1
    fprintf(fid,'Can claim equivalence based on %g%% confidence interval for equivalence.\n',(1-vars.alpha)*100);
    fprintf(fid,'Largest p value: %g\n',max(abs(pL),abs(pU)));
else
    fprintf(fid, 'Cannot claim equivalence based on %g%% confidence interval for equivalence.\n',(1-vars.alpha)*100);
    fprintf(fid, 'Largest p value: %g\n',max(abs(pL),abs(pU)));
end
if vars.percent_flag
    fprintf(fid,'theta = %g%% gives equiv of %g\n',vars.theta*100,mean(Offline_PCCs)*vars.theta);
    fprintf(fid,'theta for Edge CI = %g\n',t);
else
    fprintf(fid,'theta = %g\n',vars.theta);
end
fprintf(fid,'Cohen''s d = %g\n',d);
fprintf(fid,'\n');
fclose(fid);




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
p = plot3(1:length(I), [a(I).mean_MOS_norm],(1+max(max(h.BinCounts)))*ones(1,length(I)),'r-');
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




%% ------------Exploring standard deviation vs number of ratings-------


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
ylabel('$\sigma_s$','interpreter','latex','Rotation',0,'Position',[2.3 mean(get(gca,'YLim')) -1])

set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');

print('Plots/TIFF/STD_vs_Responses_Hist2', '-dtiff');
print('Plots/EPSC/STD_vs_Responses_Hist2', '-depsc');
print('Plots/PNG/STD_vs_Responses_Hist2', '-dpng');



%% Create latex table with methods vs time-scale values
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
JASAlatexTable(input,fid);
fclose(fid);


%% Plot for each class of file
fprintf('File class line graphs\n')

TSM_fine = [0.3838 0.4427 0.5383 0.6524 0.7821 0.8258 0.9961 1.381 1.667 1.924];
edges = [0.3, mean([TSM_fine(1:end-1);TSM_fine(2:end)]), 3];
x = [a(5361:5440).TSM]'/100;
y = [a(5361:5440).mean_MOS_norm]';
categ = {a(5361:5440).cat};
[El_Means,cats] = SubjTSMAverage(x,y,edges,categ);
x = [a(5281:5360).TSM]'/100;
y = [a(5281:5360).mean_MOS_norm]';
categ = {a(5281:5360).cat};
[Fuzz_Means] = SubjTSMAverage(x,y,edges,categ);
x = [a(5441:5520).TSM]'/100;
y = [a(5441:5520).mean_MOS_norm]';
categ = {a(5441:5520).cat};
[NMF_Means] = SubjTSMAverage(x,y,edges,categ);
TSM_methods_ALL = {'PV','IPL','WSOLA','FESOLA','HPTSM','uTVS','Elastique','FuzzyPV','NMFTSM'};


Music.overall_mean = mean(Music.TSM_method,1);
figure('Position',[1680-500 200 564 386])
hold on
for n = 1:size(Music.overall_mean,3)
    plot(TSM/100,Music.overall_mean(:,:,n),grey_lines{n})
end
% Elastique
plot(TSM/100,El_Means(1,:),grey_lines{7})
%FuzzyPV
plot(TSM/100,Fuzz_Means(1,:),grey_lines{8})
% NMFTSM
plot(TSM/100,NMF_Means(1,:),grey_lines{9})
% title('Mean MOS for Music files')
xlabel('Time-Scale Ratio (\beta)')
ylabel('Mean MOS')
legend(TSM_methods_ALL,'Location','NorthOutside','NumColumns',3);
% columnlegend(3,TSM_methods_ALL,'Location','NorthOutside')
axis([0.2 2 0.9 5.1])
hold off
set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');
print('Plots/TIFF/Means_Music', '-dtiff');
print('Plots/EPSC/Means_Music', '-depsc');
print('Plots/PNG/Means_Music', '-dpng');

Solo.overall_mean = mean(Solo.TSM_method,1);
figure('Position',[1680-500 200 564 386])
hold on
for n = 1:size(Solo.overall_mean,3)
    plot(TSM/100,Solo.overall_mean(:,:,n),grey_lines{n})
end
% Elastique
plot(TSM/100,El_Means(2,:),grey_lines{7})
%FuzzyPV
plot(TSM/100,Fuzz_Means(2,:),grey_lines{8})
% NMFTSM
plot(TSM/100,NMF_Means(2,:),grey_lines{9})
% title('Mean MOS for Solo files')
xlabel('Time-Scale Ratio (\beta)')
ylabel('Mean MOS')
legend(TSM_methods_ALL,'Location','NorthOutside','NumColumns',3);
axis([0.2 2 0.9 5.1])
hold off
set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');
print('Plots/TIFF/Means_Solo', '-dtiff');
print('Plots/EPSC/Means_Solo', '-depsc');
print('Plots/PNG/Means_Solo', '-dpng');

Voice.overall_mean = mean(Voice.TSM_method,1);
figure('Position',[1680-500 200 564 386])
hold on
for n = 1:size(Voice.overall_mean,3)
    plot(TSM/100,Voice.overall_mean(:,:,n),grey_lines{n})
end
% Elastique
plot(TSM/100,El_Means(3,:),grey_lines{7})
%FuzzyPV
plot(TSM/100,Fuzz_Means(3,:),grey_lines{8})
% NMFTSM
plot(TSM/100,NMF_Means(3,:),grey_lines{9})
% title('Mean MOS for Voice files')
xlabel('Time-Scale Ratio (\beta)')
ylabel('Mean MOS')
legend(TSM_methods_ALL,'Location','NorthOutside','NumColumns',3);
axis([0.2 2 0.9 5.1])
hold off
set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');
print('Plots/TIFF/Means_Voice', '-dtiff');
print('Plots/EPSC/Means_Voice', '-depsc');
print('Plots/PNG/Means_Voice', '-dpng');


% overall_mean = mean([Complex.overall_mean ; Music.overall_mean ; Solo.overall_mean ; Voice.overall_mean],1);
overall_mean = mean([Music.overall_mean ; Solo.overall_mean ; Voice.overall_mean],1);

figure('Position',[1680-500 200 564 386])
hold on
for n = 1:size(overall_mean,3)
    plot(TSM/100,overall_mean(:,:,n),grey_lines{n})
end


% Elastique
plot(TSM/100,El_Means(4,:),grey_lines{7})
%FuzzyPV
plot(TSM/100,Fuzz_Means(4,:),grey_lines{8})
% NMFTSM
plot(TSM/100,NMF_Means(4,:),grey_lines{9})


% %polynomial fit version
% fit_order = 3;
% TSM_fine = [0.3838 0.4427 0.5383 0.6524 0.7821 0.8258 0.9961 1.381 1.667 1.924];
% % startPoints = [min([a(5281:end).TSM]'/100) max([a(5281:end).TSM]'/100) 1 5];
% %Fuzzy
% x = [a(5281:5360).TSM]'/100;
% y = [a(5281:5360).mean_MOS_norm]';
% P = polyfit(x,y,fit_order);
% plot(TSM_fine,polyval(P,TSM_fine),'b')%,x,y,'r.')
% %NMFTSM
% x = [a(5441:5520).TSM]'/100;
% y = [a(5441:5520).mean_MOS_norm]';
% P = polyfit(x,y,fit_order);
% plot(TSM_fine,polyval(P,TSM_fine),'r')%,x,y,'r.')
% %Elastique
% x = [a(5361:5440).TSM]'/100;
% y = [a(5361:5440).mean_MOS_norm]';
% P = polyfit(x,y,fit_order);
% plot(TSM_fine,polyval(P,TSM_fine),'k')%,x,y,'r.')
% hold off


% title('Mean MeanOS for All Files')
xlabel('Time-Scale Ratio (\beta)')
ylabel('Mean MOS')
legend(TSM_methods_ALL,'Location','NorthOutside','NumColumns',3);
axis([0.2 2 0.9 5.1])
set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');
print('Plots/PDF/Means_Overall', '-dpdf');
print('Plots/EPSC/Means_Overall', '-depsc');
print('Plots/PNG/Means_Overall', '-dpng');



%% -------------------- Plot Age vs RMSE -------------------------

fprintf('Age Related Plotting\n')
figure('Position',[1680-500 200 500 250])
hold on
for f = 1:length(u)
    plot(u(f).age*ones(size(u(f).RMSE)),u(f).RMSE,'k.')
end
hold off
% title('Age vs Mean Absolute Difference to Mean')
xlabel('Age')
ylabel('$\mathcal{L}$','interpreter','latex','Rotation',0,'Position',[8.8 mean(get(gca,'YLim')) -1])
axis([0.9*min([u.age]) 1.1*max([u.age]) 0.9*min([u.RMSE]) 1.1*max([u.RMSE])])
set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');
print('Plots/PDF/Age_vs_RMSE', '-dpdf');
print('Plots/EPSC/Age_vs_RMSE', '-depsc');
print('Plots/PNG/Age_vs_RMSE', '-dpng');

fid = fopen('log_Anon.txt','a');
fprintf(fid,'Pearson Correlation of age to RMSE = %g\n',corr([u.age]',[u.RMSE]'));
fclose(fid);

figure('Position',[1680-500 200 500 250])
hold on
for f = 1:length(u)
    plot(u(f).age*ones(size(u(f).pearson_corr_mean)),u(f).pearson_corr_mean,'k.')
end
hold off
% title('Age vs Mean Absolute Difference to Mean')
xlabel('Age')
ylabel('$\rho$','interpreter','latex','Rotation',0,'Position',[8.8 mean(get(gca,'YLim')) -1])
axis([0.9*min([u.age]) 1.1*max([u.age]) 0.9*min([u.pearson_corr_mean]) 1.1*max([u.pearson_corr_mean])])
set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');
print('Plots/PDF/Age_vs_PCC', '-dpdf');
print('Plots/EPSC/Age_vs_PCC', '-depsc');
print('Plots/PNG/Age_vs_PCC', '-dpng');

fid = fopen('log_Anon.txt','a');
fprintf(fid,'Pearson Correlation of age to PCC = %g\n',corr([u.age]',[u.pearson_corr_mean]'));
fclose(fid);



%% ----------  Before and After Normalisation ----------

fprintf('Compare MAD to Mean and Median\n')
%Consider the Mean Absolute difference to the Mean or Median.
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
ylabel('Normalized Probability')
axis([min(EDGES),max(EDGES),0,1.1*max([N_MAD_MEAN N_MAD_NMEAN])])
legend('Raw', 'Normalized','location','northeast');
set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');
print('Plots/PDF/RMSE_Pre_Post_Norm', '-dpdf');
print('Plots/EPSC/RMSE_Pre_Post_Norm', '-depsc');
print('Plots/PNG/RMSE_Pre_Post_Norm', '-dpng');
















%% ---------- Histogram2 of PCC with MAD ---------------
fprintf('PCC and RMSE\n')


figure('Position',[1680-500 200 500 250])
plot([u.RMSE],[u.pearson_corr_mean],'r.')
hold on
plot([u.RMSE_norm],[u.pearson_corr_MeanOS_norm],'b.')

line([0.15 1.3],[mean([u.pearson_corr_mean]) mean([u.pearson_corr_mean])],'Color','red','LineWidth',1) %Horizontal
line([mean([u.RMSE]) mean([u.RMSE])],[0.45 1.04],'Color','red','LineWidth',1) %Vertical
line([0.15 1.3],[mean([u.pearson_corr_MeanOS_norm]) mean([u.pearson_corr_MeanOS_norm])],'Color','blue','LineWidth',1) %Horizontal
line([mean([u.RMSE_norm]) mean([u.RMSE_norm])],[0.45 1.04],'Color','blue','LineWidth',1) %Vertical
hold off

axis([0.15 1.3 0.45 1.04])
xlabel('$\mathcal{L}$','Interpreter','latex')
ylabel('$\rho$','Interpreter','latex','Rotation',0,'Position',[0.04 mean(get(gca,'YLim')) -1])
legend('Raw','Normalized','Location','SouthWest')
set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');

fid = fopen('log_Anon.txt','a');
fprintf(fid,'Raw Mean PCC: %g\n', mean([u.pearson_corr_mean]));
fprintf(fid,'Norm Mean PCC: %g\n', mean([u.pearson_corr_MeanOS_norm]));
fprintf(fid,'Raw Mean RMSE: %g\n', mean([u.RMSE]));
fprintf(fid,'Norm Mean RMSE: %g\n', mean([u.RMSE_norm]));

fclose(fid);


print('Plots/PDF/RMSE_vs_PCC', '-dpdf');
print('Plots/EPSC/RMSE_vs_PCC', '-depsc');
print('Plots/PNG/RMSE_vs_PCC', '-dpng');

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
input.tableCaption = 'Mean MOS for each class of training source file, Music, Solo Instrument, Voice, Overall classes are considered.';
input.tableLabel = 'MOS_Results';
latex_output = JASAlatexTable(input,fid);

%Means without 99.61%
fprintf(fid,'\n\n\nLatex output for file types without 99.61%\n\n');
input.data = zeros(6,4);
input.data(:,1) = mean(mean(Music.type(:,:,[1:6,8:10]),3));
input.data(:,2) = mean(mean(Solo.type(:,:,[1:6,8:10]),3));
input.data(:,3) = mean(mean(Voice.type(:,:,[1:6,8:10]),3));
% input.data(:,5) = mean(input.data(:,1:4),2);
% input.data(:,6) = sum([size(Complex.type_mean,1)*input.data(:,1) ,...
%                             size(Music.type_mean,1)*input.data(:,2) ,...
%                             size(Solo.type_mean,1)*input.data(:,3) ,...
%                             size(Voice.type_mean,1)*input.data(:,4)],2)/88;
T = mean((TSM_res.TSM1+TSM_res.TSM2+TSM_res.TSM3+TSM_res.TSM4+TSM_res.TSM5 +...
    TSM_res.TSM6+TSM_res.TSM8+TSM_res.TSM9+TSM_res.TSM10)/9);
input.data(:,4) = T;
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


%% Find the inverse percentile for the Small PEAQB Network
%These values are copied from the Training Results Analysis script
RMSE = 0.668212084000000;
rho = 0.719131234666667;

fid = fopen('log_Anon.txt','a');
p = inv_prctile(RMSE,[u.mean_abs_diff_norm_mean],'down');
fprintf(fid,'RMSE Percentile for PEAQB Small is %d\n', p);
p = inv_prctile(rho,[u.pearson_corr_MeanOS_norm],'up');
fprintf(fid,'PCC Percentile for PEAQB Small is %d\n', p);

% figure
% histogram([u.mean_abs_diff_norm_mean])
% title('RMSE')
% hold on
% line([RMSE RMSE],[0 90])
% hold off
% figure
% histogram([u.pearson_corr_MeanOS_norm])
% title('PCC')
% hold on
% line([rho rho],[0 120])
% hold off



