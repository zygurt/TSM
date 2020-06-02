%Intraclass Correlation Coefficient
%Based on Speech Enhancement: Theory and Practice (2nd Ed.) by Philipos C. Louizou
%Pages 473-474


%Start by making a master array of Files as Rows, Participants as columns


close all
clear all
% clc
addpath('../Functions/');
addpath('Functions/');
recalculate = 1;
% load('Results_v8.mat')
% load('Results_v8_MAD_STD_1st_outliers_removed.mat')
% load('Results_v8_MAD_STD_2nd_outliers_removed.mat');
load('Results_Anon_No_Outliers.mat')
if recalculate == 1
    
    results_path = 'Results_Anon/';
    audio_path = 'Sets/';
    
    %Account for Windows slash folder structure
    for n = 1:length(a)
        a(n).name = strrep(a(n).name,'\','/');
    end
    
    filelist = {a.name};
    % session = 1:length([u.mean_abs_diff]);
    
    %Create session name cell array
    session = {};
    pos = 1;
    for n = 1:size(u,2)
        for k = 1:length(u(n).mean_abs_diff_mean)
            session{pos,1} = u(n).name;
            session{pos,2} = n;
            pos = pos+1;
        end
    end
    
    OS = zeros(length(filelist),length(session));
        OS_norm = zeros(length(filelist),length(session));

    %For all of the results files, add the scores to the array
    
    %Create a list of files in the results_path
    res_filelist = rec_filelist(results_path);
    
    %Create master list of files (This should probably be replaced with
    %loading a file and a check to see if the file exists)
    audio_filelist = rec_filelist(audio_path);
    for f = 1:size(audio_filelist,1)
        a(f).name = char(audio_filelist(f));
        name_split = split(a(f).name,'_');
        a(f).TSM = str2double(char(name_split(end-1)));
        a(f).method = char(name_split(end-2));
        a(f).MOS = [];
    end
    %Add the responses to the master list
    pos = 1;
%     fprintf('Creating master list of file ratings\n')
    for f = 1:size(res_filelist,1)
        r = load(char(res_filelist(f)));
        fprintf('%s: Sets ',r.user_data.name);
        for s = 1:size(r.user_data.files,2)
            if isfield(r.user_data.files(s).file,'MOS_norm')
                fprintf('%d, ',s);
                
                %Find the username position
                
                %             match = 0;
                %                 un = 1;
                %                 while ~match
                %                     match=strcmp(session{un},r.user_data.name);
                %                     un = un+1;
                %                 end
                %                 %Remove Final an addition
                %                 un = un-1;
                
                
                %Find the file name position in filelist
                for n = 1:size(r.user_data.files(s).file,2)
                    match = 0;
                    fn = 1;
                    while ~match
                        match=strcmp(filelist{fn},r.user_data.files(s).file(n).name);
                        fn = fn+1;
                    end
                    %Remove Final an addition
                    fn = fn-1;
                    
                    if ~isempty(r.user_data.files(s).file(n).MOS_norm)
                        OS(fn,pos) = r.user_data.files(s).file(n).MOS;
                        OS_norm(fn,pos) = r.user_data.files(s).file(n).MOS_norm;
                    end
                end
                pos = pos+1;
                
                
                %Store the MOS in a
                %                 OS(fn,un) = r.user_data.files(s).file(n).MOS;
                %                 a(fn).MOS = [a(fn).MOS r.user_data.files(s).file(n).MOS];
                
            end
        end
        fprintf('\n');
    end
    
    save('Opinion_Matrix_No_Outliers.mat','session', 'OS', 'OS_norm');
    
else
    
    load('Opinion_Matrix_No_Outliers.mat');
end
% OS_v = OS~=0;
% file_means = sum(OS,2)./sum(OS_v,2);
% [~,I] = sort(file_means);
% OS_mean_sort = OS(I,:);
% C = cov(OS_mean_sort);
% 
% for n = 1:size(C,1)
%     C(n,n)=0;
% end
% figure;
% plot(OS_mean_sort,'.')
% mean_covariance = mean(abs(C),2);
% figure
% plot([u.mean_abs_diff_mean],mean_covariance,'.')
% title('MAD to Mean vs Covariance')
% xlabel('MAD to Mean')
% ylabel('Covariance')
% 
% OS_v = OS_norm~=0;
% M = zeros(size(OS_norm));
% G = sum(sum(OS_norm));
% b = size(OS_norm,1);
% t = size(OS_norm,2);
% % b_arr = sum(OS_v,1);
% % t_arr = sum(OS_v,2);
% T_arr = sum(OS_norm,1);
% B_arr = sum(OS_norm,2);
% b_arr = sum(OS_norm,2)./sum(OS_v,2);
% temp = OS_norm;
% temp(temp==0) = NaN;
% a_arr = std(temp,0,2,'omitnan');
% 
% for n = 1:size(OS_norm,1)
%     for k = 1:size(OS_norm,2)
%         if(OS_norm(n,k)==0)
%             M(n,k) = a_arr(n)*randn(1)+b_arr(n);
% %             T = T_arr(k);
% %             B = B_arr(n);
% % %             t = t_arr(n);
% % %             b = b_arr(k);
% %             M(n,k) = (t*T+b*B-G)/((t-1)*(b-1));
%         else
%             M(n,k) = OS_norm(n,k);
%         end
%     end
% end


% All = sum(OS_norm,2)./sum(OS_v,2);
% 
% [~,I] = sort(All);
% 
% x = [];
% y = [];
% for n = 1:length(All)
%     x = [x ; n*ones(size(OS,2),1)];
%     y = [y M(I(n),:)];
% end
% 
% h = histogram2(x,y',[100 100],'FaceColor','flat');
% h.ShowEmptyBins = 'Off';
% h.DisplayStyle = 'tile';
% 
% 
% ax = gca;
% ax.GridColor = [0.4 0.4 0.4];
% ax.GridLineStyle = '--';
% ax.GridAlpha = 0.5;
% ax.XGrid = 'off';
% ax.YGrid = 'on';
% ax.Layer = 'top';
% view(2)
% colormap(gray);
% c = colorbar;
% c.Label.String = 'Count';
% % c.Label.String = 'Probability';
% title_text = sprintf('All Opinion Scores Ordered by Ascending Mean (%d Ratings)',length(y));
% title(title_text)
% 
% yticks(1:5);
% yticklabels({'Bad', 'Poor', 'Fair', 'Good', 'Excellent'})
% 
% xlabel('File')
% ylabel('Opinion Score')







%These values reflect the reliability of the mean rating for each ratee (file).
verbose = 1;
[ Gqk_ISMD_OS, var_OS ] = ISMD_IRR( OS, verbose );
[ Gqk_ISMD_OS_norm, var_OS_norm ] = ISMD_IRR( OS_norm, verbose );
fid = fopen('log_Anon.txt','a');
fprintf(fid,'Opinion Scores\n');
fprintf(fid,'G = %g, File Variance = %g, Rater Variance = %g, Residual Variance = %g\n', Gqk_ISMD_OS, var_OS(1), var_OS(2), var_OS(3));
fprintf(fid,'Nomalised Opinion Scores\n');
fprintf(fid,'G = %g, File Variance = %g, Rater Variance = %g, Residual Variance = %g\n', Gqk_ISMD_OS_norm, var_OS_norm(1), var_OS_norm(2), var_OS_norm(3));
fclose(fid);
% Most Recent Results
% Opinion Scores
% G = 0.869312, File Variance = 0.65295, Rater Variance = 0.219776, Residual Variance = 0.530691
% Nomalised Opinion Scores
% G = 0.90777, File Variance = 0.780112, Rater Variance = 6.87503e-21, Residual Variance = 0.594323



% [p,tbl,stats] = anova2(OS');
% multcompare(stats)
%Test set

% OS_1 = [3 4 1 ; 2 4 3 ; 5 5 2 ; 4 3 1 ; 3 2 1];
% OS_2 = [3 3 3 ; 3 2 2 ; 2 3 3 ; 4 4 4 ; 2 2 2];
% OS_3 = [9 2 5 8 ; 6 1 3 2 ; 8 4 6 8 ; 7 1 2 6 ; 10 5 6 9 ; 6 2 4 7];
% OS = OS_3;

% Calculate ICC(2,1) based on Speech Enhancement (Loizou), Intraclass
% Correlations: Uses in Assessing Rater Reliability (Shrout and Fleiss) and
% Forming Inferences About Some Intraclass Correlation Coefficients (McGraw
% and Wong)

%MSR is mean square of the ratings across speech samples (Rows)
%MSE is the mean square of the residual (Error)
%MSC is the mean square of the ratings across listeners (Columns)
%k = number of Judges
%n = number of Targets



%Original equations

% n = size(OS,1);
% k = size(OS,2);
% rater_means = sum(OS,1) ./sum(OS~=0,1);
% file_means = sum(OS,2) ./sum(OS~=0,2);
% overall_mean = mean(rater_means);
% MSR = (k/(n-1))*sum((file_means-overall_mean).^2);
% MSC = (n/(k-1))*sum((rater_means-overall_mean).^2);
% MSE = (1/((k-1)*(n-1)))*(sum(sum((OS-overall_mean).^2))-(n-1)*MSR-(k-1)*MSC);
% ICC_21 = (MSR-MSE)/(MSR+(k-1)*MSE+(k/n)*(MSC-MSE));
% ICC_31 = (MSR-MSE)/(MSR+(k-1)*MSE);
% 
% fprintf('ICC_21 = %g, ICC_31 = %g\n',ICC_21, ICC_31)



% BMS = MSB;
% EMS = MSE;
% JMS = MSL;
% 
% ICC_21 = (BMS-EMS)/(BMS+(K-1)*EMS+K(JMS-EMS)/N);
% 
% 
% 
% %Required Responses
% rho = 0.8;
% 
% %Calculate v
% 
% F_j = JMS/EMS;
% rho_hat = ICC_21;
% 
% v_num = (K-1)*(N-1)*(K*rho_hat*F_j+N*(1+(K-1)*rho_hat)-K*rho_hat)^2;
% v_den = (N-1)*K^2*rho_hat^2*F_j^2+(N*(1+(K-1)*rho_hat)-K*rho_hat)*2;
% 
% v = v_num/v_den;
% 
% 
% 
% % rho_l = 0.4;
% m = (rho*(1-rho_l))/(rho_l*(1-rho));
