close all
clear all
%clc
addpath('../Functions/');

recalculate_results = 1;

TSM = [38, 44, 53, 65, 78, 82, 99, 138, 166, 192];
tic
if recalculate_results == 1
    fprintf('Recalculating results\n');
    results_path = 'Results_Anon/';
    audio_path = 'Sets/';
    %Create a list of files in the results_path
    res_filelist = rec_filelist(results_path);
    
    %Create master list of files (This should probably be replaced with
    %loading a file and a check to see if the file exists)
    audio_filelist = rec_filelist(audio_path);
    
    %Account for Windows slash file structure
    for n = 1:length(audio_filelist)
        audio_filelist(n) = strrep(audio_filelist(n),'\','/');
    end
    
    for f = 1:size(audio_filelist,1)
        a(f).name = char(audio_filelist(f));
        name_split = split(a(f).name,'_');
        a(f).TSM = str2double(char(name_split(end-1)));
        a(f).method = char(name_split(end-2));
        a(f).MOS = [];
    end
    %Add the responses to the master list
    fprintf('Creating master list of file ratings\n')
    for f = 1:size(res_filelist,1)
        r = load(char(res_filelist(f)));
        fprintf('%s: Sets ',r.user_data.name);
        for s = 1:size(r.user_data.files,2)
            if isfield(r.user_data.files(s).file,'MOS')
                fprintf('%d, ',s);
                for n = 1:size(r.user_data.files(s).file,2)
                    %Find the file name position in a
                    match = 0;
                    an = 1;
                    while ~match
                        match=strcmp(strrep(a(an).name,'\','/'),r.user_data.files(s).file(n).name);
                        an = an+1;
                    end
                    %Remove Final an addition
                    an = an-1;
                    %Store the MOS in a
                    a(an).MOS = [a(an).MOS r.user_data.files(s).file(n).MOS];
                end
            end
        end
        fprintf('\n');
    end
    
    %Calculate the Mean MOS for each file
    for n = 1:size(a,2)
        a(n).mean_MOS = mean(a(n).MOS);
        %         a(n).geomean_MOS = geomean(a(n).MOS);
        a(n).std_MOS = std(a(n).MOS);
        a(n).num_responses = length(a(n).MOS);
    end
    fprintf('Normalising\n');
    %Normalisation processing
    a(1).MOS_norm = [];
    u(1).name = [];
    u(1).key = [];
    u(1).filename = [];
    u(1).mean_abs_diff_norm = [];
    u(1).mean_abs_diff_mean = [];
    u(1).mean_abs_diff_median = [];
    u(1).std_abs_diff_mean = [];
    u(1).std_abs_diff_median = [];
    u(1).RMSE = [];
    u(1).num_files = [];
    u(1).mean_diff = [];
    u(1).pearson_corr_mean = [];
    u(1).pearson_corr_median = [];
    u(1).total_time_sec = [];
    u(1).total_time_min = [];
    for f = 1:size(res_filelist,1)
        r = load(char(res_filelist(f)));
        u(f).name = r.user_data.name;
        u(f).filename = char(res_filelist(f));
        if isfield(r.user_data,'age')
            u(f).age = str2double(r.user_data.age);
        end
        if isfield(r.user_data,'expert')
            if strcmp(r.user_data.expert,'Yes')
                u(f).expert = 1;
            else
                u(f).expert = 0;
            end
        end
        if isfield(r.user_data,'hearing_check_binary')
            u(f).hearing = r.user_data.hearing_check_binary;
        end
        if isfield(r.user_data,'key')
            u(f).key = r.user_data.key;
        else
            u(f).key = sprintf('MATLAB_%s',r.user_data.name);
        end
        
        %         fprintf('\nNormalising %s: ',r.user_data.name);
        %Create a set list of responses
        for s = 1:size(r.user_data.files,2)
            if isfield(r.user_data.files(s).file,'MOS')
                %                 fprintf('%d, ',s);
                %allocate normalisation set structure here
                set = [];
                for n = 1:size(r.user_data.files(s).file,2)
                    %Find the file name position in a
                    match = 0;
                    an = 1;
                    while ~match
                        match=strcmp(strrep(a(an).name,'\','/'),r.user_data.files(s).file(n).name);
                        an = an+1;
                    end
                    %Remove Final an addition
                    an = an-1;
                    %Add the scores to this set
                    set(n).MOS = a(an).MOS;
                end
                
                %Calculate normalisation parameters
                xs = mean([set.MOS]);
                ss = std([set.MOS]);
                xsi = mean([r.user_data.files(s).file.MOS]);
                ssi = std([r.user_data.files(s).file.MOS]);
                abs_diff_norm_mean=[];
                abs_diff_mean=[];
                abs_diff_median=[];
                mean_diff = [];
                %Calculate and Store the normalised result
                for n = 1:size(r.user_data.files(s).file,2)
                    match = 0;
                    an = 1;
                    while ~match
                        match=strcmp(strrep(a(an).name,'\','/'),r.user_data.files(s).file(n).name);
                        an = an+1;
                    end
                    %Remove Final an addition
                    an = an-1;
                    %Store the MOS in a
                    xi = r.user_data.files(s).file(n).MOS;
                    Zi = (xi-xsi)*ss/ssi+xs;
                    %                     if limit_results == 1
                    %                      if Zi<1
                    %                          Zi=1;
                    %                      end
                    %                      if Zi>5
                    %                          Zi=5;
                    %                      end
                    %                     end
                    r.user_data.files(s).file(n).MOS_norm = Zi;
                    a(an).MOS_norm = [a(an).MOS_norm Zi];
                    abs_diff_norm_mean = [abs_diff_norm_mean abs(Zi-mean(a(an).MOS))];
                    abs_diff_mean = [abs_diff_mean abs(xi-mean(a(an).MOS))];
                    abs_diff_median = [abs_diff_median abs(xi-median(a(an).MOS))];
                    mean_diff = [mean_diff Zi-mean(a(an).MOS)];
                end
                u(f).mean_abs_diff_norm = [u(f).mean_abs_diff_norm mean(abs_diff_norm_mean)];
                u(f).mean_abs_diff_mean = [u(f).mean_abs_diff_mean mean(abs_diff_mean)];
                u(f).RMSE = [u(f).RMSE sqrt(mean(abs_diff_mean.^2))];
                u(f).mean_abs_diff_median = [u(f).mean_abs_diff_median mean(abs_diff_median)];
                u(f).std_abs_diff_mean = [u(f).std_abs_diff_mean std(abs_diff_mean)];
                u(f).std_abs_diff_median = [u(f).std_abs_diff_median std(abs_diff_median)];
                u(f).num_files = [u(f).num_files length(abs_diff_norm_mean)];
                u(f).mean_diff = [u(f).mean_diff mean(mean_diff)];
                
                %Compute the correlation for the session
                set_MeanOS_Scores = zeros(length(set),1);
                set_MedianOS_Scores = zeros(length(set),1);
                
                for n = 1:length(set)
                    set_MeanOS_Scores(n) = mean(set(n).MOS);
                    set_MedianOS_Scores(n) = median(set(n).MOS);
                end
                subset_MeanOS_Scores = [];
                subset_MedianOS_Scores = [];
                for n = 1:length(r.user_data.files(s).file)
                    if ~isempty(r.user_data.files(s).file(n).MOS)
                        subset_MeanOS_Scores = [subset_MeanOS_Scores set_MeanOS_Scores(n)];
                        subset_MedianOS_Scores = [subset_MedianOS_Scores set_MedianOS_Scores(n)];
                    end
                end
                u(f).pearson_corr_mean = [u(f).pearson_corr_mean corr(subset_MeanOS_Scores', [r.user_data.files(s).file.MOS]')];
                u(f).pearson_corr_median = [u(f).pearson_corr_median corr(subset_MedianOS_Scores', [r.user_data.files(s).file.MOS]')];
%                 if (sum([r.user_data.files(s).file.time_taken]) <0)
%                     disp('Total Time is less than zero')
%                 end
                time_taken = [r.user_data.files(s).file.time_taken];
                u(f).total_time_sec = [u(f).total_time_sec sum(time_taken(time_taken>0))];
                u(f).total_time_min = u(f).total_time_sec/60;
                fprintf('%s, set %d, MAD=%.3f, Norm MAD=%.3f, STD=%.3f, STD_norm(pre_second_MAD_Calc)=%.3f\n',r.user_data.name, s, mean(abs_diff_mean), mean(abs_diff_norm_mean), std(abs_diff_mean), std(abs_diff_norm_mean));
                %                 fprintf('%s, set %d, MAD=%.3f, MD=%.3f\n',r.user_data.name, s, mean(abs_diff), mean(diff));
                
               
            end
        end
        %         fprintf('\n');
        
                        %Save the new user data
%                         r = load(char(res_filelist(f)));
        user_data = r.user_data;
       save(char(res_filelist(f)),'user_data');
    end
    
    %Calculate the Mean MOS for each file
    for n = 1:size(a,2)
        a(n).mean_MOS_norm = mean(a(n).MOS_norm);
        %         a(n).geomean_MOS_norm = geomean(a(n).MOS_norm);
        a(n).std_MOS_norm = std(a(n).MOS_norm);
        %Limit the final mean
        if a(n).mean_MOS_norm < 1
            a(n).mean_MOS_norm = 1;
        end
        if a(n).mean_MOS_norm > 5
            a(n).mean_MOS_norm = 5;
        end
        
        %Calculate the Median for each file
        a(n).median_MOS_norm = median(a(n).MOS_norm); 
        %Limit the final mean
        if a(n).median_MOS_norm < 1
            a(n).median_MOS_norm = 1;
        end
        if a(n).median_MOS_norm > 5
            a(n).median_MOS_norm = 5;
        end
        
        
    end
    
    %Calculate the MAD for the normalised results.
    u(1).mean_abs_diff_norm_mean = [];
    u(1).mean_abs_diff_norm_median = [];
    u(1).mean_diff_norm = [];
    u(1).RMSE_norm = [];
    u(1).std_abs_diff_norm_mean = [];
    u(1).std_abs_diff_norm_median = [];
    u(1).pearson_corr_MeanOS_norm = [];
    u(1).pearson_corr_MedianOS_norm = [];
    for f = 1:size(res_filelist,1)
        r = load(char(res_filelist(f)));
        fprintf('%s\n',char(res_filelist(f)));
        %Create a set list of absolute Differences
        for s = 1:size(r.user_data.files,2)
            abs_diff_norm_mean=[];
            abs_diff_norm_median=[];
            mean_diff_norm = [];
            MeanOS_Set = [];
            MedianOS_Set = [];
            user_MOS_norm_Set = [];
            if isfield(r.user_data.files(s).file,'MOS')
                %allocate normalisation set structure here
                
                for n = 1:size(r.user_data.files(s).file,2)
                    %Find the file name position in a
                    match = 0;
                    an = 1;
                    while ~match
                        match=strcmp(strrep(a(an).name,'\','/'),r.user_data.files(s).file(n).name);
                        an = an+1;
                    end
                    %Remove Final an addition
                    an = an-1;
                    %If isn't empty
                    if ~isempty(r.user_data.files(s).file(n).MOS_norm)
                        abs_diff_norm_mean = [abs_diff_norm_mean abs(a(an).mean_MOS_norm-r.user_data.files(s).file(n).MOS_norm)]; %This has to stay undeclared to account for incomplete MATLAB tests.
                        abs_diff_norm_median = [abs_diff_norm_median abs(a(an).median_MOS_norm-r.user_data.files(s).file(n).MOS_norm)]; %This has to stay undeclared to account for incomplete MATLAB tests.
                        mean_diff_norm = [mean_diff_norm a(an).mean_MOS_norm-r.user_data.files(s).file(n).MOS_norm]; %This has to stay undeclared to account for incomplete MATLAB tests.
                        MeanOS_Set = [MeanOS_Set a(an).mean_MOS_norm];
                        MedianOS_Set = [MedianOS_Set a(an).median_MOS_norm];
                        user_MOS_norm_Set = [user_MOS_norm_Set r.user_data.files(s).file(n).MOS_norm];
                    end
                end
                u(f).mean_abs_diff_norm_mean = [u(f).mean_abs_diff_norm_mean mean(abs_diff_norm_mean)];
                u(f).mean_abs_diff_norm_median = [u(f).mean_abs_diff_norm_median mean(abs_diff_norm_median)];
                u(f).mean_diff_norm = [u(f).mean_diff_norm mean(mean_diff_norm)];
                u(f).RMSE_norm = [u(f).RMSE_norm sqrt(mean(mean_diff_norm.^2))];
                u(f).std_abs_diff_norm_mean = [u(f).std_abs_diff_norm_mean std(abs_diff_norm_mean)];
                u(f).std_abs_diff_norm_median = [u(f).std_abs_diff_norm_median std(abs_diff_norm_median)];
                
                % Computing Pearson Correlation Coefficients

                u(f).pearson_corr_MeanOS_norm = [u(f).pearson_corr_MeanOS_norm corr(MeanOS_Set', user_MOS_norm_Set')];
                u(f).pearson_corr_MedianOS_norm = [u(f).pearson_corr_MedianOS_norm corr(MedianOS_Set', user_MOS_norm_Set')];
                
                
            end
            
        end
        
    end
    
    
    
    
    %     if limit_results == 0
    %         save('Results_v4_MOS_limited.mat','a','res_filelist','u');
    %     else
    %         save('Results_v4_OS_limited.mat','a','res_filelist','u');
    %     end
    
    %Calculate the mean Mean Absolute Difference per user.
    for f = 1:length(u)
        u(f).MMAD_norm_mean_new = mean(u(f).mean_abs_diff_norm_mean);
        u(f).MMAD_norm_median_new = mean(u(f).mean_abs_diff_norm_median);
        u(f).MMAD_norm = mean(u(f).mean_abs_diff_norm);
        u(f).MMAD = mean(u(f).mean_abs_diff_mean);
        u(f).MMAD_median = mean(u(f).mean_abs_diff_median);
        u(f).MSTD_mean = mean(u(f).std_abs_diff_mean);
        u(f).MSTD_median = mean(u(f).std_abs_diff_median);
        u(f).MSTD_mean_new = mean(u(f).std_abs_diff_norm_mean);
        u(f).MSTD_median_new = mean(u(f).std_abs_diff_norm_median);
        
    end
    
    
        save('Results_Anon_No_Outliers.mat','a','res_filelist','u');

%     save('Results_v8_2_MAD_STD_2nd_outliers_removed.mat','a','res_filelist','u');
%     save('Results_v8_MAD_STD_1st_outliers_removed.mat','a','res_filelist','u');
%     save('Results_v8.mat','a','res_filelist','u');
else
    load('Results_Anon_No_Outliers.mat');
end
fid = fopen('log_Anon.txt','a');
%Calculate the mean standard deviation prior to normalisation
A = [a(1:5280).std_MOS];
%Remove 0 values from calculation as not every file has multiple responses
mean_std = mean(A(A~=0));
fprintf(fid,'The mean standard deviation (TSMDB) prior to normalisation is %g\n',mean_std);

%Calculate the mean standard deviation prior to normalisation
A_norm = [a(1:5280).std_MOS_norm];
%Remove 0 values from calculation as not every file has multiple responses
mean_std_norm = mean(A_norm(A_norm~=0));
fprintf(fid,'The mean standard deviation (TSMDB) after normalisation is %g\n',mean_std_norm);


%Calculate the mean standard deviation prior to normalisation
A = [a.std_MOS];
%Remove 0 values from calculation as not every file has multiple responses
mean_std = mean(A(A~=0));
fprintf(fid,'The mean standard deviation (ALL) prior to normalisation is %g\n',mean_std);

%Calculate the mean standard deviation prior to normalisation
A_norm = [a.std_MOS_norm];
%Remove 0 values from calculation as not every file has multiple responses
mean_std_norm = mean(A_norm(A_norm~=0));
fprintf(fid,'The mean standard deviation (ALL) after normalisation is %g\n',mean_std_norm);

response_count = zeros(1,max([a.num_responses]));
for n = 1:length(a)
    response_count(a(n).num_responses) = response_count(a(n).num_responses)+1;
end

for n = 1:length(response_count)
    fprintf(fid,'%d files with %d responses\n',response_count(n),n);
end

fprintf(fid,'%d total responses\n',length([a.MOS]));
fprintf(fid,'Average Age = %g\n',mean([u.age]));
t = toc;
fprintf(fid,'Compile Time = %g minutes\n',t/60);


resp = [a.num_responses];
one = length(resp(resp==1));
two = length(resp(resp==2));
three = length(resp(resp==3));
four = length(resp(resp==4));
five = length(resp(resp==5));
six = length(resp(resp==6));

remaining = one*6 + two*5 + three*4 + four*3 + five*2 + six;
fprintf(fid,'%d ratings to go! %g sets.\n',remaining, remaining/80);
fclose(fid);
beep