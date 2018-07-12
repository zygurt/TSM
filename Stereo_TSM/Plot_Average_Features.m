% Overall average plotting
close all
clear all
clc

addpath('../Functions');

% load confetti_original.mat
% load confetti_results.mat
load subjective_files_results.mat
load subjective_files_original.mat

N=2048; %The larger this value, the smoother the averaged phase will be.
N_f = 2048;
max_xcorr_C = zeros(size(results, 1), size(results(size(results, 1),size(results, 2)).C(:,:), 1), size(results, 2));
max_xcorr_B = zeros(size(results, 1), size(results(size(results, 1),size(results, 2)).B(:,:), 1), size(results, 2));
st_C_dsim = zeros(size(results, 1), size(results(size(results, 1),size(results, 2)).C(:,:), 1), size(results, 2));
st_B_dsim = zeros(size(results, 1), size(results(size(results, 1),size(results, 2)).B(:,:), 1), size(results, 2));

for audio_file = 1:size(results, 1)
    for TSM_ratio = 1:size(results, 2)
        %Stereo Coherence correlation calculation
        t_orig = (1:N:N*length(original(1,audio_file).C(:,:)'))/(original(1,audio_file).Fs*results(audio_file,TSM_ratio).TSM);
        t_TSM = (1:N_f:N_f*length(results(audio_file,TSM_ratio).C(:,:)'))/results(audio_file,TSM_ratio).Fs;
        orig_st_coherence = original(1,audio_file).C(:,:);
        interp_orig_st_coherence = interp1(t_orig, orig_st_coherence, t_TSM);
        interp_orig_st_coherence(isnan(interp_orig_st_coherence)) = 0;
        
        for TSM_method = 1:size(results(audio_file,TSM_ratio).C(:,:), 1)
            A = results(audio_file,TSM_ratio).C(TSM_method,:);
            B = interp_orig_st_coherence;
            cross_corr = normcrosscorr(A, B);
            max_xcorr_C(audio_file,TSM_method,TSM_ratio) = max(cross_corr);
            st_C_dsim(audio_file,TSM_method,TSM_ratio) = d_sim(A, B);
        end
        
        
        %Stereo Centre correlation calculation
        t_orig = (1:N:N*length(original(1,audio_file).B(:,:)'))/(original(1,audio_file).Fs*results(audio_file,TSM_ratio).TSM);
        t_TSM = (1:N_f:N_f*length(results(audio_file,TSM_ratio).B(:,:)'))/results(audio_file,TSM_ratio).Fs;
        orig_st_centre = original(1,audio_file).B(:,:);
        interp_orig_st_centre = interp1(t_orig, orig_st_centre, t_TSM);
        interp_orig_st_centre(isnan(interp_orig_st_centre)) = 0;
        
        for TSM_method = 1:size(results(audio_file,TSM_ratio).B(:,:), 1)
            A = results(audio_file,TSM_ratio).B(TSM_method,:);
            B = interp_orig_st_centre;
            cross_corr = normcrosscorr(A, B);
            max_xcorr_B(audio_file,TSM_method,TSM_ratio) = max(cross_corr);
            st_B_dsim(audio_file,TSM_method,TSM_ratio) = d_sim(A, B);
        end
    end
end

%dissimilarities are organised (file, Method, TSM) as (row, col, Z)
ave_st_coherence = mean(mean(st_C_dsim,3),1);
ave_st_centre = mean(mean(st_B_dsim,3),1);

std_st_coherence = std(mean(st_C_dsim,3));
std_st_centre = std(mean(st_B_dsim,3));

%DO ALL THE PLOTTING
%Plot all of the averages and std dev depending on how many methods have been used
inc = 0.05; %std dev plotting variable
labels = {'Naive PV', 'Bonada PV', 'Altoe PV', 'Proposed PV', 'Proposed Frame PV', 'Naive WSOLA', 'Proposed WSOLA', 'Naive HP', 'Proposed HP',};
num_labels = length(labels);
fig = figure;
bar(1:num_labels,ave_st_coherence, 0.4, 'FaceColor', [0.7 0.7 0.7]);
set(gca, 'FontName', 'Times New Roman')
set(gcf, 'Position', [100, 100, 675, 430]);
set(gca, 'XLim', [0 num_labels+1],...
    'XTick', 1:num_labels,...
    'XTickLabel', labels);
%plot the std dev
for i = 1:size(st_C_dsim,2)
    line([i i], [-std_st_coherence(i) std_st_coherence(i)]+ave_st_coherence(i), 'Color', 'k', 'LineWidth', 1.5);
    line([i-inc i+inc], [std_st_coherence(i) std_st_coherence(i)]+ave_st_coherence(i), 'Color', 'k', 'LineWidth', 1.5);
    line([i-inc i+inc], -1*[std_st_coherence(i) std_st_coherence(i)]+ave_st_coherence(i), 'Color', 'k', 'LineWidth', 1.5);
end
str = sprintf('Mean Stereo Phase Coherence Dissimilarity');
title(str, 'FontWeight', 'Normal')
xlabel('Stereo Algorithm');
ylabel('L2 Norm');
xtickangle(30);
% print(fig, ['Plots/Mean_Features' str ],'-depsc', '-r0')
% print(fig, ['Plots/Mean_Features' str ],'-dpng', '-r0')

fig = figure;
bar(1:num_labels,ave_st_centre, 0.4, 'FaceColor', [0.7 0.7 0.7])
set(gca, 'FontName', 'Times New Roman')
set(gcf, 'Position', [100, 100, 675, 430]);
set(gca, 'XLim', [0 num_labels+1],...
    'XTick', 1:num_labels,...
    'XTickLabel', labels);
%plot the std dev
for i = 1:size(st_C_dsim,2)
    line([i i], [-std_st_centre(i) std_st_centre(i)]+ave_st_centre(i), 'Color', 'k', 'LineWidth', 1.5);
    line([i-inc i+inc], [std_st_centre(i) std_st_centre(i)]+ave_st_centre(i), 'Color', 'k', 'LineWidth', 1.5);
    line([i-inc i+inc], -1*[std_st_centre(i) std_st_centre(i)]+ave_st_centre(i), 'Color', 'k', 'LineWidth', 1.5);
end

str = sprintf('Mean Stereo Balance Dissimilarity');
title(str, 'FontWeight', 'Normal')
xlabel('Stereo Algorithm');
ylabel('L2 Norm');
xtickangle(30);
% print(fig, ['Plots/Mean_Features' str ],'-depsc', '-r0')
% print(fig, ['Plots/Mean_Features' str ],'-dpng', '-r0')

disp('Feature Averaging and Plotting Complete')