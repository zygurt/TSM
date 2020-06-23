% Analysing Network output

close all
clear all
clc
addpath('./Functions/');
addpath('../Functions/');
% addpath('../../External/');
make_video = 1;
Eval_folder = 'logs/Eval/To_Test_MeanOS_Folder/';
best_epoch=454  ; %MeanOS
% Eval_folder = 'logs/Eval/To_Test_MeanOS_Raw_Folder/';
% best_epoch=571; %MeanOS Raw
d = rec_filelist(Eval_folder);
d = natsort(d);

vid = VideoWriter('Video/OMOQ_To_Test_MeanOS_Linewidth1.2','MPEG-4');
vid.FrameRate = 25;
vid.Quality = 100;
open(vid)
disp(d)
figure(1)
overall_means = zeros(14,length(d));
epochs = zeros(1,length(d));
for v = 1:length(d)
    fprintf('Frame: %d, file: %s\n',v, char(d(v)));
    eval = Import_Evaluation_CSV(char(d(v)), 2, 6001);

    category = {};
    for n = 1:height(eval)
        name = split(char(eval.RefFile(n)),'/');
        switch char(name(end))
            case 'Alto_Sax_15.wav'
                category{n,1} = 'Solo_Harmonic';
            case 'Ardour_2.wav'
                category{n,1} = 'Complex';
            case 'Brass_and_perc_9.wav'
                category{n,1} = 'Complex';
            case 'Child_4.wav'
                category{n,1} = 'Voice';
            case 'Female_2.wav'
                category{n,1} = 'Voice';
            case 'Female_4.wav'
                category{n,1} = 'Voice';
            case 'Jazz_3.wav'
                category{n,1} = 'Complex';
            case 'Male_16.wav'
                category{n,1} = 'Voice';
            case 'Male_22.wav'
                category{n,1} = 'Voice';
            case 'Male_6.wav'
                category{n,1} = 'Voice';
            case 'Mexican_Flute_02.wav'
                category{n,1} = 'Solo_Harmonic';
            case 'Oboe_piano_1.wav'
                category{n,1} = 'Complex';
            case 'Ocarina_02.wav'
                category{n,1} = 'Solo_Harmonic';
            case 'Rock_4.wav'
                category{n,1} = 'Complex';
            case 'Saxophones_6.wav'
                category{n,1} = 'Complex';
            case 'Solo_flute_2.wav'
                category{n,1} = 'Solo_Harmonic';
            case 'Synth_Bass_2.wav'
                category{n,1} = 'Solo_Percussive';
            case 'Triangle_02.wav'
                category{n,1} = 'Solo_Percussive';
            case 'Woodwinds_4.wav'
                category{n,1} = 'Complex';
            case 'You_mean_this_one.wav'
                category{n,1} = 'Complex';
        end
    end

    eval = [eval category];
    TSM = [0.2257,0.2635,0.3268,0.4444,0.5620,0.6631,0.7641,0.8008,0.8375,0.8742,0.9109,0.9555,1,1.1205,1.241,1.3477,1.4543,1.6272,1.8042,2.1632]; %All Eval Values
    % TSM = eval.TSM(2:6);  %Offset because 145 is before 32
    eval = sortrows(eval,[3 4 1],'ascend');
    eval.OMOS(801:820) = NaN;
    Methods = {'DIPL','ESOLA','Elastique','FESOLA','FuzzyPV','HPTSM','IPL','NMFTSM','PV','Phavorit IPL','Phavorit SPL','SPL','WSOLA','uTVSSubj','uTVS'};
%     lines_spec = {'o-','o-','o-','+-','+-','+-','--','--','--','x--','x--','x--','x--'};
%     points_spec = {'o','o','o','+','+','+','.','.','.','x','x','x','x'};
    grey_lines= {'k-^','k-', 'k--d', 'k.-', 'k--^', 'k-x', 'k-+','k--v', 'k-o', 'k-', 'k-', 'k-', 'k-*', 'k->', 'k-s'};
%     lines= {'-', '-d', '.-', '-^', '-x', '-+','-v', '-o', '-', '-', '-', '-*', '-s'};
    % o+*.xs dv^ are the marker options
    m = 1;
    results = zeros(20,length(TSM),length(Methods)); % (source files,TSM,methods)
    for l = 1:length(Methods)
        for k = 1:length(TSM)
            for n = 1:20 %Number of evaluation source files
                results(n,k,l) = eval.OMOS(m);
                m = m+1;
            end
        end
    end
    
    method_TSM_means = mean(results,1,'omitnan');
    method_means = mean(method_TSM_means(:,[2:12 14:20],:));
    for n = 1:length(method_means)
      fprintf('%s mean: %g\n',char(Methods(n)),method_means(:,:,n))
      overall_means(n,v) = method_means(:,:,n);
    end
%     fprintf('Methods used in Subjective Testing\n')
    % % chosen_methods = [2,3,4,5,6,7,8,12,13];
    chosen_methods = [9,7,13,4,6,15,3,5,8];
%     figure('Position',[1680-500 200 500 350])
    figure(1)
    clf
    hold on
    title_split = split(char(d(v)),'.');
    title_split = split(title_split(1),'_');
    epochs(v) = str2double(char(title_split(end)));
    fprintf('%s\n\n',strcat('Epoch',char(title_split(end))));
    if epochs(v)<best_epoch
      %red
      for n = chosen_methods
%         plot(TSM,method_TSM_means(:,:,n),lines{n}, 'Color',[0.6350, 0.0780, 0.1840])%,'LineWidth',1.2)
        plot(TSM,method_TSM_means(:,:,n),grey_lines{n}, 'Color',[0.6350, 0.0780, 0.1840])%,'LineWidth',1.2)
       % fprintf('%s mean: %g\n',char(Methods(n)),method_means(:,:,n))
      end
    else
      %green
      for n = chosen_methods
        plot(TSM,method_TSM_means(:,:,n),grey_lines{n}, 'Color',[0.4660, 0.6740, 0.1880],'LineWidth',1.2)
       % fprintf('%s mean: %g\n',char(Methods(n)),method_means(:,:,n))
      end
    end
    hold off
    grid on
    title(strcat('Epoch',char(title_split(end))));
    % title('Overall Means for TSM Methods')
%     xlabel('Epoch')
    xlabel('Time-Scale Ratio (\beta)')
    ylabel('OMOS')
%     if epochs(v)>10
%         axis([epochs(v)-10, epochs(v), 0.2, 1.6, 1, 5])
%     end
%     view([-39 37])
    axis([0.2 2.2 1 5])
%     set(gcf, 'Position', get(0, 'Screensize'));
    legend(Methods(chosen_methods),'location','northoutside','NumColumns',3)
    set(gca,...
        'FontSize', 18, ...
        'FontName', 'Times');
    set(gcf, 'Position', [0 0 1920 1080])
        writeVideo(vid,getframe(gcf));
%     print('plots/MATLAB/TIFF/Method_means_455', '-dtiff');
%     print('plots/MATLAB/EPSC/Method_means_455', '-depsc');
%     print('plots/MATLAB/PNG/Method_means_455', '-dpng');


    % %Histogram of TSM against MOS
    % figure('Position',[0 0 500 300])
    % h = histogram2(eval.TSM,eval.OMOS,[20 100],'FaceColor','flat');
    % h.ShowEmptyBins = 'off';
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
    % axis([0.2 1.5 1 5])
    % % colormap(flipud(gray));
    % c = colorbar;
    % c.Label.String = 'Count';
    % xlabel('Time-Scale Ratio')
    % ylabel('OMOS')
    % set(gca,...
    %     'FontSize', 12, ...
    %     'FontName', 'Times');
    % print('plots/MATLAB/TIFF/TSM_OMOS_455', '-dtiff');
    % print('plots/MATLAB/EPSC/TSM_OMOS_455', '-depsc');
    % print('plots/MATLAB/PNG/TSM_OMOS_455', '-dpng');

end
close(vid)

figure('Position',[353 418 1022 462])
plot(epochs,overall_means);
hold on
line([best_epoch best_epoch],[1 5],'Color','red','LineStyle','--')
hold off
title('Method mean at each epoch')
xlabel('Epoch')
ylabel('Mean OMOS')
legend(Methods,'location','eastoutside')
% print('plots/MATLAB/TIFF/Method_means_455', '-dtiff');
print('plots/MATLAB/EPSC/Means_at_Epochs_To_Test_MeanOS', '-depsc');
print('plots/MATLAB/PNG/Means_at_Epochs_To_Test_MeanOS', '-dpng');

% eval = sortrows(eval,[6 3 4],'ascend');
% T = length(TSM);
% num_Complex = 8;
% num_Solo_Harmonic = 4;
% num_Solo_Percussive = 2;
% num_Voice = 6;
% Complex = zeros(num_Complex,T,length(Methods));
% %THIS BIT ISN'T WORKING.
% for n = 1:length(Methods)
%     for k = 1:T
%         ((n-1)*k*T)+(k-1)*num_Complex+1:((n-1)*T)+(k-1)*num_Complex+num_Complex
%         Complex(:,k,n) = eval.OMOS(((n-1)*T)+(k-1)*num_Complex+1:((n-1)*T)+(k-1)*num_Complex+num_Complex);
%         %eval.OMOS(T*num_Complex*(n-1)+1:T*num_Complex*(n-1)+T*num_Complex)
%     end
% end
%
% for n = 1:length(Methods)
%     figure
%     boxplot(Complex(:,:,n),TSM,'colors','k');
%     title(Methods(n))
%     xlabel('Time-Scale Ratio')
%     ylabel('OMOS')
%     V = axis;
%     V(3) = 1;
%     V(4) = 5;
%     axis(V)
%
% end
%
%
%
%
%
% figure
% hold on
% for n = 1:length(Methods)
%     plot(eval.TSM(T*num_Complex*(n-1)+1:T*num_Complex*(n-1)+T*num_Complex),eval.OMOS(T*num_Complex*(n-1)+1:T*num_Complex*(n-1)+T*num_Complex),points_spec{n})
% end
% hold off
% legend(Methods,'location','bestoutside')

% figure%('Position',[1680-500 200 500 300])
% hold on
% for f = 1:size(res,2)
%     plot(res(f).data.Sum_Loss,res(f).data.Mean_PCC,'.')
% end
% hold off
% legend(legend_labels(1:end-1),'location','best')
% xlabel('Sum of Loss')
% ylabel('Mean PCC')
%
% set(gca,...
%     'FontSize', 12, ...
%     'FontName', 'Times');
%
% print('plots/MATLAB/TIFF/Sum_Loss_MeanPCC', '-dtiff');
% print('plots/MATLAB/EPSC/Sum_Loss_MeanPCC', '-depsc');
% print('plots/MATLAB/PNG/Sum_Loss_MeanPCC', '-dpng');
%
% figure%('Position',[1680-500 200 500 300])
% hold on
% for f = 1:size(res,2)
%     plot(res(f).data.Sum_Loss,res(f).data.Diff_PCC,'.')
% end
% hold off
% legend(legend_labels(1:end-1),'location','best')
% xlabel('Sum of Loss')
% ylabel('PCC Difference')
%
% set(gca,...
%     'FontSize', 12, ...
%     'FontName', 'Times');
%
% print('plots/MATLAB/TIFF/Sum_Loss_PCCDifference', '-dtiff');
% print('plots/MATLAB/EPSC/Sum_Loss_PCCDifference', '-depsc');
% print('plots/MATLAB/PNG/Sum_Loss_PCCDifference', '-dpng');
%
% figure%('Position',[1680-500 200 500 300])
% hold on
% for f = 1:size(res,2)
%     [N,EDGES] = histcounts(res(f).data.Sum_Loss,20);
%     plot((EDGES(1:end-1)+EDGES(2:end))/2,N);
% end
% hold off
% legend(legend_labels(1:end-1),'location','best')
% xlabel('Sum of Loss')
% ylabel('Count')
%
% set(gca,...
%     'FontSize', 12, ...
%     'FontName', 'Times');
%
% print('plots/MATLAB/TIFF/Sum_Loss_Hist', '-dtiff');
% print('plots/MATLAB/EPSC/Sum_Loss_Hist', '-depsc');
% print('plots/MATLAB/PNG/Sum_Loss_Hist', '-dpng');
%
% % Plot the MeanPCCs
% figure%('Position',[1680-500 200 500 300])
% hold on
% for f = 1:size(res,2)
%     [N,EDGES] = histcounts(res(f).data.Mean_PCC,10,'Normalization','probability');
%     plot((EDGES(1:end-1)+EDGES(2:end))/2,N);
% end
% [N,EDGES] = histcounts([u.pearson_corr_MeanOS_norm],20,'Normalization','probability'); %'BinWidth',0.05
% plot((EDGES(1:end-1)+EDGES(2:end))/2,N);
%
% ylabel('Probability')
% hold off
% legend(legend_labels,'location','best')
% xlabel('$\rho$','Interpreter','latex')
%
% set(gca,...
%     'FontSize', 12, ...
%     'FontName', 'Times');
%
% print('plots/MATLAB/TIFF/MeanPCC_Hist', '-dtiff');
% print('plots/MATLAB/EPSC/MeanPCC_Hist', '-depsc');
% print('plots/MATLAB/PNG/MeanPCC_Hist', '-dpng');
%
% %Create matrix for anova test
% anova_loss_mat = res(1).data.Sum_Loss;
% for f = 2:size(res,2)
%     anova_loss_mat = [anova_loss_mat res(f).data.Sum_Loss];
% end
% anova1(anova_loss_mat,legend_labels(1:end-1));
% pause(1)
% ylabel('Sum of RMSE Loss')
% xtickangle(30)
%
% set(gca,...
%     'FontSize', 12, ...
%     'FontName', 'Times');
%
% print('plots/MATLAB/TIFF/Anova_Sum_Loss', '-dtiff');
% print('plots/MATLAB/EPSC/Anova_Sum_Loss', '-depsc');
% print('plots/MATLAB/PNG/Anova_Sum_Loss', '-dpng');
%
% anova_pcc_mat = res(1).data.Mean_PCC;
% for f = 2:size(res,2)
%     anova_pcc_mat = [anova_pcc_mat res(f).data.Mean_PCC];
% end
% anova1(anova_pcc_mat,legend_labels(1:end-1));
% pause(1)
% ylabel('Mean PCC')
% xtickangle(30)
%
% set(gca,...
%     'FontSize', 12, ...
%     'FontName', 'Times');
%
% print('plots/MATLAB/TIFF/Anova_MeanPCC', '-dtiff');
% print('plots/MATLAB/EPSC/Anova_MeanPCC', '-depsc');
% print('plots/MATLAB/PNG/Anova_MeanPCC', '-dpng');
%
% anova_distance_mat = res(1).data.Sum_Loss_Diff_PCC_distance;
% for f = 2:size(res,2)
%     anova_distance_mat = [anova_distance_mat res(f).data.Sum_Loss_Diff_PCC_distance];
% end
% anova1(anova_distance_mat,legend_labels(1:end-1));
% pause(1)
% ylabel('Sum_Loss_Diff_PCC_distance Measure')
% xtickangle(30)
%
% set(gca,...
%     'FontSize', 12, ...
%     'FontName', 'Times');
%
% print('plots/MATLAB/TIFF/Anova_Distance', '-dtiff');
% print('plots/MATLAB/EPSC/Anova_Distance', '-depsc');
% print('plots/MATLAB/PNG/Anova_Distance', '-dpng');
% % legend(legend_labels(1:end-1),'location','best')
