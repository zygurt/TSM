% Plot the Features and log features

close all
clear all
clc

%% Feature Correlation Plot

testing_var = 0;

% figure %Maximize this figure before continuing
% file_to_load = 'MOVs_Final_Interp_to_test_with_source.mat';  %TSMDB Add 88 to 5280 below
file_to_load = 'Features/MOVs_20200620Interpolate_to_test.mat';  %TSMDB
num_files = 5280;
% file_to_load = 'Features/MOVs_20200620ToTest_Incl_Source.mat';  %TSMDB
% num_files = 5520;
MOV_start = 6;
% file_to_load = 'MOVs_20191123Interpolate_to_test.mat';
% MOV_start = 5;
load(file_to_load)
M = MOVs;
% file_to_load = 'MOVs_20200203Interpolate_to_test.mat'; %Eval
% load(file_to_load)
% M = [M ; MOVs];
New_Figure = 3;
O = OMOV';
n = size(MOVs,1);
chosen_features = 1:size(M,2);
% [~,OMOV_I] = sort(OMOV(6:end));
% chosen_features = [1:5,OMOV_I+5];
% chosen_features = [1,4,7:32,37,38];
chosen_OMOV = OMOV(chosen_features);
% M(isinf(M(:,18)),18) = 80; %Remove INF values from old SER calculation
small_MOV = M(:,chosen_features);

for n = 1:size(chosen_OMOV,2)
    chosen_OMOV(n) = {strrep(char(chosen_OMOV(n)),'_',' ')};
end

%Combination Modifications

% % chosen_OMOV(10) = {'BandwidthRefB'};
% % chosen_OMOV(11) = {'BandwidthTestB'};
% chosen_OMOV(12) = {'Interp BandwidthTestB New'};
% % chosen_OMOV(18) = {'RmsModDiffA'};
% % chosen_OMOV(19) = {'RmsNoiseLoudAsymA'};
% % chosen_OMOV(20) = {'AvgLinDistA'};
% % chosen_OMOV(21) = {'SegmentalNMRB'};
% % 
% chosen_OMOV(22) = {'Interp DM'};
% 
% chosen_OMOV(24) = {'\DeltaP'};
% chosen_OMOV(25) = {'TrRat'};
% chosen_OMOV(26) = {'HPSTrRat'};
% % 
% chosen_OMOV(27) = {'MPhNW'};
% chosen_OMOV(28) = {'SPhNW'};
% chosen_OMOV(29) = {'MPhMW'};
% chosen_OMOV(30) = {'SPhMW'};
% chosen_OMOV(31) = {'SSMAD'};
% chosen_OMOV(32) = {'SSMD'};
% 
% chosen_OMOV(42) = {'Anchor DM'};

% %Interpolate To Test Feature Name Modifications
chosen_OMOV(5) = {'Time-Scale Ratio (\beta)'};
chosen_OMOV(12) = {'BandwidthTestB New'};
chosen_OMOV(24) = {'\DeltaP'};
chosen_OMOV(25) = {'TrRat'};
chosen_OMOV(26) = {'HPSTrRat'};
chosen_OMOV(27) = {'MPhNW'};
chosen_OMOV(28) = {'SPhNW'};
chosen_OMOV(29) = {'MPhMW'};
chosen_OMOV(30) = {'SPhMW'};
chosen_OMOV(31) = {'SSMAD'};
chosen_OMOV(32) = {'SSMD'};

%Must be old adjustments
% chosen_OMOV(2) = {'MeanOS Raw'};
% chosen_OMOV(10) = {'BandwidthTestNew'};
% chosen_OMOV(22) = {'\DeltaP'};
% chosen_OMOV(23) = {'TrRat'};
% chosen_OMOV(24) = {'HPSTrRat'};
% chosen_OMOV(25) = {'MPhNW'};
% chosen_OMOV(26) = {'SPhNW'};
% chosen_OMOV(27) = {'MPhMW'};
% chosen_OMOV(28) = {'SPhMW'};
% chosen_OMOV(29) = {'SSMAD'};
% chosen_OMOV(30) = {'SSMD'};

% % Create log10 features
% for k = 4:size(small_MOV,2)
%     if min(small_MOV(:,k))>0
%         small_MOV = [small_MOV, 10*log10(small_MOV(:,k))];
%         chosen_OMOV(end+1) = strcat('log10(', chosen_OMOV(k), ')');
%     end
% end

% M_mean = mean(small_MOV(1:5280,MOV_start:end));
% M_std = std(small_MOV(1:5280,MOV_start:end));
% small_MOV(:,MOV_start:end) = (small_MOV(:,MOV_start:end)-M_mean)./M_std;

M_min = min(small_MOV(1:num_files,MOV_start:end));
M_max = max(small_MOV(1:num_files,MOV_start:end));
% M_min = min(small_MOV(88:end,MOV_start:end));
% M_max = max(small_MOV(88:end,MOV_start:end));
small_MOV(:,MOV_start:end) = (small_MOV(:,MOV_start:end)-M_min)./(M_max-M_min);
[~,I] = sort(small_MOV(:,5));
s = small_MOV(I,:);

feat_corr_slow = abs(corr(s(1:3876,:)));
% figure('Position',[1700 200 900 900])
% imshow(feat_corr_slow,'InitialMagnification','fit','colormap',parula)
% title('Slower')
% add_labels(chosen_OMOV)
% % set(gcf, 'Position', get(0, 'Screensize'));
% f = sprintf('Plots/Chosen_Feat_%s_Slower_Correlation',file_to_load(1:end-4));
% % print([f '.png'],'-dpng')
% % print([f '.eps'],'-depsc')


feat_corr_fast = abs(corr(s(3877:end,:)));
% figure('Position',[1700 200 900 900])
% imshow(feat_corr_fast,'InitialMagnification','fit','colormap',parula)
% title('Faster')
% add_labels(chosen_OMOV)
% % set(gcf, 'Position', get(0, 'Screensize'));
% f = sprintf('Plots/Chosen_Feat_%s_Faster_Correlation',file_to_load(1:end-4));
% % print([f '.png'],'-dpng')
% % print([f '.eps'],'-depsc')


feat_corr_split = 0.5.*(feat_corr_slow+feat_corr_fast);
figure('Position',[0 0 880 600])
% set(groot,'defaultAxesTickLabelInterpreter','latex');  
imshow(feat_corr_split,'InitialMagnification','fit','colormap',parula)
% title('Average')
add_labels(chosen_OMOV)
xticks([])
% set(gcf, 'Position', get(0, 'Screensize'));
% f = sprintf('/Plots/Combination_Feat_%s_AvgSplit_Correlation',file_to_load(1:end-4));
print('Plots/PNG/Feature_corr.png','-dpng')
print('Plots/EPSC/Feature_corr.eps','-depsc')

% feature_corr = (corr(small_MOV)+1)/2;
% feature_corr_abs = abs(corr(small_MOV));
% figure('Position',[1700 200 900 900])
% imshow(feature_corr_abs,'InitialMagnification','fit','colormap',parula)
% title('Overall')
% add_labels(chosen_OMOV)
% % set(gcf, 'Position', get(0, 'Screensize'));
% f = sprintf('Plots/Chosen_Feat_%s_Correlation',file_to_load(1:end-4));
% % print([f '.png'],'-dpng')
% % print([f '.eps'],'-depsc')


%% Animated Feature Plots
load('../Subjective_Testing/TSM_MOS_Scores.mat');
% load('MOVs_20200224Interpolate_to_test.mat'); %New Phasiness
% file_to_load = 'MOVs_20191123Interpolate_to_test.mat';  %TSMDB
% file_to_load = 'MOVs_20200227Interpolate_to_test.mat';
% file_to_load = 'TransientMOVs_20200303Interpolate_to_test.mat';
% load(file_to_load)
MOVs = small_MOV;
for k = 6:size(MOVs,2)
    feat = k;
%     if min(MOVs(:,feat))>0
%         figure;
%         %     f.Position = [1913 157 1280 720];
%         hold on
%         %Plot to set legend entries correctly
%         plot3(nan,nan,nan,'x','Color',[0,0.447,0.741])
%         plot3(nan,nan,nan,'x','Color',[0.85,0.325,0.098])
%         plot3(nan,nan,nan,'+','Color',[0.929,0.694,0.125])
%         plot3(nan,nan,nan,'+','Color',[0.494,0.184,0.556])
%         plot3(nan,nan,nan,'*','Color',[0.466,0.674,0.188])
%         plot3(nan,nan,nan,'*','Color',[0.301,0.745,0.933])
%         plot3(nan,nan,nan,'o','Color',[0,0.447,0.741])
%         plot3(nan,nan,nan,'o','Color',[0.85,0.325,0.098])
%         plot3(nan,nan,nan,'o','Color',[0.929,0.694,0.125])
%         for n = 1:size(MOVs,1)
%             switch data(n).method
%                 case 'PV'
%                     plot3(MOVs(n,1),MOVs(n,3),10*log10(MOVs(n,feat)),'x','Color',[0,0.447,0.741])
%                 case 'IPL'
%                     plot3(MOVs(n,1),MOVs(n,3),10*log10(MOVs(n,feat)),'x','Color',[0.85,0.325,0.098])
%                 case 'WSOLA'
%                     plot3(MOVs(n,1),MOVs(n,3),10*log10(MOVs(n,feat)),'+','Color',[0.929,0.694,0.125])
%                 case 'FESOLA'
%                     plot3(MOVs(n,1),MOVs(n,3),10*log10(MOVs(n,feat)),'+','Color',[0.494,0.184,0.556])
%                 case 'HPTSM'
%                     plot3(MOVs(n,1),MOVs(n,3),10*log10(MOVs(n,feat)),'*','Color',[0.466,0.674,0.188])
%                 case 'uTVS'
%                     plot3(MOVs(n,1),MOVs(n,3),10*log10(MOVs(n,feat)),'*','Color',[0.301,0.745,0.933])
%                 case 'Elastique'
%                     plot3(MOVs(n,1),MOVs(n,3),10*log10(MOVs(n,feat)),'o','Color',[0,0.447,0.741])
%                 case 'FuzzyTSM'
%                     plot3(MOVs(n,1),MOVs(n,3),10*log10(MOVs(n,feat)),'o','Color',[0.85,0.325,0.098])
%                 case 'NMFTSM'
%                     plot3(MOVs(n,1),MOVs(n,3),10*log10(MOVs(n,feat)),'o','Color',[0.929,0.694,0.125])
%                 otherwise
%                     disp('Unknown Method')
%             end
%         end
%         hold off
%         xlabel('MeanOS')
%         ylabel('TSM')
%         zlabel(['10log_1_0(' strrep(char(chosen_OMOV(feat)),'_','\_')])
%         title(strrep(char(chosen_OMOV(feat)),'_','\_'))
%         axis([1 5 0.2 2 min(10*log10(MOVs(:,feat))) max(10*log10(MOVs(:,feat)))])
%         legend({'PV', 'IPL', 'WSOLA', 'FESOLA', 'HPTSM', 'uTVS', 'Elastique', 'FuzzyTSM', 'NMFTSM'})
%         grid on
%         v = VideoWriter(['./Plots/Video/10log10_' char(OMOV(feat))]);
%         v.FrameRate = 25;
%         open(v)
%         w = 0.5*(1 - cos(2*pi*(0:100-1)'/(100-1)));
%         azum = zeros(size(w'));
%         for n = 1:length(w)
%             azum(n) = sum(w(1:n));
%         end
%         azum = [azum fliplr(azum)];
%         azum = 90*azum/max(azum);
%         w = 0.5*(1 - cos(2*pi*(0:50-1)'/(50-1)));
%         el = zeros(size(w'));
%         for n = 1:length(w)
%             el(n) = sum(w(1:n));
%         end
%         el = [el fliplr(el)];
%         el = 40*el/max(el);
%         el = [el el];
% 
%         set(gcf, 'Position', get(0, 'Screensize'));
%         for n = 1:length(azum)
%             view([azum(n) el(n)])
%             %         pause(1/25)
%             %         disp(f.Position)
%             %         fprintf('Setting Position\n')
%             %         f.Position = [1913 157 1280 720];
%             %         disp(f.Position)
%             %         pause(1/25)
%             %         fprintf('Writing Frame\n')
%             writeVideo(v,getframe(gcf));
% 
%         end
%         close(v)
%         close(gcf)
%     end
    f = figure(2);
    %     f.Position = [1913 157 1280 720];
    hold on
    %Plot to set legend entries correctly
        plot3(nan,nan,nan,'x','Color',[0,0.447,0.741])
        plot3(nan,nan,nan,'x','Color',[0.85,0.325,0.098])
        plot3(nan,nan,nan,'+','Color',[0.929,0.694,0.125])
        plot3(nan,nan,nan,'+','Color',[0.494,0.184,0.556])
        plot3(nan,nan,nan,'*','Color',[0.466,0.674,0.188])
        plot3(nan,nan,nan,'*','Color',[0.301,0.745,0.933])
        plot3(nan,nan,nan,'o','Color',[0,0.447,0.741])
        plot3(nan,nan,nan,'o','Color',[0.85,0.325,0.098])
        plot3(nan,nan,nan,'o','Color',[0.929,0.694,0.125])
    for n = 1:size(MOVs,1)
        switch data(n).method
            case 'PV'
                plot3(MOVs(n,1),MOVs(n,5),MOVs(n,feat),'x','Color',[0,0.447,0.741])
            case 'IPL'
                plot3(MOVs(n,1),MOVs(n,5),MOVs(n,feat),'x','Color',[0.85,0.325,0.098])
            case 'WSOLA'
                plot3(MOVs(n,1),MOVs(n,5),MOVs(n,feat),'+','Color',[0.929,0.694,0.125])
            case 'FESOLA'
                plot3(MOVs(n,1),MOVs(n,5),MOVs(n,feat),'+','Color',[0.494,0.184,0.556])
            case 'HPTSM'
                plot3(MOVs(n,1),MOVs(n,5),MOVs(n,feat),'*','Color',[0.466,0.674,0.188])
            case 'uTVS'
                plot3(MOVs(n,1),MOVs(n,5),MOVs(n,feat),'*','Color',[0.301,0.745,0.933])
            case 'Elastique'
                plot3(MOVs(n,1),MOVs(n,5),MOVs(n,feat),'o','Color',[0,0.447,0.741])
            case 'FuzzyTSM'
                plot3(MOVs(n,1),MOVs(n,5),MOVs(n,feat),'o','Color',[0.85,0.325,0.098])
            case 'NMFTSM'
                plot3(MOVs(n,1),MOVs(n,5),MOVs(n,feat),'o','Color',[0.929,0.694,0.125])
            otherwise
                disp('Unknown Method')
        end
    end
    hold off
    xlabel('MeanOS')
    ylabel('TSM')
    zlabel(strrep(char(chosen_OMOV(feat)),'_','\_'))
    title(strrep(char(chosen_OMOV(feat)),'_','\_'))
    axis([1 5 0.2 2 min(MOVs(:,feat)) max(MOVs(:,feat))])
    legend({'PV', 'IPL', 'WSOLA', 'FESOLA', 'HPTSM', 'uTVS', 'Elastique', 'FuzzyTSM', 'NMFTSM'},'Location','NorthEastOutside')
    grid on
    set(gca,...
    'FontSize', 18, ...
    'FontName', 'Times');
    v = VideoWriter(['./Plots/Video/' char(chosen_OMOV(feat))]);
    v.FrameRate = 25;
    open(v)
    w = 0.5*(1 - cos(2*pi*(0:100-1)'/(100-1)));
    azum = zeros(size(w'));
    for n = 1:length(w)
        azum(n) = sum(w(1:n));
    end
    azum = [azum fliplr(azum)];
    azum = 90*azum/max(azum);
    w = 0.5*(1 - cos(2*pi*(0:50-1)'/(50-1)));
    el = zeros(size(w'));
    for n = 1:length(w)
        el(n) = sum(w(1:n));
    end
    el = [el fliplr(el)];
    el = 40*el/max(el);
    el = [el el];

    set(gcf, 'Position', get(0, 'Screensize'));
    for n = 1:length(azum)
        view([azum(n) el(n)])
        %         pause(1/25)
        %         disp(f.Position)
        %         fprintf('Setting Position\n')
        %         f.Position = [1913 157 1280 720];
        %         disp(f.Position)
        %         pause(1/25)
        %         fprintf('Writing Frame\n')
        writeVideo(v,getframe(gcf));

    end
    close(v)
    close(gcf)
end



%% Some other plot I need to look back at
%
%
%
% %Plot just the chosen phasiness features
% figure('Position',[0 0 1200 700])
% subp = 1;
% n=5520;
% for k = 22:27
%     subplot(2,3,subp)
%     h = histogram2(small_MOV(1:n,1),log10(small_MOV(1:n,k)),[40 40],'FaceColor','flat');
%     h.DisplayStyle = 'tile';
%     h.EdgeAlpha = 0;
%     view(2)
%     ax = gca;
%     ax.XGrid = 'off';
%     ax.YGrid = 'off';
%     %     colormap(gray);
%     c = colorbar;
%     c.Label.String = 'Count';
% %     t = sprintf('%s vs MOS',strrep(char(chosen_OMOV{k}),'_','\_'));
%     switch k
%         case 22
%             title('(a)')
%             xlabel('MeanOS')
%             ylabel('$\overline{\Delta \varphi}_{NW}$','Interpreter','latex')
%         case 23
%             title('(b)')
%             xlabel('MeanOS')
%             ylabel('$\sigma_{\Delta \varphi_{NW}}$','Interpreter','latex')
%         case 24
%             title('(c)')
%             xlabel('MeanOS')
%             ylabel('$\Delta \overline{ \Delta \varphi}_{NW}$','Interpreter','latex')
%         case 25
%             title('(d)')
%             xlabel('MeanOS')
%             ylabel('$\overline{\Delta \varphi}_{MAG}$','Interpreter','latex')
%         case 26
%             title('(e)')
%             xlabel('MeanOS')
%             ylabel('$\sigma_{\Delta \varphi_{MAG}}$','Interpreter','latex')
%         case 27
%             title('(f)')
%             xlabel('MeanOS')
%             ylabel('$\Delta \overline{ \Delta \varphi}_{MAG}$','Interpreter','latex')
%
%         otherwise
%             fprintf('Unknown title and labels switch\n')
%
%
%     end
%
%     subp = subp+1;
%     set(gca,...
%     'FontSize', 12, ...
%     'FontName', 'Times');
% end
%
%
% print('Plots/TIFF/Phasiness_Features_MOS', '-dtiff');
% print('Plots/EPSC/Phasiness_Features_MOS', '-depsc');
% print('Plots/PNG/Phasiness_Features_MOS', '-dpng');
%
%
% tsm = small_MOV(:,4);
% % tsm(tsm<1) = 1./(tsm(tsm<1));
%
%
%
% figure('Position',[0 0 1200 700])
% subp = 1;
% % n=5520;
% n = size(small_MOV,1);
% for k = 22:27
%     subplot(2,3,subp)
%     h = histogram2(tsm,small_MOV(1:n,k),[40 40],'FaceColor','flat');
%     h.DisplayStyle = 'tile';
%     h.EdgeAlpha = 0;
%     view(2)
%     ax = gca;
%     ax.XGrid = 'off';
%     ax.YGrid = 'off';
%     %     colormap(gray);
%     c = colorbar;
%     c.Label.String = 'Count';
%     t = sprintf('%s vs TSM',strrep(char(chosen_OMOV{k}),'_','\_'));
%     switch k
%         case 22
%             title('(a)')
%             xlabel('$\bf{R}$','Interpreter','latex')
%             ylabel('$\overline{\Delta \varphi}_{NW}$','Interpreter','latex')
%         case 23
%             title('(b)')
%             xlabel('$\bf{R}$','Interpreter','latex')
%             ylabel('$\sigma_{\Delta \varphi_{NW}}$','Interpreter','latex')
%         case 24
%             title('(c)')
%             xlabel('$\bf{R}$','Interpreter','latex')
%             ylabel('$\Delta \overline{ \Delta \varphi}_{NW}$','Interpreter','latex')
%         case 25
%             title('(d)')
%             xlabel('$\bf{R}$','Interpreter','latex')
%             ylabel('$\overline{\Delta \varphi}_{MAG}$','Interpreter','latex')
%         case 26
%             title('(e)')
%             xlabel('$\bf{R}$','Interpreter','latex')
%             ylabel('$\sigma_{\Delta \varphi_{MAG}}$','Interpreter','latex')
%         case 27
%             title('(f)')
%             xlabel('$\bf{R}$','Interpreter','latex')
%             ylabel('$\Delta \overline{ \Delta \varphi}_{MAG}$','Interpreter','latex')
%
%         otherwise
%             fprintf('Unknown title and labels switch\n')
%
%
%     end
%
%     subp = subp+1;
%     set(gca,...
%     'FontSize', 12, ...
%     'FontName', 'Times');
%
% end
%
% print('Plots/TIFF/Phasiness_Features_TSM', '-dtiff');
% print('Plots/EPSC/Phasiness_Features_TSM', '-depsc');
% print('Plots/PNG/Phasiness_Features_TSM', '-dpng');
%

% %% Plot Feature comparison to MOS and TSM Ratio
% figure
% set(gcf, 'Position', get(0, 'Screensize'));
% clf
% MOV_start = 4;
% MOVs = small_MOV;
% n = num_files;
% for k = MOV_start:size(MOVs,2)
%     if(mod(k-MOV_start,New_Figure)==0)
%         if(k>MOV_start)
%             f = sprintf('Plots/PNG/%s_Features_%d_to_%d.png',file_to_load,k-(mod(size(MOVs,2)-MOV_start,New_Figure))-1,k-1);
%             print(f,'-dpng')
%             f = sprintf('Plots/EPSC/%s_Features_%d_to_%d.eps',file_to_load,k-(mod(size(MOVs,2)-MOV_start,New_Figure))-1,k-1);
%             print(f,'-depsc')
%             clf %Clear currrent figure
%         end
%     end
%     subplot(3,4,mod((k-MOV_start)*4,12)+1);
%     h = histogram2(MOVs(1:n,1),MOVs(1:n,k),[75 75],'FaceColor','flat');
%     h.DisplayStyle = 'tile';
%     view(2)
%     %     colormap(gray);
%     c = colorbar;
%     c.Label.String = 'Count';
%     t = sprintf('%s vs MOS',strrep(char(chosen_OMOV{k}),'_','\_'));
%     title(t)
%     xlabel('MOS')
%     ylabel('Feature')
%
%     subplot(3,4,mod((k-MOV_start)*4,12)+2);
%     h = histogram2(MOVs(1:n,3),MOVs(1:n,k),[20 75],'FaceColor','flat');
%     h.DisplayStyle = 'tile';
%     view(2)
%     %     colormap(gray);
%     c = colorbar;
%     c.Label.String = 'Count';
%     t = sprintf('%s vs TSM',strrep(char(chosen_OMOV{k}),'_','\_'));
%     title(t)
%     xlabel('Time Scale')
%     ylabel('Feature')
%
%     temp = MOVs(1:n,k);
%     if sum(temp<0)==0
%         subplot(3,4,mod((k-MOV_start)*4,12)+3);
%         h = histogram2(MOVs(1:n,1),log10(MOVs(1:n,k)),[75 75],'FaceColor','flat');
%         h.DisplayStyle = 'tile';
%         view(2)
%         %     colormap(gray);
%         c = colorbar;
%         c.Label.String = 'Count';
%         t = sprintf('log10(%s) vs MOS',strrep(char(chosen_OMOV{k}),'_','\_'));
%         title(t)
%         xlabel('MOS')
%         ylabel('Feature')
%
%         subplot(3,4,mod((k-MOV_start)*4,12)+4);
%         h = histogram2(MOVs(1:n,3),log10(MOVs(1:n,k)),[20 75],'FaceColor','flat');
%         h.DisplayStyle = 'tile';
%         view(2)
%         %     colormap(gray);
%         c = colorbar;
%         c.Label.String = 'Count';
%         t = sprintf('log10(%s) vs TSM',strrep(char(chosen_OMOV{k}),'_','\_'));
%         title(t)
%         xlabel('Time Scale')
%         ylabel('Feature')
%     end
%
% end
% % mod(size(MOVs,2)-MOV_start,k)
% f = sprintf('Plots/PNG/%s_Features_%d_to_%d.png',file_to_load,k-(mod(size(MOVs,2)-MOV_start,New_Figure)),k);
% print(f,'-dpng')
% f = sprintf('Plots/EPSC/%s_Features_%d_to_%d.eps',file_to_load,k-(mod(size(MOVs,2)-MOV_start,New_Figure)),k);
% print(f,'-depsc')
%
% close all





%% Plot Feature comparison to MOS and TSM Ratio
% figure('Position',[0 20 1100 405])
% % set(gcf, 'Position', get(0, 'Screensize'));
% % clf
% MOV_start = 6;
% MOVs = small_MOV;
% n = num_files;
% for k = MOV_start:size(MOVs,2)
%     subplot(1,2,1);
%     h = histogram2(MOVs(1:n,1),MOVs(1:n,k),[75 75],'FaceColor','flat');
%     h.DisplayStyle = 'tile';
%     h.EdgeAlpha = 0;
%     view(2)
%     %     colormap(gray);
%     c = colorbar;
%     c.Label.String = 'Count';
%     %     t = sprintf('%s vs MOS',strrep(char(chosen_OMOV{k}),'_','\_'));
%     %     title(t)
%     xlabel('MOS')
%     ylabel(strrep(char(chosen_OMOV{k}),'_','\_'))
%     set(gca,...
%         'FontSize', 12, ...
%         'FontName', 'Times');
%     
%     subplot(1,2,2);
%     h = histogram2(MOVs(1:n,5),MOVs(1:n,k),[20 75],'FaceColor','flat');
%     h.DisplayStyle = 'tile';
%     h.EdgeAlpha = 0;
%     view(2)
%     %     colormap(gray);
%     c = colorbar;
%     c.Label.String = 'Count';
%     %     t = sprintf('%s vs TSM',strrep(char(chosen_OMOV{k}),'_','\_'));
%     %     title(t)
%     xlabel('Time Scale Ratio')
%     ylabel(strrep(char(chosen_OMOV{k}),'_','\_'))
%     set(gca,...
%         'FontSize', 12, ...
%         'FontName', 'Times');
%     f = sprintf('%s_Feature_%s.png',file_to_load(1:end-4),strrep(char(chosen_OMOV{k}),'\',''));
%     print(f,'-dpng')
%     f = sprintf('%s_Feature_%s.eps',file_to_load(1:end-4),strrep(char(chosen_OMOV{k}),'\',''));
%     print(f,'-depsc')
% %     clf
% end

% mod(size(MOVs,2)-MOV_start,k)
% f = sprintf('Plots/PNG/%s_Features_%d_to_%d.png',file_to_load,k-(mod(size(MOVs,2)-MOV_start,New_Figure)),k);
% print(f,'-dpng')
% f = sprintf('Plots/EPSC/%s_Features_%d_to_%d.eps',file_to_load,k-(mod(size(MOVs,2)-MOV_start,New_Figure)),k);
% print(f,'-depsc')

%% Plot Phasiness Feature comparison to MOS and TSM Ratio
figure('Position',[0 0 618 835])
% set(gcf, 'Position', get(0, 'Screensize'));
% clf
source = 88;
MOV_start = 27;
MOV_end = 30;
MOVs = small_MOV;
n = num_files;
for k = MOV_start:MOV_end
    subplot(MOV_end-MOV_start+1,2,mod((k-MOV_start),MOV_end-MOV_start+1)*2+1);
    h = histogram2(MOVs(source+1:source+n,1),MOVs(source+1:source+n,k),[75 75],'FaceColor','flat');
    h.DisplayStyle = 'tile';
    h.EdgeAlpha = 0;
    view(2)
    %     colormap(gray);
    c = colorbar;
    c.Label.String = 'Count';
    %     t = sprintf('%s vs MOS',strrep(char(chosen_OMOV{k}),'_','\_'));
    %     title(t)
    xlabel('MeanOS')
    ylabel(strrep(char(chosen_OMOV{k}),'_','\_'))
    set(gca,...
        'FontSize', 12, ...
        'FontName', 'Times');
    
    subplot(MOV_end-MOV_start+1,2,mod((k-MOV_start),MOV_end-MOV_start+1)*2+2);
    h = histogram2(MOVs(source+1:source+n,5),MOVs(source+1:source+n,k),[20 75],'FaceColor','flat');
    h.DisplayStyle = 'tile';
    h.EdgeAlpha = 0;
    view(2)
    %     colormap(gray);
    c = colorbar;
    c.Label.String = 'Count';
    %     t = sprintf('%s vs TSM',strrep(char(chosen_OMOV{k}),'_','\_'));
    %     title(t)
    xlabel('Time-Scale Ratio (\beta)')
    ylabel(strrep(char(chosen_OMOV{k}),'_','\_'))
    set(gca,...
        'FontSize', 12, ...
        'FontName', 'Times');
    
    %     clf
end

f = sprintf('Plots/PNG/Phasiness_Features.png');%,file_to_load(1:end-4),strrep(char(chosen_OMOV{k}),'\',''));
print(f,'-dpng')
f = sprintf('Plots/EPSC/Phasiness_Features.eps');%,file_to_load(1:end-4),strrep(char(chosen_OMOV{k}),'\',''));
print(f,'-depsc')

%% Plot Transient Feature comparison to MOS and TSM Ratio
figure('Position',[0 0 618 651])
% set(gcf, 'Position', get(0, 'Screensize'));
% clf
MOV_start = 24;
MOV_end = 26;
MOVs = small_MOV;
n = num_files;
for k = MOV_start:MOV_end
    subplot(MOV_end-MOV_start+1,2,mod((k-MOV_start),MOV_end-MOV_start+1)*2+1);
    h = histogram2(MOVs(source+1:source+n,1),MOVs(source+1:source+n,k),[75 75],'FaceColor','flat');
    h.DisplayStyle = 'tile';
    h.EdgeAlpha = 0;
    view(2)
    %     colormap(gray);
    c = colorbar;
    c.Label.String = 'Count';
    %     t = sprintf('%s vs MOS',strrep(char(chosen_OMOV{k}),'_','\_'));
    %     title(t)
    xlabel('MeanOS')
    ylabel(strrep(char(chosen_OMOV{k}),'_','\_'))
    set(gca,...
        'FontSize', 12, ...
        'FontName', 'Times');
    
    subplot(MOV_end-MOV_start+1,2,mod((k-MOV_start),MOV_end-MOV_start+1)*2+2);
    h = histogram2(MOVs(source+1:source+n,5),MOVs(source+1:source+n,k),[20 75],'FaceColor','flat');
    h.DisplayStyle = 'tile';
    h.EdgeAlpha = 0;
    view(2)
    %     colormap(gray);
    c = colorbar;
    c.Label.String = 'Count';
    %     t = sprintf('%s vs TSM',strrep(char(chosen_OMOV{k}),'_','\_'));
    %     title(t)
    xlabel('Time-Scale Ratio (\beta)')
    ylabel(strrep(char(chosen_OMOV{k}),'_','\_'))
    set(gca,...
        'FontSize', 12, ...
        'FontName', 'Times');
    
    %     clf
end

f = sprintf('Plots/PNG/Transient_Features.png');%,file_to_load(1:end-4),strrep(char(chosen_OMOV{k}),'\',''));
print(f,'-dpng')
f = sprintf('Plots/EPSC/Transient_Features.eps');%,file_to_load(1:end-4),strrep(char(chosen_OMOV{k}),'\',''));
print(f,'-depsc')

%% Plot Spectral Similarity Feature comparison to MOS and TSM Ratio
figure('Position',[0 0 618 420])
% set(gcf, 'Position', get(0, 'Screensize'));
% clf
MOV_start = 31;
MOV_end = 32;
MOVs = small_MOV;
n = num_files;
for k = MOV_start:MOV_end
    subplot(MOV_end-MOV_start+1,2,mod((k-MOV_start),MOV_end-MOV_start+1)*2+1);
    h = histogram2(MOVs(source+1:source+n,1),MOVs(source+1:source+n,k),[75 75],'FaceColor','flat');
    h.DisplayStyle = 'tile';
    h.EdgeAlpha = 0;
    view(2)
    %     colormap(gray);
    c = colorbar;
    c.Label.String = 'Count';
    %     t = sprintf('%s vs MOS',strrep(char(chosen_OMOV{k}),'_','\_'));
    %     title(t)
    xlabel('MeanOS')
    ylabel(strrep(char(chosen_OMOV{k}),'_','\_'))
    set(gca,...
        'FontSize', 12, ...
        'FontName', 'Times');
    
    subplot(MOV_end-MOV_start+1,2,mod((k-MOV_start),MOV_end-MOV_start+1)*2+2);
    h = histogram2(MOVs(source+1:source+n,5),MOVs(source+1:source+n,k),[20 75],'FaceColor','flat');
    h.DisplayStyle = 'tile';
    h.EdgeAlpha = 0;
    view(2)
    %     colormap(gray);
    c = colorbar;
    c.Label.String = 'Count';
    %     t = sprintf('%s vs TSM',strrep(char(chosen_OMOV{k}),'_','\_'));
    %     title(t)
    xlabel('Time-Scale Ratio (\beta)')
    ylabel(strrep(char(chosen_OMOV{k}),'_','\_'))
    set(gca,...
        'FontSize', 12, ...
        'FontName', 'Times');
    
    %     clf
end

f = sprintf('Plots/PNG/Spec_Sim_Features.png');%,file_to_load(1:end-4),strrep(char(chosen_OMOV{k}),'\',''));
print(f,'-dpng')
f = sprintf('Plots/EPSC/Spec_Sim_Features.eps');%,file_to_load(1:end-4),strrep(char(chosen_OMOV{k}),'\',''));
print(f,'-depsc')


close all
% 
function add_labels(chosen_OMOV)
h = gca;
h.Visible = 'On';
% xticks(1:size(chosen_OMOV,2))
% xticklabels(chosen_OMOV)
% xtickangle(90)
yticks(1:size(chosen_OMOV,2))
set(h,'TickLabelInterpreter', 'tex');
yticklabels(chosen_OMOV)
c = colorbar;
c.Label.String = '|\rho|';
set(gca,...
    'FontSize', 12, ...
    'FontName', 'Times');
c.Label.FontSize = 18;
c.Label.FontName = 'Times';
c.Label.Position = [2.7 0.535];
c.Label.Rotation = 0; % to rotate the text



end
