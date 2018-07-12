%% Feature plotting
close all
clear all
clc

%I suggest setting a break point after the figure is created, and maximising the figure on a monitor
% in a portait orientation before running this code.  The bigger the screen, the better.

fig = figure;
addpath('../Functions');
tic

pathOutput = 'Plots/Features/';
methods = {'Naive PV', 'Bonada PV', 'Altoe PV', 'Prop. File PV', 'Prop. Frame PV', 'Naive WSOLA', 'Prop. File WSOLA', 'Naive HP', 'Prop. File HP', 'Original'};

load confetti_original.mat
load confetti_results.mat
pathInput = 'Confetti_Source/';
%load subjective_files_results.mat
%load subjective_files_original.mat
% pathInput = 'AudioIn/';

%Calculate the colours
num_of_colours = length(methods)-1;
H = (0:num_of_colours-1)*1/num_of_colours;
S = ones(size(H));
V = ones(size(H));
V(2:2:end) = 0.9*V(2:2:end);
HSV = [H',S',V'];
RGB = hsv2rgb(HSV);

d = dir(pathInput);
N=2048; %The larger this value, the smoother the averaged phase will be.
N_f = 2048;

for audio_file = 1:size(results, 1)
    for TSM_ratio = 1:size(results, 2)

        %Plot the original Stereo Phase Coherence at the same time scale as the
        %adjusted files
        subplot(3,1,1);
        hold on
        %Plot the calculated SPC features
        t = (1:N_f:N_f*length(results(audio_file,TSM_ratio).C(:,:)'))/results(audio_file,TSM_ratio).Fs;
        for n=1:(length(methods)-1)
            plot(t,results(audio_file,TSM_ratio).C(n,:)','Color',RGB(n,:))
        end
        %Plot the original SPC feature
        t = (1:N:N*length(original(1,audio_file).C(:,:)'))/(original(1,audio_file).Fs*results(audio_file,TSM_ratio).TSM);
        plot(t,original(1,audio_file).C(:,:)', 'k')
        set(gca, 'XLim', [0 max(t)],  'YLim', [-1.1 1.1],...
         'XTick', 0:1:max(t), 'YTick', -1:2:1,...
         'XTickLabel', 0:1:max(t), ...
         'YTickLabel', {'180 degrees', '0 degrees'})
        title('Stereo Phase Coherence')
        legend(methods, 'Location', 'eastoutside')
        xlabel('Time (seconds)')
        ylabel('Stereo Phase Coherence')
        hold off

        %Plot the stereo Balance
        subplot(3,1,2)
        hold on
        %Plot the calculated Balance features
        t = (1:N_f:N_f*length(results(audio_file,TSM_ratio).B(:,:)'))/results(audio_file,TSM_ratio).Fs;
        for n=1:(length(methods)-1)
            plot(t,results(audio_file,TSM_ratio).B(n,:)','Color',RGB(n,:))
        end
        %Plot the original Balance feature
        t = (1:N:N*length(original(1,audio_file).B(:,:)'))/(original(1,audio_file).Fs*results(audio_file,TSM_ratio).TSM);
        plot(t,original(1,audio_file).B(:,:)', 'k')
        set(gca, 'XLim', [0 max(t)],  'YLim', [-1.1 1.1],...
         'XTick', 0:1:max(t), 'YTick', -1:2:1,...
         'XTickLabel', 0:1:max(t), ...
         'YTickLabel', {'Right', 'Left'})
        title('Stereo Balance');
        xlabel('Time (seconds)');
        ylabel('Stereo Balance');
        legend('Original', methods, 'Location', 'eastoutside')
        hold off

        %Plot the original audio file
        subplot(3,1,3)
        [x,Fs] = audioread([pathInput results(audio_file,TSM_ratio).filename]);
        x = x/max(max(abs(x))); %Normalise
        t = (1:length(x))/Fs;
        %Split the signal to use 1 plot
        if(size(x,2)==1)
            x = [x,x];
        end
        x(:,1) = x(:,1)+1;
        x(:,2) = x(:,2)-1;
        plot(t,x)
        title('Original Audio File')
        xlabel('Time (seconds)');
        ylabel('Amplitude');
        axis([0 t(end) -1.1*max(max(abs(x))) 1.1*max(max(abs(x)))])
        set(gca, 'YLim', [-1.1*max(max(abs(x))) 1.1*max(max(abs(x)))],...
            'YTick', [-2 -1 0 1 2],...
            'YTickLabel', {'-1      ' 'Right' '1 , -1' 'Left' '1'} );

        %Add the super title
        plot_title = sprintf('%s at TSM=%.2f',original(1, audio_file).filename(1:end-4),results(audio_file,TSM_ratio).TSM);
        plot_title = strrep(plot_title, '_', '\_');
        p=mtit(plot_title,...
            'fontsize',12,'color',[0 0 0],...
            'xoff',0,'yoff',0.025);
        plot_title = sprintf('%s at TSM=%.2f',results(audio_file,TSM_ratio).filename(1:end-4),results(audio_file,TSM_ratio).TSM);
        %Print the figure to file
        set(gcf,'PaperPositionMode','auto')
        print(fig, ['Plots/Features/' plot_title],'-dpng', '-r0')
        disp(plot_title);
    end
end
disp('Processing Complete')
toc