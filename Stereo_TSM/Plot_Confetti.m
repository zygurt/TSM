%Plotting all results to examine variance of stereo phase coherence.
close all
clear all
clc
%Load the result files generated in testing
load confetti_original.mat
load confetti_results.mat
% load subjective_files_original.mat
% load subjective_files_results.mat
methods = {'Naive PV', 'Bonada PV', 'Altoe PV', 'Prop. File PV', 'Prop. Frame PV', 'Naive WSOLA', 'Prop. File WSOLA', 'Naive HP', 'Prop. File HP'};
plotting_methods = {'Naive PV', 'Bonada PV', 'Altoe PV', 'Prop. File PV', 'Prop. Frame PV', 'Naive WSOLA', 'Prop. File WSOLA', 'Naive HP', 'Prop. File HP', 'Original'};

pointer = ['.' '.'];
%Generate multiple different colours
num_of_colours = length(methods);
H = (0:num_of_colours-1)*1/num_of_colours;
S = ones(size(H));
V = ones(size(H));
V(2:2:end) = 0.9*V(2:2:end);
HSV = [H',S',V'];
RGB = hsv2rgb(HSV);
number_of_files = size(results,1);
time_scales = size(results,2);
num_methods = size(results(1,1).C_o,1);
min_val = 2;

%% Plot all of the coherence values in the same figure
% This section places overall stereo phase coherence for all methods on 1 graph.
figure
hold on
for k=1:number_of_files
    for n=1:time_scales
        x = ((k-1)*time_scales+n);
        for m=1:length(methods)
            plot(x,results(k,n).C_o(m),pointer(mod(m,2)+1),'Color',RGB(m,:));
        end
        plot(x,original(k).C_o,'k.');
    end
    line([(k-1)*time_scales+n+0.5 (k-1)*time_scales+n+0.5],[-1 1]);
end
set(gca, 'FontName', 'Times New Roman')
title('Mean Stereo Phase Coherence at Multiple Time-Scale Ratios');
ylabel('Stereo Phase Coherence');
xlabel('File and time scale. Vertical lines show file distinction.');

x_tick_names = {};
for k = 1:number_of_files
    x_tick_names{k} = strrep(results(k,1).filename,'_', ' ');
end
x_tick_location = time_scales/2:time_scales:number_of_files*time_scales;
set(gca, 'XLim', [0 number_of_files*time_scales],  'YLim', [-1 1],...
    'XTick', x_tick_location, 'YTick', -1:1:1,...
    'XTickLabel', x_tick_names)
axis tight;
xtickangle(90);

hold off
legend(plotting_methods,'Location','NorthEastOutside');

%% Plot all Balance values on the same figure
figure
hold on
for k=1:number_of_files
    for n=1:time_scales
        x = ((k-1)*time_scales+n);
        for m=1:length(methods)
            plot(x,results(k,n).B_o(m),pointer(mod(m,2)+1),'Color',RGB(m,:));
        end
        plot(x,original(k).B_o,'k.');
    end
    line([(k-1)*time_scales+n+0.5 (k-1)*time_scales+n+0.5],[-1 1]);
end
set(gca, 'FontName', 'Times New Roman')
title('Mean Balance at Multiple Time-Scale Ratios');
ylabel('Balance');
xlabel('File and time scale. Vertical lines show file distinction.');

x_tick_names = {};
for k = 1:number_of_files
    x_tick_names{k} = strrep(results(k,1).filename,'_', ' ');
end
x_tick_location = time_scales/2:time_scales:number_of_files*time_scales;
set(gca, 'XLim', [0 number_of_files*time_scales],  'YLim', [-1 1],...
    'XTick', x_tick_location,...
    'XTickLabel', x_tick_names )
axis([0 (number_of_files-1)*time_scales+time_scales+0.5 -0.081 0.05]);
xtickangle(90);

hold off
legend(plotting_methods,'Location','NorthEastOutside');



%% Plot all of the files, time scales and methods
%This section gives each method its own figure for overall stereo phase coherence.

% x_tick_names = {};
% for k = 1:number_of_files
%     x_tick_names{k} = strrep(results(k,1).filename,'_', ' ');
% end
% x_tick_location = time_scales/2:time_scales:number_of_files*time_scales;
% 
% for f=1:num_methods
%     figure
%     hold on
%     for k=1:number_of_files
%         for n=1:time_scales
%             x = (k-1)*time_scales+n;
%             plot(x,results(k,n).overall_coherence(f),'.');
%             %         if n == 1
%             plot(x,original(k).overall_coherence,'k.');
%             %         end
%         end
%         line([(k-1)*time_scales+n+0.5 (k-1)*time_scales+n+0.5],[-1 1]);
%     end
%     hold off
%     fig_title = sprintf('Stereo Phase Coherence for files at time-scale ratios for %s',methods{f});
%     title(fig_title);
%     ylabel('Stereo Phase Coherence');
%     xlabel('File and time scale. Vertical lines show file distinction.');
%     set(gca, 'XLim', [0 number_of_files*time_scales],  'YLim', [-1 1],...
%         'XTick', x_tick_location, 'YTick', -1:1:1,...
%         'XTickLabel', x_tick_names, ...
%         'YTickLabel', {'180 degrees', '90 degrees', '0 degrees'})
%     xtickangle(90);
% end


