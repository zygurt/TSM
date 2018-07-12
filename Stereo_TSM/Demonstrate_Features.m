%% ------------ Testing of Stereo Feature functions --------------
close all
clear all
clc

addpath('../Functions');

% pathInput = './AudioIn/';
% filename = 'Electropop.wav';
% filename = 'Choral.wav';
% filename = 'Jazz.wav';
% filename = 'Saxophone_Quartet.wav';

%Create and plot features for Sum and Difference fade
pathInput = './Synthetic_Audio_Files/';
filename = 'White_SD_Fade.wav';
[x,fs] = audioread([pathInput filename]);
if(size(x,2)~=2)
    disp('File is not stereo');
    x = [x x];
end
%Create features
[fr_b, fi_b] = st_balance(x,2048);
[fr_SPC, fi_SPC] = st_phase_coherence(x,2048);
%Plot the features
figure
subplot(211)
plot(fr_SPC(1:end-1),'k')
hold on
plot(fr_b(1:end-1),'k--')
set(gca, 'FontName', 'Times New Roman')
plot_title = sprintf("Stereo Features for %s",strrep(filename(1:end-4),'_',' '));
title(plot_title)
axis([1 (length(fr_b)-1) -1.1 1.1]);
xlabel('Time (Analysis Frame)');
ylabel('Feature Value');
legend('Coherence','Balance','location','best');

%Create and plot features for Sine Tone Panning file
pathInput = './Synthetic_Audio_Files/';
filename = 'Sine_Panning.wav';
[x,fs] = audioread([pathInput filename]);
if(size(x,2)~=2)
    disp('File is not stereo');
    x = [x x];
end
%Create features
[fr_b, fi_b] = st_balance(x,2048);
[fr_SPC, fi_SPC] = st_phase_coherence(x,2048);
%Plot the features

subplot(212)
plot(fr_SPC(1:end-1),'k')
hold on
plot(fr_b(1:end-1),'k--')
hold off
set(gca, 'FontName', 'Times New Roman')
plot_title = sprintf("Stereo Features for %s",strrep(filename(1:end-4),'_',' '));
title(plot_title)
axis([1 (length(fr_b)-1) -1.1 1.1]);
xlabel('Time (Analysis Frame)');
ylabel('Feature Value');
legend('Coherence','Balance','location','best');


x0=10;
y0=10;
width=300;
height=300;
set(gcf,'units','points','position',[x0,y0,width,height])