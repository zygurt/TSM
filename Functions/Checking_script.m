%Script to check that the functions work correctly
close all
clear all
clc

[x,fs] = audioread('Electropop.wav');

% [fr_b_norm, fi_b_norm] = st_balance(x,2048,1);
% [fr_b, fi_b] = st_balance(x,2048,0);
% [fr_SPC, fi_SPC] = st_phase_coherence(x,2048);
% [fr_w, fi_w] = st_width(x,2048);
% 
% figure(1)
% plot(fr_b_norm)
% hold on
% plot(fr_b)
% plot(fr_SPC)
% plot(fr_w)
% hold off

TSM = 0.9;
N = 2048;

y_n = PV(x, N, TSM);
y_A = PV_Altoe(x, N, TSM);
y_B = PV_Bonada(x, N, TSM);
y_MS_Fi = PV_MS_File(x, N, TSM);
y_MS_Fr = PV_MS_Frame(x, N, TSM);

figure(2)
subplot(6,1,1)
plot((1:length(x))/fs,x)
axis tight
title('Original')
xlabel('Time (s)')

subplot(6,1,2)
plot((1:length(y_n))/fs,y_n)
axis tight
title('PV\_naive')
xlabel('Time (s)')

subplot(6,1,3)
plot((1:length(y_A))/fs,y_A)
axis tight
title('PV\_Altoe')
xlabel('Time (s)')

subplot(6,1,4)
plot((1:length(y_B))/fs,y_B)
axis tight
title('PV\_Bonada')
xlabel('Time (s)')

subplot(6,1,5)
plot((1:length(y_MS_Fi))/fs,y_MS_Fi)
axis tight
title('PV\_MS\_File')
xlabel('Time (s)')

subplot(6,1,6)
plot((1:length(y_MS_Fr))/fs,y_MS_Fr)
axis tight
title('PV\_MS\_Frame')
xlabel('Time (s)')


soundsc(y_MS_Fi,fs)
pause(1.1*length(y_MS_Fi)/fs);
soundsc(y_MS_Fr,fs)
