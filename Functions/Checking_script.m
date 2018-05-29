%Script to check that the functions work correctly
close all
clear all
clc

addpath('../Stereo');
addpath('../N_Channel');
addpath('../Time_Domain');


%% ------------ Testing of Stereo Feature functions --------------
% pathInput = '../FDTSM/AudioIn/';
% filename = 'Electropop.wav';

% [x,fs] = audioread([pathInput filename]);

% [fr_b_norm, fi_b_norm] = st_balance(x,2048,1);
% [fr_b, fi_b] = st_balance(x,2048,0);
% [fr_SPC, fi_SPC] = st_phase_coherence(x,2048);
% [fr_w, fi_w] = st_width(x,2048);
%
% figure
% plot(fr_b_norm)
% hold on
% plot(fr_b)
% plot(fr_SPC)
% plot(fr_w)
% hold off

%% ------------ Testing Stereo Phase Vocoder functions --------------

% pathInput = '../FDTSM/AudioIn/';
% filename = 'Electropop.wav';
% 
% [x,fs] = audioread([pathInput filename]);

%Simple Mono testing
% fs = 44100;
% x = sin(2*pi*1000*(linspace(0,3,3*fs))');
% x = [x,x,x,x,x,x];

% TSM = 0.8;
% N = 2048;
%Stereo Phase Vocoders
% y_n = PV(x, N, TSM);
% y_A = PV_Altoe(x, N, TSM);
% y_B = PV_Bonada(x, N, TSM);
% y_MS_Fi = PV_MS_File(x, N, TSM);
% y_MS_Fr = PV_MS_Frame(x, N, TSM);

% figure
% subplot(6,1,1)
% if TSM < 1
% plot((1:length(y_n))/fs,[x ; zeros(length(y_n)-length(x),size(x,2))])
% title('Original (Zero padded to same length as scaled signals)')
% else
%     plot((1:length(x))/fs,x)
%     title('Original')
% end
% axis tight
% xlabel('Time (s)')
% 
% subplot(6,1,2)
% plot((1:length(y_n))/fs,y_n)
% axis tight
% title('PV\_naive')
% xlabel('Time (s)')
% 
% subplot(6,1,3)
% plot((1:length(y_A))/fs,y_A)
% axis tight
% title('PV\_Altoe')
% xlabel('Time (s)')
% 
% subplot(6,1,4)
% plot((1:length(y_B))/fs,y_B)
% axis tight
% title('PV\_Bonada')
% xlabel('Time (s)')
% 
% subplot(6,1,5)
% plot((1:length(y_MS_Fi))/fs,y_MS_Fi)
% axis tight
% title('PV\_MS\_File')
% xlabel('Time (s)')
% 
% subplot(6,1,6)
% plot((1:length(y_MS_Fr))/fs,y_MS_Fr)
% axis tight
% title('PV\_MS\_Frame')
% xlabel('Time (s)')


%Stereo Phase Locking Phase Vocoders
% y_NPL_PV = PL_PV( x, N, TSM, 0 ); %No Phase Locking
% y_IPL_PV = PL_PV( x, N, TSM, 1 ); %Identity
% y_SPL_PV = PL_PV( x, N, TSM, 2 ); %Scaled
% 
% figure
% 
% subplot(3,1,1)
% plot((1:length(y_NPL_PV))/fs,y_NPL_PV)
% axis tight
% title('NoPL\_PV')
% xlabel('Time (s)')
% 
% subplot(3,1,2)
% plot((1:length(y_IPL_PV))/fs,y_IPL_PV)
% axis tight
% title('IPL\_PV')
% xlabel('Time (s)')
% 
% subplot(3,1,3)
% plot((1:length(y_SPL_PV))/fs,y_SPL_PV)
% axis tight
% title('SPL\_PV')
% xlabel('Time (s)')


%Phavorit Stereo Phase Locking Phase Vocoders
% y_Phavorit_IPL_PV = Phavorit_PV( x, N, TSM, 0 ); %Identity
% y_Phavorit_SPL_PV = Phavorit_PV( x, N, TSM, 1 ); %Scaled
% 
% figure
% subplot(2,1,1)
% plot((1:length(y_Phavorit_IPL_PV))/fs,y_Phavorit_IPL_PV)
% axis tight
% title('y\_Phavorit\_IPL\_PV')
% xlabel('Time (s)')
% 
% subplot(2,1,2)
% plot((1:length(y_Phavorit_SPL_PV))/fs,y_Phavorit_SPL_PV)
% axis tight
% title('y\_Phavorit\_SPL\_PV')
% xlabel('Time (s)')

%Play original followed by time scaled version
% soundsc(x,fs);
% pause(1.1*length(x)/fs)
% soundsc(y_SPL_PV,fs);
% pause(1.1*length(y_SPL_PV)/fs)
% soundsc(y_Phavorit_SPL_PV,fs);



%% ------------ Checking the find peaks function ------------
%
% channels = 5;
% data_points = 50;
% x = rand(data_points,channels);
% p = find_peaks(x);
% figure
% for c = 1:size(x,2)
%     subplot(size(x,2),1,c)
%     %Plot the original signal
%     stem(x(:,c))
%     hold on
%     %Plot the peak locations
%     for a = 1:length(p(c).pa)
%         subplot(size(x,2),1,c)
%         line([p(c).pa(a),p(c).pa(a)],[0,x(p(c).pa(a),c)],'Color','red');
%     end
%     %Plot the region
%     for a = 1:length(p(c).rl)
%         subplot(size(x,2),1,c)
%         line([p(c).rl(a),p(c).ru(a)],[-0.1,-0.1],'Color','magenta');
%     end
%     title(sprintf('Channel %d',c));
%     hold off
% end

%% ------------ Checking the previous peak function ------------
%
% frames = 3;
% data_points = 50;
% x = rand(data_points,frames);
% p = find_peaks(x);
% figure
% for c = 1:size(x,2)
%     subplot(size(x,2),1,c)
%     %Plot the original signal
%     stem(x(:,c))
%     hold on
%     %Plot the peak locations
%     for a = 1:length(p(c).pa)
%         subplot(size(x,2),1,c)
%         line([p(c).pa(a),p(c).pa(a)],[0,x(p(c).pa(a),c)],'Color','red');
%     end
%         %Plot the region
%     for a = 1:length(p(c).rl)
%         subplot(size(x,2),1,c)
%         line([p(c).rl(a),p(c).ru(a)],[-0.1,-0.1],'Color','magenta');
%     end
%     %Plot the previous peak locations
%     if(c>1)
%         for a = 1:length(p(c).pa)
%             prev_p = previous_peak(p(c).pa(a) , p(c-1).pa , p(c-1).rl , p(c-1).ru);
%             %Plot arrow from current peak to previous peak
%             quiver(p(c).pa(a), ... %x
%                    x(p(c).pa(a),c), ... %y
%                    prev_p-p(c).pa(a), ... %delta x
%                    x(prev_p,c-1)-x(p(c).pa(a),c), ... %delta y
%                    0, 'k'); %Remove scaling
%             %Plot the previous peak
%             %line([prev_p,prev_p],[0,x(p(c-1).pa(a),c-1)],'Color','green','LineStyle','--');
%         end
%     end
%     hold off
%     axis([1 data_points -0.2 1.1])
%     title(sprintf('Frame %d',c));
%
% end

%% ------------ Checking the find peaks log function ------------

% channels = 4;
% data_points = 128;
% x = rand(data_points,channels);
% p = find_peaks_log(x);
% figure
% for c = 1:size(x,2)
%     subplot(size(x,2),1,c)
%     %Plot the original signal
%     stem(x(:,c))
%     hold on
%     %Plot the peak locations
%     for a = 1:length(p(c).pa)
%         subplot(size(x,2),1,c)
%         line([p(c).pa(a),p(c).pa(a)],[0,x(p(c).pa(a),c)],'Color','red');
%     end
%     %Plot the region
%     for a = 1:length(p(c).rl)
%         subplot(size(x,2),1,c)
%         line([p(c).rl(a),p(c).ru(a)],[-0.1,-0.1],'Color','magenta');%,'Marker','.');
%     end
%     title(sprintf('Channel %d',c));
%     hold off
% end

%% ------------ Checking the previous peak function ------------

% frames = 3;
% data_points = 256;
% x = rand(data_points,frames);
% p = find_peaks_log(x);
% figure
% for c = 1:size(x,2)
%     subplot(size(x,2),1,c)
%     %Plot the original signal
%     stem(x(:,c))
%     hold on
%     %Plot the peak locations
%     for a = 1:length(p(c).pa)
%         subplot(size(x,2),1,c)
%         line([p(c).pa(a),p(c).pa(a)],[0,x(p(c).pa(a),c)],'Color','red');
%     end
%     %Plot the region
%     for a = 1:length(p(c).rl)
%         subplot(size(x,2),1,c)
%         line([p(c).rl(a),p(c).ru(a)],[-0.1,-0.1],'Color','magenta');
%     end
%     %Plot the previous peak locations
%     if(c>1)
%         for a = 1:length(p(c).pa)
%             prev_p = previous_peak_heuristic(p(c).pa(a) , p(c-1).pa , p(c-1).rl , p(c-1).ru);
%             %Plot arrow from current peak to previous peak
%             if prev_p >0
%                 quiver(p(c).pa(a), ... %x
%                     x(p(c).pa(a),c), ... %y
%                     prev_p-p(c).pa(a), ... %delta x
%                     x(prev_p,c-1)-x(p(c).pa(a),c), ... %delta y
%                     0, 'k'); %Remove scaling
%             else
%                 quiver(p(c).pa(a), ... %x
%                     x(p(c).pa(a),c), ... %y
%                     0, ... %delta x
%                     0.1, ... %delta y
%                     0); %Remove scaling
%             end
%             %Plot the previous peak
%             %line([prev_p,prev_p],[0,x(p(c-1).pa(a),c-1)],'Color','green','LineStyle','--');
%         end
%     end
%     hold off
%     axis([1 data_points -0.2 1.1])
%     title(sprintf('Frame %d',c));
%     
% end


%% ---------------Testing of the maxcrosscorr function-------------------------
% %This function became maxcrosscorrlag.  The same variable names are used
% %inside the function
% x = 10*ones(1,10);
% y = [0,0,0,0,0,0,10*ones(1,10),0,0,0,1,1,0,0,2,3,4,2,1,2,3,4,2,2,1,2,3,3];
% 
% [lag_x, lag_y, max_loc, xc_a] = maxcrosscorr(x,y,4,4);
% fprintf('lag_x = %d, lag_y = %d, max_loc = %d\n', lag_x, lag_y, max_loc);
% subplot(311)
% plot(x)
% subplot(312)
% plot(y)
% subplot(313)
% plot(xc_a)
%lag_x is the amount that x needs to move for maximum cross correlation
%lag_y is the amount that y needs to move for maximum cross correlation
%positive values mean later in time, negative values mean earlier in time
%max_loc is the location of the maximum within the cross correlation array
%xc_a is the cross correlation array
%low_lim and high_lim are used to remove large cross correlation values at
%the extreme ranges of the cross correlation.  Values in these ranges will
%not be calculated, and left as 0.

%% ---------------Testing of SOLA Time Domain Time Scale Modification---------------
% 
% pathInput = '../FDTSM/AudioIn/';
% filename = 'Male_Speech.wav';
% 
% [x,fs] = audioread([pathInput filename]);
% x = sum(x,2)/max(sum(x,2));
% TSM = 0.8;
% ms = 80;
% N = 2^(nextpow2(ms*(10^-3)*fs));
%  
% %Stereo Phase Vocoders
% y = SOLA(x, N, TSM);
% y_DAFX = SOLA_DAFX(x, N, TSM);
% figure
% subplot(311)
% plot(x)
% subplot(312)
% plot(y)
% subplot(313)
% plot(y_DAFX)
% 
% 
% soundsc(x,fs)
% pause((length(x)/fs)*1.1);
% soundsc(y,fs)
% pause((length(y)/fs)*1.1);
% soundsc(y_DAFX,fs)

%% ---------------Testing of WSOLA Time Domain Time Scale Modification---------------

% pathInput = '../FDTSM/AudioIn/';
% filename = 'Male_Speech.wav';
% 
% [x,fs] = audioread([pathInput filename]);
% x = sum(x,2)/max(sum(x,2));
% TSM = 0.8;
% ms = 25;
% N = 2^(nextpow2(ms*(10^-3)*fs));
% % N = ms*(10^-3)*fs;
%  
% %Stereo Phase Vocoders
% y = WSOLA(x, N, TSM);
% y_Driedger = WSOLA_Driedger(x, N, TSM);
% 
% soundsc(x,fs)
% figure
% subplot(311)
% plot(x)
% title('Original');
% pause((length(x)/fs)*1.1);
% 
% soundsc(y,fs)
% subplot(312)
% plot(y)
% title('WSOLA');
% pause((length(y)/fs)*1.1);
% 
% subplot(313)
% plot(y_Driedger)
% title('WSOLA_Driedger');
% soundsc(y_Driedger,fs)
% pause((length(y_Driedger)/fs)*1.1);

%% -------------Zero Frequency Resonator Testing---------------------

% pathInput = '../FDTSM/AudioIn/';
% filename = 'Male_Speech.wav';
% 
% [x,fs] = audioread([pathInput filename]);
% x = sum(x,2)/max(sum(x,2));
% 
% y = ZFR(x, fs);
% 
% line(1:length(y),y,'Color','red')
% hold on
% plot(x)
% hold off
% legend('Epochs','Speech signal','Location','best')
% title('Epoch locations within speech');

%% -------------ESOLA Testing---------------------

pathInput = '../FDTSM/AudioIn/';
filename = 'Male_Speech.wav';

[x,fs] = audioread([pathInput filename]);
x = sum(x,2)/max(sum(x,2));
TSM = 1;
ms = 20;
N = ms*(10^-3)*fs;

y_ESOLA = ESOLA(x, N, TSM, fs);

figure
subplot(211)
soundsc(x,fs)
plot(x)
title('Original');
pause((length(x)/fs)*1.1);

subplot(212)
soundsc(y_ESOLA,fs)
plot(y_ESOLA)
title('ESOLA');

LogSpectrogram(x,fs,50,2);
title('Original')
LogSpectrogram(y_ESOLA,fs,50,2);
title('ESOLA');
