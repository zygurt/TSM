%Script to check that the functions work correctly
close all
clear all
clc

addpath('../Frequency_Domain');
addpath('../Time_Domain');

%% ------------ Testing of Stereo Feature functions --------------
% pathInput = '../Audio_test_files/';
% % filename = 'Electropop.wav';
% filename = 'Sine_Panning.wav';
% % filename = 'TSM_music_MS_fade.wav';
% % filename = 'Choral.wav';
% % filename = 'Jazz.wav';
% % filename = 'Saxophone_Quartet.wav';
% %  filename = 'White_MS_fade.wav';
% % filename = 'Alto_Sax_08.wav';
% [x,fs] = audioread([pathInput filename]);
% if(size(x,2)~=2)
%     disp('File is not stereo');
%     x = [x x];
% end
%     %Generate DC signal
%     % len = 44100;
%     % x(1:len,1) = ones(len,1);
%     % x(1:len,2) = zeros(len,1);
%     %Create features
%     [fr_b, fi_b] = st_balance(x,2048);
%     [fr_SPC, fi_SPC] = st_phase_coherence(x,2048);
% %     [fr_w, fi_w] = st_width(x,2048);
%     %Plot the features
%     figure
%     subplot(211)
%     plot(fr_SPC(1:end-1),'k')
%     hold on
%     plot(fr_b(1:end-1),'k--')
% %     plot(fr_w(1:end-1))
%     hold off
%     plot_title = sprintf("Stereo Features for %s",strrep(filename(1:end-4),'_',' '));
%     title(plot_title)
%     axis([1 (length(fr_b)-1) -1.1 1.1]);
%     xlabel('Time (Analysis Frame)');
%     ylabel('Feature Value');
%     legend('Coherence','Balance','location','best');
%
% filename = 'White_MS_fade.wav';
% [x,fs] = audioread([pathInput filename]);
%     %Generate DC signal
%     % len = 44100;
%     % x(1:len,1) = ones(len,1);
%     % x(1:len,2) = zeros(len,1);
%     %Create features
%     [fr_b, fi_b] = st_balance(x,2048);
%     [fr_SPC, fi_SPC] = st_phase_coherence(x,2048);
%     %Plot the features
%     subplot(212)
%     plot(fr_SPC(1:end-1),'k')
%     hold on
%     plot(fr_b(1:end-1),'k--')
%     plot_title = sprintf("Stereo Features for %s",strrep(filename(1:end-4),'_',' '));
%     title(plot_title)
%     axis([1 (length(fr_b)-1) -1.1 1.1]);
%     xlabel('Time (Analysis Frame)');
%     ylabel('Feature Value');
%     legend('Coherence','Balance','location','best');
%     x0=10;
%     y0=10;
%     width=300;
%     height=300;
%     set(gcf,'units','points','position',[x0,y0,width,height])
%
%
%
%
%
%     %Plot the stereo signal
% %     x = x./max(max(abs(x)));
% %     x_plot = [x(:,1)+1, x(:,2)-1];
% %     t = (1:length(x_plot))/fs;
% %     figure
% %     plot([t',t'],x_plot);
% %     axis([0 max(t) min(min(x_plot))*1.1 max(max(x_plot))*1.1])
% %     plot_title = sprintf("%s",strrep(filename(1:end-4),'_',' '));
% %     title(plot_title);
% %
% %     set(gca, 'YLim', [min(min(x_plot))*1.1 max(max(x_plot))*1.1],...
% %         'YTick', -2:1:2,...
% %         'YTickLabel', {'-1     ', 'Right Amplitude','1, -1', 'Left Amplitude', '1'})
% %
% %     set(gcf,'units','points','position',[x0,y0,width,height])

%% ------------ Testing Stereo Phase Vocoder functions --------------

% pathInput = '../Audio_test_files/';
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
% pathInput = '../Audio_test_files/';
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

% pathInput = '../Audio_test_files/';
% filename = 'Male_Speech.wav';
%
% [x,fs] = audioread([pathInput filename]);
% % x = sum(x,2)/max(sum(x,2));
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

% pathInput = '../Audio_test_files/';
% % filename = 'Male_Speech.wav';
% filename = 'Solo_flute_2.wav';
%
% [x,fs] = audioread([pathInput filename]);
% x = sum(x,2)/max(sum(x,2));
%
% y = ZFR(x, 1, fs, 1);
% % s, N_CALC, fs, N_scale
% figure('Position',[0 0 500 300])
% H = line((1:length(y))/fs,y,'LineStyle',':','Color','black');
% hold on
% plot((1:length(x))/fs,x,'k');
% hold off
% % legend('Epochs','Speech signal','Location','best')
% title('Epoch Locations');
% xlabel('Time(s)')
% ylabel('Amplitude')
% % axis([0.165 0.215 -0.01 1.01]) %Male Speech
% axis([2.909 2.923 -0.01 0.27]) %Flute
% savename = sprintf('../%s_Epochs',filename(1:end-4));
% print(savename,'-dpng')
% print(savename,'-depsc')

%% -------------ESOLA Testing---------------------

% pathInput = '../Audio_test_files/';
% %  filename = 'Male_Speech_ESOLA.wav';
% %   filename = 'mrds0_sx447.wav';
% % filename = 'fadg0_sa1.wav';
% % filename = 'female_waar.wav';
% % filename = 'Wash.wav';
% %filename = 'Female_opera_excerpt.wav';
% % filename = 'Male_Speech.wav';
% % filename = 'Alto_Sax_08.wav';
% % filename = 'Drums_2.wav';
% % filename = 'Mexican_Flute_02.wav';
% filename = 'Solo_flute_2.wav';
%
% [x,fs] = audioread([pathInput filename]);
% x = sum(x,2)/max(sum(x,2));
% TSM = 0.8;
% ms = 20;
% N = ms*(10^-3)*fs;
%
% y_ESOLA = ESOLA(x, N, TSM, fs);
% [y_FESOLA, N_ZFR] = FESOLA(x, N, TSM, fs);
%
% figure
% subplot(211)
% % soundsc(x,fs)
% plot(x)
% title('Original');
% % pause((length(x)/fs)*1.1);
%
% subplot(212)
% soundsc(y_ESOLA,fs)
% plot(y_ESOLA)
% title('ESOLA');
%
% %  audiowrite([filename(1:end-4) '_k2_20ms_50per.wav'],y_ESOLA,fs);
%
% % LogSpectrogram(x,fs,50,2);
% % title('Original')
% % LogSpectrogram(y_ESOLA,fs,50,2);
% % title('ESOLA');

%% ------------Checking linear interpolation---------

% TSM = 0.9;
% ak = rand(10,1);
% a = 1/TSM;
%
% old_points = (1:length(ak))-1;  %-1 to 0 index
% new_points = round(a*old_points)+1; %+1 to 1 index
%
% ak_hat = zeros(1,ceil(length(ak)*a));
% ak_hat(new_points) = ak(old_points+1);
%
% count = 0;
%
% n = find(ak_hat);
% n_ = find(~ak_hat);
% if(TSM<0.5)
%     for k = 1:length(n)-1
%         ak_hat(n(k):n(k+1)) = linspace(ak_hat(n(k)),ak_hat(n(k+1)),n(k+1)-n(k)+1);
%     end
% else
%     for k = 1:length(n_)-1
% %         ak_hat(n_(k)-1:n_(k)+1) = linspace(ak_hat(n_(k)-1),ak_hat(n_(k+1)-1),n_(k+1)-n_(k)+1);
%         ak_hat(n_(k)) = (ak_hat(n_(k)-1)+ak_hat(n_(k+1)-1))/2;
%     end
% end
%
% subplot(211)
% plot(ak)
% title('Original')
% subplot(212)
% plot(ak_hat)
% title('Linear interpolation')

%% -----------------Testing mel_filterbank generation---------------------

% [ H ] = mel_filterbank( 88, 22050 8096, 44100*6 );
% %Plot the filterbanks
% figure
% plot(H');
% %Plot the sum of the filterbanks
% figure
% plot(sum(H,1))

%% -------------uTVS Testing---------------------

pathInput = '../Subjective_Testing/Source/Objective/';
filename = 'Alto_Sax_15.wav';
%filename = 'mrds0_sx447.wav';

[x,fs] = audioread([pathInput filename]);
% x = rand(100000,1)*2-1;
% fs = 44100;
x = sum(x,2)/max(abs(sum(x,2)));
TSM = [0.9842,1.312];
% y = uTVS(x, fs, TSM(1));
y = uTVS_batch(x, fs, TSM, 'Alto_Sax_15');

% TSM = linspace(0.9, 1.1, 20 );
% [~] = uTVS_batch( x, fs, TSM, 'White_noise' );
% figure
% subplot(211)
% plot(x)
% title('Original');
%
% subplot(212)
% soundsc(y_uTVS,fs)
% plot(y_uTVS)
% title('uTVS');
%
% LogSpectrogram(y_uTVS,fs,50,2);
% t = sprintf('uTVS at %g percent',TSM*100);
% title(t);
% f = sprintf('uTVS_%g_percent',TSM*100);
% % print(f,'-dpng');
% f = [f '.wav'];
% % audiowrite(f,y_uTVS,fs);

%% ------------------Checking filterbank.m------------------
% N = 2048;
% K = 10;
% c = linspace(1,1024,12);
% bank = filterbank(c(2:end-1),N);
% plot(bank')

%% ------------------Checking my_fft.m----------------------
% interations = 50;
% nfreq = [2,5,11,17,29];
% low = 10;
% high = 13;
% t_fft = zeros(high-low+1,1);
% t_my_fft = zeros(high-low+1,1);
% for n = low:high
% %     x = rand(1,2^n);
% n
%     xf = zeros(2^n,length(nfreq));
%     for q = 1:length(nfreq)
%         xf(:,q) = sin((2*pi*nfreq(q).*(1:2^n))/2^n);
%     end
%     x = sum(xf,2);
%     tic
%     for k = 1:interations
%         y_fft = fft(x,length(x));
%     end
%     t_fft(n-low+1) = toc;
% %     figure(1)
% %     plot(abs(y_fft(2:end/2+1)))
%     tic
%     for k = 1:interations
%         y_my_fft = my_fft(x, length(x));
%     end
%     t_my_fft(n-low+1) = toc;
% %     figure(2)
% %     plot(abs(y_my_fft(2:end/2+1)))
%
% end
% figure(3)
% semilogy(2.^(low:high),t_fft/interations)
% hold on
% semilogy(2.^(low:high),t_my_fft/interations)
% hold off
% legend('fft','my\_fft','Location','best')
% %
% % figure
% % subplot(211)
% % plot(diff')
% % subplot(212)
% % plot(abs(diff'))

%% -------------FESOLA Testing---------------------

% % TSM = [0.3838 0.4427 0.5383 0.6524 0.7821 0.8258 0.9961 1.246 1.496 1.684 1.8891];
% TSM = [0.3838 0.5383 0.7821 0.9961 1.496 1.8891];
% pathInput = '../Audio_test_files/FESOLA_TEST/';
% % pathInput = '../Audio_test_files/FESOLA_Sine/';
% % pathInput = '../Audio_test_files/';
% pathOutput = '../WIP/FESOLA/FESOLA_Output/';
% d = dir(pathInput);
% % TSM = 0.3838;
% % TSM=1;
% ms = 20;
%
% for n = 3:numel(d)
%     [x,fs] = audioread([pathInput d(n).name]);
%     %     [x,fs] = audioread('../Audio_test_files/FESOLA_Speech/Female_5.wav');
%     x = sum(x,2)/max(sum(x,2));
%     N = 2^nextpow2(ms*(10^-3)*fs);
%     final_ms = N/44.1;
%     disp(d(n).name);
%     for t = 1:length(TSM)
%
%         [y_FESOLA, ZFR_N] = FESOLA( x, N, TSM(t), fs );
%         out_filename = sprintf('%s%s_FESOLA_%g_ms_%g_per_ZFR_%d.wav',pathOutput,d(n).name(1:end-4),final_ms,TSM(t)*100, ZFR_N);
%         audiowrite(out_filename,y_FESOLA,fs);
%     end
% end
%
% figure
% subplot(211)
% % soundsc(x,fs)
% plot(x)
% title('Original');
% % pause((length(x)/fs)*1.1);
%
% subplot(212)
% soundsc(y_FESOLA,fs)
% plot(y_FESOLA)
% title('FESOLA');
%
%
%
% LogSpectrogram(x,fs,50,2);
% title('Original')
% LogSpectrogram(y_ESOLA,fs,50,2);
% title('ESOLA');

%% -------------Checking SliceTSM---------------------

% file1 = '../Subjective_Testing/Source/Solo/Perc_2.wav';
% file2 = '../Subjective_Testing/Source/Solo/Tambourine_01.wav';
% % file2b = '../Subjective_Testing/Source/Solo/Tambourine_02.wav';
% % file3 = '../Audio_test_files/Transient.wav';
% file4 = '../Subjective_Testing/Source/Solo/Drums_2.wav';
% % file5 = '../Subjective_Testing/Source/Music/Chiptune.wav';
% % file6 = '../Subjective_Testing/Source/Solo/Alto_Sax_07.wav';
% % file7 = '../Subjective_Testing/Source/Solo/Alto_Sax_08.wav';
% file8 = '../Subjective_Testing/Source/Solo/Bass_2.wav';
% file9 = '../Audio_test_files/FFperc110-001.wav';
% file10 = '../Audio_test_files/FFperc110-003.wav';
% file11 = '../Audio_test_files/Dubstep Slow-126bpm.wav';
% [x,fs] = audioread(file11);
% x = sum(x,2)/max(abs(sum(x,2)));
% TSM = 0.8;
% % x = [0 0 0 0 5 10 10 9 9 8 8 7 7 6 6 5 5 4 4 3 3 2 2 1 1 0 0 0 0];
% % x(1:2:end) = x(1:2:end)*-1;
% % x = x';
%
% sensitivity = 0.5;
%
% y_SliceTSM = SliceTSM_overlap(x, fs, TSM, sensitivity);
%
% figure
% plot(x+1);
% hold on
% plot(y_SliceTSM-1);
% hold off
%
% soundsc(x,fs)
% pause(length(x)/fs)
% soundsc(y_SliceTSM,fs);
% % subplot(211)
% % plot(x)
% % title('Original');

%% -----------LPC Residual--------------

% file1 = '../Subjective_Testing/Source/Solo/Perc_2.wav';
% file2 = '../Subjective_Testing/Source/Solo/Tambourine_01.wav';
% file2b = '../Subjective_Testing/Source/Solo/Tambourine_02.wav';
% file3 = '../Audio_test_files/Transient.wav';
% file4 = '../Subjective_Testing/Source/Solo/Drums_2.wav';
% file5 = '../Subjective_Testing/Source/Music/Chiptune.wav';
% file6 = '../Subjective_Testing/Source/Solo/Alto_Sax_07.wav';
% file7 = '../Subjective_Testing/Source/Solo/Alto_Sax_08.wav';
% file8 = '../Subjective_Testing/Source/Solo/Bass_2.wav';
% file9 = '../Audio_test_files/FFperc110-001.wav';
% file10 = '../Audio_test_files/FFperc110-003.wav';
% file11 = '../Audio_test_files/Dubstep Slow-126bpm.wav';
% [x,fs] = audioread(file4);
% x = sum(x,2)/max(abs(sum(x,2)));
%
% N = 1024;
% H = N/4;
% [est_x, est_x_2] = LPC_residual( x, fs, N, H );
%
% % mesh(est_x);
% % view([0 90])
%
% est_x_vec = sum(est_x);
% abslogestx = abs(log10(est_x_vec));
% abslogestx_diff = abslogestx(2:end) - abslogestx(1:end-1);
% est_x_2_vec = sum(abs(est_x_2));
%
% figure
% subplot(311)
% plot(x)
% subplot(312)
% plot(abslogestx)
% subplot(313)
% plot(abslogestx_diff)

%% ------------------Test the onset detect function------------

% file2 = '../Subjective_Testing/Source/Solo/Alto_Sax_08.wav';
%
% [x,fs] = audioread(file2);
% x = sum(x,2)/max(abs(sum(x,2)));
%
% [ onsets ] = onset_detect( x, fs );
%
% figure
% plot(x)
% hold on
% y_line = ones(size(onsets));
% line([onsets;onsets],[-1*y_line ; y_line],'Color','red');
% hold off

%% PV processing

%  pathInput = '../Audio_test_files/';
%  filename = 'Alto_Sax_16.wav';
%
% [x,fs] = audioread([pathInput filename]);
%
% TSM = 0.41;
% N = 2048;
% y_n = PV(x, N, TSM);
%
% soundsc(y_n,fs);
% audiowrite('../Alto_sax_16_PV_41_per.wav',y_n,fs);


%% ------------------------ ESOLA vs FESOLA Audio File Generation ----------------


% % TSM = [0.3838 0.4427 0.5383 0.6524 0.7821 0.8258 0.9961 1.246 1.496 1.684 1.8891];
% TSM = [0.5383 0.7821];
% pathInput = '../WIP/FESOLA/AudioIn/';
% % pathInput = '../Audio_test_files/FESOLA_Sine/';
% % pathInput = '../Audio_test_files/';
% pathOutput = '../WIP/FESOLA/AudioOut/';
% d = dir(pathInput);
% % TSM = 0.3838;
% % TSM=1;
% ms = 20;
%
% for n = 3:numel(d)
%     [x,fs] = audioread([pathInput d(n).name]);
%     x = sum(x,2)/max(sum(x,2));
%     N = 2^nextpow2(ms*(10^-3)*fs);
%     final_ms = N/44.1;
%     disp(d(n).name);
%     for t = 1:length(TSM)
%         [y_ESOLA] = ESOLA( x, N, TSM(t), fs );
%         out_filename = sprintf('%s%s_ESOLA_%g_per.wav',pathOutput,d(n).name(1:end-4),TSM(t)*100);
%         audiowrite(out_filename,y_ESOLA,fs);
%         [y_FESOLA, ~] = FESOLA( x, N, TSM(t), fs );
%         out_filename = sprintf('%s%s_FESOLA_%g_per.wav',pathOutput,d(n).name(1:end-4),TSM(t)*100);
%         audiowrite(out_filename,y_FESOLA,fs);
%     end
% end

%% ---------------- Generate simple test files ---------------
% global debug_var
% debug_var = 1;
% General.TSM = 0.56;
% General.version = 'basic';
% General.model = 'FFT';
% General.N = 2048;
% General.StepSize = 1024;
% General.Testname = 'Sine';
% General.fs = 44100;
% % [x,General.fs] = audioread('../Audio_test_files/Sine_Fade_Left.wav');
% % x = mean(x,2);
%
% % x = rand(4*General.fs,1);
% t = (1:(4*General.fs))/General.fs;
% f = 1000;
% x = sin(2*pi*f*t');
% x = [zeros(General.fs,1) ; x ; zeros(General.fs,1)];
% % x = x-mean(x);
% y = PV(x,2048,General.TSM);
%
% addpath(genpath('../Objective_MOQ/Functions'));
%
%
% [x, y, General] = Signal_Prep(x, y, General);
%
% [p_ave, p_std] = Phasiness2(x,y,General);
