close all
clear all
clc

addpath('../Functions');

%% --------  Comparison with baseline filterbank method ------------
pathInput = 'AudioIn/';
pathOutput = 'AudioOut/';
audio_file = 'Male_Speech.wav';

%Load audio file
[x, FS] = audioread([pathInput audio_file]);
x = sum(x,2);

num_regions = [round(linspace(2,500,50)), 750 1025];
t_prop = zeros(length(num_regions),1);
t_fbank = zeros(length(num_regions),1);
N = 2048;
for k = 1:length(num_regions)
    K = num_regions(k);
    region.TSM = linspace(0.5,2,K);
    region.TSM = ones(K,1);
    region.upper = round(linspace(1,N/2+1,K+1));
    region.upper = region.upper(2:end);
    tic
        y = FDTSM( x, N, region );
    t_prop(k) = toc;
    
    region_info.TSM = region.TSM;
    tic
        y_fbank = FDTSM_Filterbank( x, N, region_info );
    t_fbank(k) = toc;    
    
    %Create the output name
    output_filename = [audio_file(1:end-4) '_' sprintf('%d_regions',num_regions(k)) '_prop_FDTSM.wav'];
    %Save audio file
    audiowrite([pathOutput output_filename], y, FS);
    %Create the output name
    output_filename = [audio_file(1:end-4) '_' sprintf('%d_regions',num_regions(k)) '_fbank_FDTSM.wav'];
    %Save audio file
    audiowrite([pathOutput output_filename], y_fbank, FS);

end

save('Prop_vs_Filterbank_Times.mat','t_prop','t_fbank','num_regions');

figure
semilogy(num_regions, t_fbank, 'k:')
hold on
semilogy(num_regions, t_prop, 'k')
hold off
set(gca, 'FontName', 'Times New Roman')
title('FDTSM Processing Time');
xlabel('Number of Regions');
ylabel('Time (sec)');
legend('Filterbank Method','Proposed FFT Method','Location','SouthEast');

x0=100;
y0=100;
width=300;
height=200;
set(gcf,'units','points','position',[x0,y0,width,height])

%Time taken
%N 2048, R = 1025;
%Prop = 13.50 sec
%Fbank = 


%% ---------------------------Hold number of regions, increase frame length----------------------
%Processed on Xeon 6 Core 1.6 GHz
%Direct calculation uses all 6 cores
%FFT calculation uses a single core
% trials = 50;
% low = 6;
% high = 12;
% t_direct = zeros(high-low+1,1);
% t_fft = zeros(high-low+1,1);
% t_my_fft = zeros(high-low+1,1);
% t_online_fft = zeros(high-low+1,1);
% for R = low:high
%     N = 2^R
%     x = rand(N,1);
%     X = zeros(N,1);
%     disp('Direct Method');
%     exp_arr = exp((-1i*2*pi*(0:N-1)'*(0:N/2))/N); %n'* k
%     tic
%     for t = 1:trials
%         xr = repmat(x,1,size(exp_arr,2));
%         X = sum(xr.*exp_arr,2);
%     end
%     t_direct(R-low+1) = toc;
%
%     %Figure 3
%     num_regions = N/2+1;
%     region.TSM = linspace(0.5,2,num_regions);
%     region.upper = 1:(N/2+1);
%
%     %Calculate the lower bounds of each region
%     region.lower = [1 region.upper(1:end-1)+1];
%     region.num_regions = length(region.TSM);
%
%     disp('FFT Method');
%     tic
%     for t = 1:trials
%         for r = 1:region.num_regions
%             %Extract frame
%             FRAME_CURRENT = fft(x, length(x));
%             %Colate only the appropriate frequency bins
%             FRAME_COMP(region.lower(r):region.upper(r)) = FRAME_CURRENT(region.lower(r):region.upper(r));
%         end
%     end
%     t_fft(R-low+1) = toc;
%
%     disp('my FFT Method');
%     tic
%     for t = 1:trials
%         for r = 1:region.num_regions
%             %Extract frame
%             FRAME_CURRENT = my_fft(x, length(x));
%             %Colate only the appropriate frequency bins
%             FRAME_COMP(region.lower(r):region.upper(r)) = FRAME_CURRENT(region.lower(r):region.upper(r));
%         end
%     end
%     t_my_fft(R-low+1) = toc;
%
% end

% Save and Plot the results
% save('Frequency_Domain_Transformation_Time.mat', 't_direct','t_fft','t_my_fft');
% figure
% semilogy(2.^(low:high),t_direct/trials,'k');
% hold on
% semilogy(2.^(low:high),t_fft/trials,'k--');
% semilogy(2.^(low:high),t_my_fft/trials,'k:');
% hold off
% set(gca, 'FontName', 'Times New Roman')
% xlabel('Frame size (N)');
% ylabel('Time (s)');
% title('Frequency Domain Transformation Time');
% legend('Direct Calculation','FFTW Calculation','Radix-2 FFT Calculation','Location','best');
% x0=100;
% y0=100;
% width=300;
% height=200;
% set(gcf,'units','points','position',[x0,y0,width,height])

%% --------------------- Calculate regions breakeven point for various N  ---------------------
% trials = 10;
% low_b = 5;
% high_b = 13;
% t_direct = zeros(high_b-low_b+1,1);
% t_fft = zeros(high_b-low_b+1,1);
% N_vals = zeros(high_b-low_b+1,1);
% r_vals = zeros(high_b-low_b+1,1);
% for b = low_b:high_b %increasing N
%     N = 2^b;
%     R = 2;
%     low_r = 2;
%     high_r = N/4;
%     t_f = 0;
%     t_d = 0;
%     while (t_f<=t_d && R<high_r)
%         R = R+round(N/100);
%         x = rand(N,1);
%         X = zeros(N,1);
%         disp('Direct Method');
%         exp_arr = exp((-1i*2*pi*(0:N-1)'*(0:N/2))/N); %n'* k
%         tic
%         for t = 1:trials
%             xr = repmat(x,1,size(exp_arr,2));
%             X = sum(xr.*exp_arr,2);
%         end
%         t_d = toc;
%
%         %Figure 3
%         region.upper = round(linspace(1,(N/2+1),R+1));
%         region.upper = region.upper(2:end);
%
%         %Calculate the lower bounds of each region
%         region.lower = [1 region.upper(1:end-1)+1];
%         FRAME_COMP = zeros(N/2+1,1);
%         disp('FFT Method');
%         tic
%         for t = 1:trials
%             for r = 1:R
%                 %Extract frame
%                 FRAME_CURRENT = fft(x, length(x));
%                 %Colate only the appropriate frequency bins
%                 FRAME_COMP(region.lower(r):region.upper(r)) = FRAME_CURRENT(region.lower(r):region.upper(r));
%             end
%         end
%         t_f = toc;
%
%     end
%     t_direct(b-low_b+1) = t_d/trials;
%     t_fft(b-low_b+1) = t_f/trials;
%     N_vals(b-low_b+1) = N
%     r_vals(b-low_b+1) = R
% end
% %Plot the data
% save('Breakeven_point.mat', 't_direct','t_fft', 'N_vals', 'r_vals');
% %Fit a linear equation to the data
% [F0, G] = fit(N_vals,r_vals,'poly1')
% 
% figure
% plot(F0,'k', N_vals,r_vals, 'k.')
% set(gca, 'FontName', 'Times New Roman')
% xlabel('Frame Length (N)');
% ylabel('Regions');
% title('Break Even Point for DFT Faster than FFT')
% axis tight
% x0=100;
% y0=100;
% width=300;
% height=200;
% set(gcf,'units','points','position',[x0,y0,width,height])


%% --------  Comparison between Direct and FFT methods ------------
% pathInput = 'AudioIn/';
% pathOutput = 'AudioOut/';
% audio_file = 'Male_Speech.wav';
%
% %Load audio file
% [x, FS] = audioread([pathInput audio_file]);
%
% N = 2048;
% trials = 50;
% num_regions = round(linspace(2,N/2,200));
% x = sum(x,2);
% t_direct = zeros(length(num_regions),1);
% t_fft = zeros(length(num_regions),1);
% for k = 1:length(num_regions)
%     K = num_regions(k);
%     %    region.TSM = linspace(0.5,2,K);
%     region.TSM = ones(K,1);
%     region.upper = round(linspace(1,N/2+1,K+1));
%     region.upper = region.upper(2:end);
%
%     %Process Direct
%     fprintf('Direct Method, %d Regions, ',K)
%     tic
%     for t = 1:trials
%         y_direct = FDTSM_Direct( x, N, region );
%     end
%     t_direct(k) = toc/trials;
%
%     %Process FFT
%     fprintf('FFT Method, %d Regions\n',K)
%     tic
%     for t = 1:trials
%         y_fft = FDTSM( x, N, region );
%     end
%     t_fft(k) = toc/trials;
%
% end
% save('FDTSM_using_DFT_vs_FFT_Time.mat','t_direct','t_fft','num_regions','N','trials','audio_file');
%
% figure
% plot(num_regions, t_direct, 'k')
% hold on
% plot(num_regions, t_fft, 'k:')
% hold off
% set(gca, 'FontName', 'Times New Roman')
% title('FDTSM Processing Time for Number of Regions');
% xlabel('Number of Regions');
% ylabel('Time (sec)');
% axis tight
% legend('Direct Method','FFT Method','Location','best');
%
% x0=100;
% y0=100;
% width=300;
% height=200;
% set(gcf,'units','points','position',[x0,y0,width,height])
% %
% % %Cross over point at N/4 regions for Tim-UNI-PC

%% Processing Examples

% pathInput = 'AudioIn/';
% pathOutput = 'AudioOut/';
% 
% %Figure White Linear 0.5-2
% disp('White Linear 0.5-2')
% audio_file = 'White.wav';
% [x, FS] = audioread([pathInput audio_file]);
% N = 2048;
% num_regions = N/2+1;
% region.TSM = linspace(0.5,2,num_regions);
% region.upper = 1:(N/2+1);
% y_white_linear = FDTSM( x, N, region );
% LinSpectrogram(y_white_linear,FS,50,2);
% audiowrite([pathOutput 'FDTSM_White_Linear.wav'], y_white_linear, FS);
% 
% %Figure White Random N/2+1 regions
% disp('White Random')
% audio_file = 'White.wav';
% [x, FS] = audioread([pathInput audio_file]);
% N = 2048;
% num_regions = N/2+1;
% low=0.5;
% high=2;
% region.TSM = (low + (high-low).*rand(num_regions,1))';
% slow_bin = 50;
% region.TSM(slow_bin) = 0.2;
% region.upper = 1:(N/2+1);
% y_random_TSM_max_regions = FDTSM( x, N, region );
% LogSpectrogram(y_random_TSM_max_regions,FS,50,2);
% audiowrite([pathOutput 'FDTSM_White_Random.wav'], y_random_TSM_max_regions, FS);
% 
% %Figure White Random 32 regions
% disp('White Random 32')
% audio_file = 'White.wav';
% [x, FS] = audioread([pathInput audio_file]);
% N = 2048;
% num_regions = 32;
% low=0.1;
% high=1;
% region.TSM = (low + (high-low).*rand(num_regions,1))';
% region.upper = ceil(linspace(1,N/2+1,num_regions));
% y_random_TSM_32_regions = FDTSM( x, N, region );
% LinSpectrogram(y_random_TSM_32_regions,FS,50,2);
% audiowrite([pathOutput 'FDTSM_White_Random_32_Regions.wav'], y_random_TSM_32_regions, FS);
% 
% %Figure 6
% %Use FDTSM_GUI_example script
% 
% %Figure Male Speech
% disp('Speech')
% audio_file = 'Male_Speech.wav';
% [x, FS] = audioread([pathInput audio_file]);
% N = 2048;
% region.TSM = [1 0.9 1];
% region.upper = [N/256 N/16 N/2+1];
% y_speech = FDTSM( x, N, region );
% LogSpectrogram(y_speech,FS,50,2);
% audiowrite([pathOutput 'FDTSM_Male_Speech.wav'], y_speech, FS);
% 
% %Figure Reconstruction
% disp('Reconstruct')
% x = y_white_linear;
% N = 2048;
% num_regions = N/2+1;
% region.TSM = 1./linspace(0.5,2,num_regions);
% region.upper = 1:(N/2+1);
% y_recon = FDTSM( x, N, region );
% LinSpectrogram(y_recon,FS,50,2);
% audiowrite([pathOutput 'FDTSM_White_Recon.wav'], y_recon, FS);


%% Comparison of output for proposed and filterbank
% pathInput = 'AudioOut/';
% file1 = 'White_2_regions_fbank_FDTSM.wav';
% file2 = 'White_2_regions_prop_FDTSM.wav';
% 
% [x_fbank, fs] = audioread([pathInput file1]);
% [x_prop, FS] = audioread([pathInput file2]);
% 
% LinSpectrogram(x_fbank, fs, 50, 2);
% LinSpectrogram(x_prop, FS, 50, 2);


