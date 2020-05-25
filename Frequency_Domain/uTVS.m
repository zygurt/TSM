function [ y ] = uTVS( x, fs, TSM )
%[ y ] = uTVS( x, fs, TSM )
%muTVS proposed by Sharma et al., Mel-Scale Sub-band Modelling for
%Perceptually Improved Time-Scale Modification of Speech Audio Signals,
%2017
%   x is the input signal
%   fs is the sampling frequency
%   TSM is the TSM ratio (beta) 0.5 = 50%, 1 = 100% and 2.0 = 200% speed
%This method generates Instantaneous Amplitude (IA) and Instantaneous Phase (IP) without
%the use of fft, removing the quasi-stationarity requirement
%Printing to the screen is done to allow user to see that processing is
%taking place.  This is a slow method of processing.

% Tim Roberts - Griffith University 2018
addpath('../Functions');
%Initial variables
K = 2*floor(fs/1000);   %32 for fs=16kHz
N = 2^nextpow2(fs/8);   %2048 for fs=16kHz
S = N/4;
a = 1/TSM;
oversample = 6;
%% --------------------------Analysis------------------------------
%This section splits the input signal into K band passed signals
disp('Analysis');
%Oversample
xr = resample(x,oversample,1);
fo = fs*oversample;
%Create window (Hann)
w = 0.5*(1 - cos(2*pi*(0:N-1)'/(N-1)));
%Frame the input
xw = buffer(xr, N, N-S);
%Window the frames
xw = xw.*repmat(w,1,size(xw,2));
%Convert to frequency domain
XW = fft(xw,N);
%Generate Filterbanks
disp('    Generate Filterbanks');
H = mel_filterbank(K,fs/2,N,fo);
% figure
% plot(H)
%Take the magnitude of the first half of the fft
XW_crop = XW(1:N/2+1,:);
% XW_mag = abs(XW(1:N/2+1,:));
% XW_phase = angle(XW(1:N/2+1,:));
%Prepare framed filterbank output
XWF = zeros(size(XW_crop,1),K,size(XW_crop,2));
%Mulitply through with the filterbanks.
disp('    Multiply filterbanks through signal');
for k = 1:K
    fprintf('%d, ',k);
    for f = 1:size(XW,2)
        XWF(:,:,f) = repmat(XW_crop(:,f),[1,K]).*H'; %Magnitude filter
%         XWF(:,:,f) = XWF(:,:,f).*exp(repmat(XW_phase(:,f),[1,K])); %Combine back with phase
    end
end
%Reconstruct second half of the signal
xwf_recon = real(ifft([XWF;conj(XWF(end-1:-1:2,:,:))]));
%Prepare filterbank channels
xwf = zeros(size(xwf_recon,3)*S+1.75*N,K); %Need to make this longer.  Janky solution for now.
%Create the output window
wo = repmat(w,1,size(xwf_recon,2));
%Overlap add the signal back together
disp('    Overlap Add the signal back together');
for f = 1:size(XWF,3)
    xwf((f-1)*S+1:(f-1)*S+N,:) = xwf((f-1)*S+1:(f-1)*S+N,:)+xwf_recon(:,:,f).*wo;
end
%At this point, xwf_jl is a K channel signal version of the original x
%input signal

%% --------------------------Modification------------------------------
%For each bank:
disp('Modification')
%Hilbert Transform to extract IA and IP
disp('    Hilbert');
xak_h = hilbert(xwf);
%Calculate the Instantaneous Amplitude and Phase
ak = abs(xak_h);
phik = unwrap(angle(xak_h));
%Interpolate missing values
disp('    Interpolate each filterband: ')
t_original = (1:size(xwf,1))/fo;
t_scaled = linspace(min(t_original),max(t_original),round(a*size(xwf,1)));
ak_hat_i = zeros(length(t_scaled),K);
phik_hat_i = zeros(length(t_scaled),K);

for k = 1:K
    ak_hat_i(:,k) = interp1(t_original, ak(:,k), t_scaled);
    phik_hat_i(:,k) = a*interp1(t_original, phik(:,k), t_scaled);
    fprintf('%d, ',k);
end
%Multiply output IA and IP
x_hat = ak_hat_i.*cos(phik_hat_i);
%% --------------------------Synthesis------------------------------
disp('Synthesis')
%Combine the filterbank audio signals
x_hat_sum = sum(x_hat,2);
%Resampling the output
y = resample(x_hat_sum,1,oversample);
%Normalise the output
y=y/max(abs(y));
disp('File Processing Complete');
end
