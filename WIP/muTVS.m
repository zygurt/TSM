function [ y ] = muTVS( x, fs, TSM )
%[ y ] = muTVS( x, fs, TSM )
%muTVS proposed by Sharma et al., Mel-Scale Sub-band Modelling for
%Perceptually Improved Time-Scale Modification of Speech Audio Signals,
%2017
%   x is the input signal
%   fs is the sampling frequency
%   TSM is the TSM ratio 0.5 = 50%, 1 = 100% and 2.0 = 200% speed

% Tim Roberts - Griffith University 2018

%Generate Instantaneous Amplitude (IA) and Instantaneous Phase (IP) without
%the use for fft, which assumes quasi-stationarity

% if TSM > 1
%     disp('Only stretching works for now');
%     y = 0;
%     return
% end

%Initial variables
K = 32;
K = 2*floor(fs/1000);
N = 2048;
S = N/4;
a = 1/TSM;
oversample = 4;
%% --------------------------Analysis------------------------------
%This section splits the input signal into K band passed signals
disp('Analysis');
%Need to add oversampling in here.  Not sure what parameters.
xr = resample(x,oversample,1);
Fo = fs*oversample;
%Create window
w = 0.5*(1 - cos(2*pi*(0:N-1)'/(N-1)));
%Frame the input
xw = buffer(xr, N, N-S);
%Window the frames
xw = xw.*repmat(w,1,size(xw,2));
%Convert to frequency domain
XW = fft(xw,N);
%Generate Filterbanks
disp('    Generate Filterbanks');
H_JL = [zeros(K,1) , msf_filterbank(K,Fo,0,fs/2,N)];
%Take the first half of the fft
XW_crop = XW(1:N/2+1,:);
%Prepare framed filterbank output
XWF_JL = zeros(size(XW_crop,1),K,size(XW_crop,2));
%Mulitply through with the filterbanks.
disp('    Multiply filterbanks through signal');
for k = 1:K
    for f = 1:size(XW,2)
        XWF_JL(:,:,f) = repmat(XW_crop(:,f),[1,K]).*H_JL';
    end
end
%Reconstruct second half of the signal
XWF_JL_recon = real(ifft([XWF_JL;conj(XWF_JL(end-1:-1:2,:,:))]));
%Prepare filterbank channels
xwf_jl = zeros(size(XWF_JL_recon,3)*S+1.75*N,K); %Need to make this longer.  Janky solution for now.
%Create the output window
wo = repmat(w,1,size(XWF_JL_recon,2));
%Overlap add the signal back together
disp('    Overlap Add the signal back together');
for f = 1:size(XWF_JL,3)
    xwf_jl((f-1)*S+1:(f-1)*S+N,:) = xwf_jl((f-1)*S+1:(f-1)*S+N,:)+XWF_JL_recon(:,:,f).*wo;
end

%At this point, xwf_jl is a K channel signal version of the original x
%input signal

%% --------------------------Modification------------------------------
%For each bank:
disp('Modification')
%Hilbert Transform to extract IA and IP
disp('    Hilbert');
xak_h = hilbert(xwf_jl); 

%Calculate the Instan
ak = abs(xak_h);
phik = unwrap(angle(xak_h));

%Time scale through interpolation
ak_hat = zeros(ceil(length(ak)*a),K);
phik_hat = zeros(ceil(length(phik)*a),K);

old_points = (1:length(ak));
new_points = round(a*(old_points-1))+1; %-1 to 0 index old points, +1 to 1 index new-points
disp('    Assign new time scale');
%Assigning the new time scale adds an additional signal.  Not sure how or
%from where.
%As the speed decreases, all partials in each filterbank converge to center
%frequency of the bin.
ak_hat(new_points,:) = ak(old_points,:);
phik_hat(new_points,:) = a*phik(old_points,:);

ak_hat_i = zeros(length(ak_hat),K);
phik_hat_i = zeros(length(phik_hat),K);

disp('    Interpolate each filterband: ')
for k = 1:K
    ak_hat_i(:,k) = linear_interp2(ak_hat(:,k), TSM);
    phik_hat_i(:,k) = linear_interp2(phik_hat(:,k), TSM);
    disp(k)
end

%Multiply output IA and IP
x_hat = ak_hat_i.*cos(phik_hat_i);

%% --------------------------Synthesis------------------------------
disp('Synthesis')
%Combine the filterbank audio signals
% xw_jl_sum = sum(xwf_jl,2);
x_hat_sum = sum(x_hat,2);
%Need to add resampling the output.

y = resample(x_hat_sum,1,oversample);

y=y/max(y);
end

%Using Aaron's mel spaced filterbanks
% [H_AN, ~, ~] = melfbank(K, N/2+1, fs);
% XWF_AN = zeros(size(XW_crop,1),K,size(XW_crop,2));
% XWF_AN(:,:,f) = repmat(XW_crop(:,f),[1,K]).*H_AN';
% XWF_AN_recon = real(ifft([XWF_AN;conj(XWF_AN(end-1:-1:2,:,:))]));
% xwf_an = zeros(size(XWF_AN_recon,3)*S+1.75*N,K);
% xwf_an((f-1)*S+1:(f-1)*S+N,:) = xwf_an((f-1)*S+1:(f-1)*S+N,:)+XWF_AN_recon(:,:,f).*wo;
% xw_an_sum = sum(xwf_an,2);

%List from paper:
% 1. Oversample: (Resample x[n] from Fs to Fo
% 2. Filterbank: Pass x[n] through filter bank
% 3. IA, IP: Express x_k[n] = a_k[n]*cos(phi_k[n]).  Estimate a_k[n] and
% phi_k[n] (unwrapped).
% 4. Time-scaling:
%   Assign
%   Evaluate
%   Express
% 5. Synthesis: Sum all bands.  Resample from Fo to Fs

%Test code moved out to simplify code flow.
%Plot the Filter banks
% figure
% plot(H_AN')
% title('Aaron Filterbanks')
% figure
% plot(H_JL')
% title('James Filterbanks')

%Plot the filterbanks for frame 30
% figure
% plot(20*log(abs(XWF_AN(:,:,30))))
% title('AN filterbank frame')
% figure
% plot(20*log(abs(XWF_JL(:,:,30))))
% title('JL filterbank frame')

%Stop the overlap adding if the signal goes beyond the bounds of the
%allocated space.
%     if((f-1)*S+N>length(xwf_an))
%         disp('STOP')
%     end

%Plot all of the narrow band signals
% figure
% plot(xwf_an+repmat(1:K,[size(xwf_an,1),1]));
% title('AN narrowband signals')
% figure
% plot(xwf_jl+repmat(1:K,[size(xwf_jl,1),1]));
% title('JL narrowband signals')

%Play each of the filterbank signals
% for k=1:K
%     soundsc(xwf_an(:,k),fs);
%     pause((length(xwf_an)/fs)*1.1)
%     soundsc(xwf_jl(:,k),fs);
%     pause((length(xwf_jl)/fs)*1.1)
% end

%Try wrapping instantaneous phase
%Tried, and wrapping makes no difference.
% k = round(phik_hat_i/(2*pi)) ;
% phik_hat_i_adjust = phik_hat_i-k*2*pi;


%Playback the resulting signals
% soundsc(xw_an_sum,fs);
% pause((length(xw_an_sum)/fs)*1.1)
% soundsc(xw_jl_sum,fs);
% pause((length(xw_jl_sum)/fs)*1.1)