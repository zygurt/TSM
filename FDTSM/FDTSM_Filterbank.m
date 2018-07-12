function [ y ] = FDTSM_Filterbank( x, N, region_info )
%[ y ] = FDTSM_Filterbank( x, N, region_info )
%   Frequency Dependent Time-Scale Modification using mel-spaced
%   filterbanks
%   x is mono input signal
%   N is the window size
%   region_info should contain
%       region_info.TSM - Vector of TSM ratios
%       region_info.upper - Upper bounds of each region
                      % - max(region_info.upper) = N/2
%       region_info.centre - centre of each region

addpath('../Functions');
addpath('../Frequency_Domain');
%Initial variables
K = length(region_info.TSM);   %32 for fs=16kHz
S = N/4;
%% --------------------------Analysis------------------------------
%This section splits the input signal into K band passed signals
disp('Analysis');
%Oversample
%Create window (Hann)
w = 0.5*(1 - cos(2*pi*(0:N-1)'/(N-1)));
%Frame the input
xw = buffer(x, N, N-S);
%Window the frames
xw = xw.*repmat(w,1,size(xw,2));
%Convert to frequency domain
XW = fft(xw,N);
%Generate Filterbanks
disp('    Generate Filterbanks');
c = linspace(1,N/2+1,length(region_info.TSM)+2);
H = filterbank(c(2:end-1),N);
%Take the first half of the fft
XW_crop = XW(1:N/2+1,:);
%Prepare framed filterbank output
XWF = zeros(size(XW_crop,1),K,size(XW_crop,2));
%Mulitply through with the filterbanks.
disp('    Multiply filterbanks through signal');
for k = 1:K
    k
    for f = 1:size(XW,2)
        XWF(:,:,f) = repmat(XW_crop(:,f),[1,K]).*H';
    end
end
%Reconstruct second half of the signal
XWF_recon = real(ifft([XWF;conj(XWF(end-1:-1:2,:,:))]));
%Prepare filterbank channels
xwf = zeros(size(XWF_recon,3)*S+1.75*N,K); %Need to make this longer.  Janky solution for now.
%Create the output window
wo = repmat(w,1,size(XWF_recon,2));
%Overlap add the signal back together
disp('    Overlap Add the signal back together');
for f = 1:size(XWF,3)
    xwf((f-1)*S+1:(f-1)*S+N,:) = xwf((f-1)*S+1:(f-1)*S+N,:)+XWF_recon(:,:,f).*wo;
end
%At this point, xwf_jl is a K channel signal version of the original x
%input signal

%% --------------------------Modification--------------------------
disp('Modification')
max_length = 0;
for k = 1:K
    y_split(k).result = PV(xwf(:,k), N, region_info.TSM(k));
    if length(y_split(k).result)>max_length
        max_length = length(y_split(k).result);
    end
end

%% --------------------------Synthesis------------------------------
disp('Synthesis')
%Combine the filterbank audio signals
y = zeros(max_length,1);
for k=1:K
    y(1:length(y_split(k).result)) = y(1:length(y_split(k).result))+y_split(k).result;
end
%Normalise the output
y=y/max(abs(y));
disp('File Processing Complete');

end

