function [ y, ZFR_Length ] = ZFR( s, N_CALC, fs, N_scale )
%[ y ] = ZFR( s, N )
%   Zero Frequency Resonator
%
%   Proposed by Murt and Yegnanarayana, Epoch Extraction from Speech Signals, 2008
%   Implementation based on Rudresh et al., Epoch-Synchronous Overlap-Add
%   (ESOLA) for Time- and Pitch-Scale Modification of Speech Signals, 2018
%
%   s is the speech signal
%   N_CALC is used to switch between N values used for mean-subtraction operation
%       0: Speech 44100
%       1: Calculates fundamental frequency and adjusts accordingly
%       Otherwise: Baseline fs/220
%   fs is the sampling frequency of the signal
%   N_scale is the used to determine where N lies between 1 to 2 times the
%   average pitch period of the speaker. 1 < N_scale < 2.

%   Positive zero crossings of y indicate epochs

% Tim Roberts - Griffith University 2018

%Pre-process speech
x = s(:)-[0;s(1:end-1)];

%Pass signal through ideal zero-frequency resonator (integrator) twice.
%   _________________G________________
%   (1-re^(jw_0)z^-1)(1-re^(jw_0)z^-1)
%   Becomes for 0 frequency
%   G = 1
%   a1 = -2
%   a2 = 2
% Thanks to Novak3 who pointed out that the filter coefficients should be
% a = [1, -2, 1]
% This was an error on my part treating a2 and a3 as a1 and a2 as well as a
% mathematical error as I turned a page.
% For the true 2 pole ideal zero-frequency resonator, use a=[1,-2,1],
% however the single pole resonator a=[-2 2] works as well.
a = [-2 2];
y1 = filter(1,a,x);
y2 = filter(1,a,y1);

switch N_CALC
    case 0
        N = 217; %For fs=44100 this works passably for male and female speech
    case 1
        ms = 40;
        n = 2^(nextpow2(ms*(10^-3)*fs));
        s_buf = buffer(x,n,3*n/4);
        TEMP_max_loc = zeros(size(s_buf,2),1);
        for m = 1:size(s_buf,2)
            temp = xcorr(s_buf(:,m),s_buf(:,m));
            TEMP = fft(temp(n+1:end));
            [~,TEMP_max_loc(m)] = max(abs(TEMP));
        end
        mode_TEMP_max_loc = mode(TEMP_max_loc);
        voiced_frames = find(TEMP_max_loc<mode_TEMP_max_loc);
        % S = fft(s_buf(:,voiced_frames(ceil(length(voiced_frames)/2)))); %Just choosing the middle voiced frame
        S2 = fft(s_buf(:,voiced_frames(:)));
        S2 = sum(abs(S2),2);
        % S = abs(S);
        [~,max_S_loc] = max(S2(2:n/2+1));  %ignore DC component
        bin_width = fs/n;
        To = 1/((max_S_loc+1)*bin_width);   %+1 to make up for ignored DC component
        %Calculate N for mean-subtraction process
        N = round((fs*(N_scale*To-1/fs))/2);
    otherwise
        N = fs/220;
end

ZFR_Length = N;
% fprintf('ZFR N value = %d\n',N);
% figure(1)
% plot(s)
% hold on
% plot(linspace(1,length(s),length(TEMP_max_loc)),TEMP_max_loc/max(TEMP_max_loc))
% hold off
% figure(2)
% plot(S2(1:length(S2)/2))

%Remove trend in y2 by successively applying a mean-subtraction operation.
%(2N+1) is chosen to lie between 1 to 2 times the average pitch period of the speaker
y3 = zeros(length(s),size(s,2));
for n = 3:length(y2)
    temp = 0;
    count = 0;
    for m = -N:N
        %Track when index is outside size of y2
        if(n+m)>0 && (n+m)<length(y2)
            temp = temp+y2(n+m);
            count = count+1;
        end
    end
    y3(n-2) = y2(n)-(temp/count);
end

%Positive zero crossing indicate epochs
% length(s)-1 to avoid exceeding bounds of y3
y = zeros(length(s),size(s,2));
for n = 1:length(s)-1
    if (y3(n)*y3(n+1)<0) && (y3(n)<y3(n+1))
        y(n) = 1;
        %Else it is already zero
    end
end

end

