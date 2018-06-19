function [ y ] = ZFR( s, fs, N_scale )
%[ y ] = ZFR( s, N )
%   Zero Frequency Resonator
%
%   Proposed by Murt and Yegnanarayana, Epoch Extraction from Speech Signals, 2008
%   Implementation based on Rudresh et al., Epoch-Synchronous Overlap-Add
%   (ESOLA) for Time- and Pitch-Scale Modification of Speech Signals, 2018
%
%   s is the speech signal
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
a = [-2 2];
y1 = filter(1,a,x);
y2 = filter(1,a,y1);

% Additional Step
%Find the fundamental pitch period.  Made this up.  Maybe I should find a paper.
%Pitch period is generally determined using autocorrelation.
%Because I just want an estimate, and the fundamental is usually loudest,
%I'm just averaging the entire signal and finding the maximum location.
ms = 40;
n = 2^(nextpow2(ms*(10^-3)*fs));
s_buf = buffer(x,n,3*n/4);
S = fft(s_buf);
S_avg = mean(abs(S),2);
[~,max_loc] = max(S_avg(2:n/2+1));  %ignore DC component
bin_width = fs/n;
To = 1/(max_loc*bin_width);
%Calculate N for mean-subtraction process
N = round((fs*(N_scale*To-1/fs))/2);
fprintf('ZFR N value = %d\n',N);

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

