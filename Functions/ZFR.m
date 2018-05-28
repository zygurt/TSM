function [ y ] = ZFR( s, fs )
%[ y ] = ZFR( s, N )
%   Zero Frequency Resonator
%
%   Proposed by Murt and Yegnanarayana, Epoch Extraction from Speech Signals, 2008
%   Implementation based on Rudresh et al., Epoch-Synchronous Overlap-Add
%   (ESOLA) for Time- and Pitch-Scale Modification of Speech Signals, 2018
%
%   s is the speech signal
%   N is the frame length
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

%Remove trend in y2 by successively applying a mean-subtraction operation.
%(2N+1) is chosen to lie between 1 to 2 times the average pitch period of the speaker
%This could be improved by knowing the average pitch period of the speaker

%Calculating N
if(fs==44100)
    N = 217;
else
    f0 = [85, 165, 180, 255];   %Male and female ranges of fundamental frequency
    a = linspace(1,2,length(f0))'; %Acceptable range.
    N0 = (a.*(fs./f0)+1)/2; %Matrix of possible values of N.
    N = round(mean([N0(1,1) N0(4,4)])); %Split the difference between lowest male and highest female
end


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

