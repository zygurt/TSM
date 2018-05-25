function [ y ] = WSOLA_Driedger( x, N, TSM )
%WSOLA
%This algorithm works by choosing the next segment based on similarity to
%the next natural segment.
%Adaptation from Dreidger's MATLAB TSM Toolbox 1.0
% [DM14] Jonathan Driedger, Meinard Mueller
%        TSM Toolbox: MATLAB Implementations of Time-Scale Modification
%        Algorithms
%        Proceedings of the 17th International Conference on Digital Audio
%        Effects, Erlangen, Germany, 2014.
%Tricks of the trade (Dreidger 2016)
%N is normally set to be about 50ms


%User Parameters
alpha = 1/TSM;
wn = 0.5*(1 - cos(2*pi*(0:N-1)'/(N-1))); %hanning window
Ss = N/2;               %Synthesis hop
tol = N/2;              %Tolerance
%Setup
Output_length = ceil(alpha*length(x));
sPosWin = 1:Ss:Output_length+N/2;     %array of synthesis positions
aPosWin = round(interp1([1 Output_length],[1 size(x,1)],sPosWin,'linear','extrap'));
Sa = [0 aPosWin(2:end)-aPosWin(1:end-1)];
%WSOLA
y = zeros(Output_length, 1);
minFac = min(Ss./Sa);
xC = [zeros(N/2 + tol,1) ; x; zeros(ceil(1/minFac)*N+tol,1)];
aPosWin = aPosWin + tol;
yC = zeros(Output_length + 2*N, 1);
ow = zeros(Output_length + 2*N, 1);
delta = 0;
for n = 1:length(aPosWin)-1
    curr_syn_win_range = sPosWin(n):sPosWin(n) + N-1;
    curr_ana_win_range = aPosWin(n)+delta:aPosWin(n)+delta+N-1;
    %OLA
    yC(curr_syn_win_range) = yC(curr_syn_win_range)+xC(curr_ana_win_range).*wn;
    ow(curr_syn_win_range) = ow(curr_syn_win_range)+wn;
    natProg = xC(curr_ana_win_range + Ss);
    next_ana_win_range = aPosWin(n+1)-tol:aPosWin(n+1)+tol+N-1;
    next_ana_window = xC(next_ana_win_range);
    %Cross Correlation between next_ana_window and natProg
    xcf = conv(next_ana_window(length(next_ana_window):-1:1),natProg);
    xcf = xcf(N:end - N + 1);
    %find position of max xcorr
    [~,maxlag] = max(xcf);
    %maxlag = find(xcf==max(xcf));
    delta = tol-maxlag+1;
end
%Process the last frame
yC(sPosWin(end):sPosWin(end)+N-1) = yC(sPosWin(end):sPosWin(end)+N-1) + xC(aPosWin(n)+delta:aPosWin(n)+delta+N-1).*wn;
ow(sPosWin(end):sPosWin(end)+N-1) = ow(sPosWin(end):sPosWin(end)+N-1) + wn;
%Normalise the output
ow(ow<10^-3) = 1; % avoid potential division by zero
yC = yC./ow;
%Remove zero padding
yC = yC(N/2+1:end);
yC = yC(1:length(y));
y = yC;
y = y(N+1:end)/max(abs(y));

end
