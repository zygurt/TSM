function [ y ] = SOLA_DAFX( x, N, TSM )
%[ y ] = SOLA_DAFX( x, N, TSM )
%   Adaptation of SOLA from DAFx 2002 p. 209-211
%TSM  0.5 < TSM < 4
DAFx_in = x';
Sa = N/4;
M = ceil(length(DAFx_in)/Sa); %Number of segments

alpha = 1/TSM;
Ss = round(Sa*alpha);
L = round(Sa*alpha/2)+mod(round(Sa*alpha/2),2); %Overlap (Adjusted to be even)
%This only overlaps half of the full overlap

DAFx_in(M*Sa+N)=0;
Overlap = DAFx_in(1:N);

%SOLA algorithm

for ni = 1:M-1
   grain = DAFx_in(ni*Sa+1:N+ni*Sa);
   XCORRsegment = xcorr(grain(1:L),Overlap(1,ni*Ss:ni*Ss+(L-1)));
   [~,index(1,ni)]=max(XCORRsegment);
   fadeout = 1:(-1/(length(Overlap)-(ni*Ss-(L-1)+index(1,ni)-1))):0;
   fadein = 0:(1/(length(Overlap)-(ni*Ss-(L-1)+index(1,ni)-1))):1;
   Tail = Overlap(1,(ni*Ss-(L-1))+ ...
                    index(1,ni)-1:length(Overlap)).*fadeout;
   Begin = grain(1:length(fadein)).*fadein;
   Add = Tail+Begin;
   Overlap = [Overlap(1,1:ni*Ss-L+index(1,ni)-1) ...
              Add ...
              grain(length(fadein)+1:N)];
end
y = Overlap;
end

