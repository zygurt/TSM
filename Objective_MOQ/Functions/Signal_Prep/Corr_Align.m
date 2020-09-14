function [ x,y ] = Corr_Align( x,y )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
%Find the best lead or lag for alignment of the signals
k_arr = zeros(round(0.1*size(x,2)),1);
k_arr2 = zeros(round(0.1*size(x,2)),1);
for k = 1:round(0.1*size(x,2))
    k_arr(k) = sum(x(1:end-k+1).*y(k:end));
    k_arr2(k) = sum(x(k:end).*y(1:end-k+1));
end
%This priorities the maximum for forwards and backwards
%Find overall max in each correlation, then find the location
[max_correlation, loc] = max([k_arr; k_arr2]);
if loc < length(k_arr)
    km = min(find(k_arr==max_correlation))-1;
else
    km = -1*(min(find(k_arr2==max_correlation))-1);
end
if(isempty(km))
    km = 0;
end


%Adjust for the correlation lead/lag by truncating beginning of late signal
% and end of early signal
if km<0
    y = y(1:end-abs(km));
    x = x((abs(km)+1):end);

elseif km>0
    y = y((abs(km)+1):end);
    x = x(1:end-abs(km));

end

end

