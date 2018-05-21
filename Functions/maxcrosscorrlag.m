function [lag_x, lag_y] = maxcrosscorrlag(x, y, low_lim, high_lim)
% [k, lag] = maxcrosscorrlag(x, y, low_lim, high_lim)
% Compute the location and lag of the maximum cross correlation between 2 vectors
% x    |=============| -->
% y                |=============|
% xc_a |=========================|
%
% lag_x is the number of samples that x needs to move for maximum cross correlation
% lag_y is the number of samples that y needs to move for maximum cross correlation
%   positive values mean later in time, negative values mean earlier in time
% xc_a is the cross correlation array
%
% low_lim and high_lim are used to remove large cross correlation values at
%   the extreme ranges of the cross correlation.  Values in these ranges will
%   not be calculated, and left as 0.
%   Default values of 1. (Compute entire range)

% Tim Roberts - Griffith University 2018

%Set default low and high limits
if(nargin == 2)
    low_lim = 1;
    high_lim = 1;
end

%Orient in the same direction
if(size(x,1)<size(x,2))
    x = x';
end
if(size(y,1)<size(y,2))
    y = y';
end

%Match the lengths
x_length = length(x);
y_length = length(y);
if x_length ~= y_length
    if x_length > y_length
        y = [y ; zeros(x_length - y_length, 1)];
    else
        x = [x ; zeros(y_length - x_length, 1)];
    end
end

%Initialise the sub xcorr arrays
xc_ax = zeros(length(x), 1);
xc_ay = zeros(length(y), 1);

%Calculate cross correlations
for n = low_lim:length(x)-1
    xc_ax(n) = crosscorr_t(x(end-n+1:end),y(1:n));
end
for n = high_lim:length(y)
    xc_ay(n) = crosscorr_t(y(end-n+1:end),x(1:n));
end

%Combine cross correlation arrays
xc_a = [xc_ax(1:end-1) ; flipud(xc_ay)];

%Compute lag arrays
x_lag_y = (1:length(xc_a))-length(x);
y_lag_x = length(x) - (1:length(xc_a));
[maximum, max_loc] = max(xc_a);

%Check for entire zero array. (Cross correlation of zeros)
if(maximum == 0)
    lag_x = 0;
    lag_y = 0;
else
    lag_x = x_lag_y(max_loc);
    lag_y = y_lag_x(max_loc);
end

end