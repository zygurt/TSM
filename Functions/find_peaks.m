function [ peaks ] = find_peaks( mag_X )
% [ peaks ] = find_peaks( mag_X )
% Find the peaks of an array.  Takes the magnitude spectrum as input
%   The function finds the peaks of an array and returns the peak location
%   An element is considered a peak if it is greater than its 4 nearest
%   neighbours 
%   As per Laroche and Dolson 'Improved Phase Vocoder Time-Scale Modification of Audio' 1999
%   This function also returns the lower and upper bounds of the regions of
%   each peak
%   This function expects column magnitude data and returns peaks for each
%   column
% peaks(c).pa  => Peak Array
% peaks(c).rl  => Region Lower Bound
% peaks(c).ru  => Region Upper Bound
% peaks(c).empty_flag  => 1 = empty, 0 = data

% Tim Roberts - Griffith University 2018

num_chan = size(mag_X,2);
%Initialise
zero_pad_X = [zeros(2,num_chan); mag_X; zeros(2,num_chan)];
for c = 1:num_chan
    peaks(c).pa = [];
    peaks(c).empty_flag = 1;
end
%Find the peaks
for c = 1:num_chan
    for n=3:length(zero_pad_X)-2
        if zero_pad_X(n,c)>zero_pad_X(n-2,c) && zero_pad_X(n,c)>zero_pad_X(n-1,c) && zero_pad_X(n,c)>zero_pad_X(n+1,c) && zero_pad_X(n,c)>zero_pad_X(n+2,c)
            peaks(c).pa = [peaks(c).pa ; (n-2)];
            peaks(c).empty_flag = 0;
        end
    end
end
%Set the region bounds as half way between peaks
for c = 1:num_chan
    if (peaks(c).empty_flag)
        %There are no peaks
        peaks(c).rl = [];
        peaks(c).ru = [];
    else
        %Take 0 padding into account when calculating the upper and lower
        %regions.
        peaks(c).rl = zeros(length(peaks(c).pa),1);
        peaks(c).ru = zeros(length(peaks(c).pa),1);
        peaks(c).rl(1) = 1;
        peaks(c).rl(2:end) = ceil((peaks(c).pa(2:end)+peaks(c).pa(1:end-1))/2);
        peaks(c).ru(1:end-1) = peaks(c).rl(2:end) - 1;
        peaks(c).ru(length(peaks(c).ru)) = length(mag_X);
    end
end
end