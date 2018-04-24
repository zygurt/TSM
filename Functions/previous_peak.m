function [ prev_peak ] = previous_peak( current_peak, prev_peaks, prev_rl, prev_ru)
%Find the previous peak that relates to the current one
%   Current peak is scalar
%   prev_peaks is vector
%   prev_rl is vector (region upper)
%   prev_ru is vector (region upper)

if isempty(prev_peaks) == 0
    index = 1;
    while (prev_rl(index) <= current_peak && prev_ru(index) >= current_peak) == 0
        index = index+1;
    end
    prev_peak = prev_peaks(index);
else
    prev_peak = 0;
end
end