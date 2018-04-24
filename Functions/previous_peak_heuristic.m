function [ prev_peak ] = previous_peak_heuristic( current_peak, prev_peaks, prev_rl, prev_ru)
%Find the previous peak that relates to the current one
%   Current peak is scalar
%   prev_peaks is vector
%   prev_rl is vector (region upper)
%   prev_ru is vector (region upper)

each_side = [0 1 2 4 8 16]; %Bins each side to check
transition = [16 32 64 128 256 512];

if isempty(prev_peaks) == 0
    index = 1;
    while (prev_rl(index) <= current_peak && prev_ru(index) >= current_peak) == 0
        index = index+1;
    end
    
    if index<transition(1)
        k = 1;
    elseif index<transition(2)
        k = 2;
    elseif index<transition(3)
        k = 3;
    elseif index<transition(4)
        k = 4;
    elseif index<transition(5)
        k = 5;
    else
        k = 6;
    end
    
    if (abs(current_peak-prev_peaks(index)) <= each_side(k))
        prev_peak = prev_peaks(index);
    else
        prev_peak = 0;
    end
else
    prev_peak = 0;
end
end