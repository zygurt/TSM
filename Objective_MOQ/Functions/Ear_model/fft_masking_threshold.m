function [ M ] = fft_masking_threshold( E, version )
%UNTITLED4 Summary of this function goes here
%   Masking Threshold
%   ITU-R BS.1387-1 Section 2.1.9
global debug_var

if debug_var
    disp('  Masking Threshold')
end
if (strcmp(version, 'basic') || strcmp(version, 'Basic'))
    res = 0.25; %bark
    Z = 109;
elseif (strcmp(version, 'advanced') || strcmp(version, 'Advanced'))
    res = 0.5;  %bark
    Z = 55;
else
    disp('Unknown version')
    M = 0;
    return
end
m = zeros(1,Z);
for k = 1:Z
    if k*res <= 12
        m(k) = 3;
    else
        m(k) = 0.25*k*res;
    end
end

M = E./(10.^(repmat(m,size(E,1),1)/10));

end

