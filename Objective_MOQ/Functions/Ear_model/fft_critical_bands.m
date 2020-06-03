function [ bands ] = fft_critical_bands( version )
%[ bands ] = critical_bands( version )
%   version is either 'basic' or 'advanced'
%   ITU-R BS.1387-1 Section 2.1.5
%   Can be used to generate the frequency tables shown in Table 6 and 7
global debug_var

if debug_var
    disp('  Critical Bands')
end
f_low = 80;
f_high = 18000;

if (strcmp(version, 'basic') || strcmp(version, 'Basic'))
    res = 0.25; %bark
elseif (strcmp(version, 'advanced') || strcmp(version, 'Advanced'))
    res = 0.5;  %bark
else
    disp('Unknown version')
    bands = 0;
    return
end
bark = hz2bark(f_low):res:hz2bark(f_high);
hz = bark2hz(bark);
fl = hz;
fu = [fl(2:end) f_high];
fc = (fl+fu)/2;
bands = [fl;fc;fu];

end

