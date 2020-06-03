function [ fac ] = fft_scaling_factor_calculation( x )
%[ fac ] = scaling_factor_calculation( x )
%   Calculates the scaling factor described in
%   ITU-R BS.1387-1 Section 2.1.3
%   Equation 5
global debug_var

if debug_var
    disp('  Scaling Factor Calculation');
end
if(nargin == 0)
    Lp = 92;
else
    Lp = 20*log10(rms(x)/(20*10^-6));
end
N = 2048;
f = 1019.5; %Hz
fs = 44100;
N_frames = 10;
norm_sig = sin(2*pi*f*(1:fs*N_frames)/fs);
Norm = max(abs(fft(norm_sig(1:N))));
fac = (10.^(Lp/20))/Norm;

% MAG = abs(fft(sig(1:N).*w));
% [maximum, loc] = max(MAG(1:N/2+1));
% bin_width = (fs/N);
% peak_loc_freq = loc*bin_width;
% fprintf('Peak location = %g Hz\n',peak_loc_freq);
% plot(MAG(1:N/2));

end

