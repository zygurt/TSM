function [ fac ] = fb_scaling_factor_calculation( x )
%[ fac ] = fb_scaling_factor_calculation( x )
%   Implemented as per ITU-R BS.1387-1 Section 2.2.3
global debug_var

if debug_var
    disp('  Filter Bank Scaling Factor Calculation');
end
if(nargin == 0)
    Lp = 92;
else
    Lp = 20*log10(rms(x)/(20*10^-6));
end
fac = 10.^(Lp/20);

end

