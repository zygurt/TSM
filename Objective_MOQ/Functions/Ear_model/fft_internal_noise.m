function [ Pp ] = fft_internal_noise( Pe, fc )
%[ Pp ] = internal_noise( Pe, fc )
%   Adds internal noise to the Pitch Energies as per
%   ITU-R BS.1387-1 Section 2.1.6
%   Equations 13-14
global debug_var

if debug_var
    disp('  Internal Noise')
end
P_thresh = 10.^(0.4*0.364*(fc/1000).^-0.8);
Pp = Pe+repmat(P_thresh,size(Pe,1),1);

end

