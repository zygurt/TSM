function [ F_noise, P_noise ] = error_sig( Fe_Ref, Fe_Test, fs, bands, model )
%[ P_noise ] = error_sig( Fe_Ref, Fe_Test, fs, bands, model )
%   Implemented as per ITU-R BS.1387-1 Section 3.4
%   Only calculated for the FFT_based Model
global debug_var

if debug_var
disp('  Error Signal Calculation')
end
if (strcmp(model, 'fft') || strcmp(model, 'FFT'))

    F_noise = abs(abs(Fe_Ref)-abs(Fe_Test));
    
    P_noise = fft_pitch_mapping(F_noise,fs,bands);
    
else
    disp('Error signal not needed for fb method')
    P_noise = 0;
    return
end






end

