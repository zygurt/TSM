function [ Ref_Model, Test_Model ] = fb_Ear_Model(sig_ref, sig_test, General, match_method)
%UNTITLED2 Summary of this function goes here
%   Implementation of the PEAQ Filterbank Ear Model
%   Implemented as per ITU-R BS.1387-1 Section 2.2
global debug_var

if debug_var
    disp('FB Ear Modelling')
end
%Scale the input signals
fac_ref = fb_scaling_factor_calculation( sig_ref );
fac_test = fb_scaling_factor_calculation( sig_test );
sig_ref_scaled = fac_ref*sig_ref;
sig_test_scaled = fac_test*sig_test;

%DC rejection filter
sig_ref_dc_reject = fb_DC_reject( sig_ref_scaled );
sig_test_dc_reject = fb_DC_reject( sig_test_scaled );

%Decomposition into auditory filter bands
[ x_bank_ref ] = fb_filter_bank( sig_ref_dc_reject, General );
[ x_bank_test ] = fb_filter_bank( sig_test_dc_reject, General );

%Outer and Middle Ear Weighting
[ x_bank_ref_ear ] = fb_outer_middle_ear( x_bank_ref );
[ x_bank_test_ear ] = fb_outer_middle_ear( x_bank_test );

%Frequency domain spreading and Rectification
[ E0_ref ] = fb_freq_spreading( x_bank_ref_ear, General );
[ E0_test ] = fb_freq_spreading( x_bank_test_ear, General );

%Time Domain Spreading
[ Ref_Model, Test_Model ] = fb_tds( E0_ref, E0_test, General, match_method ); %Frequency swapped to 2nd dimension to match fft calculation

end

