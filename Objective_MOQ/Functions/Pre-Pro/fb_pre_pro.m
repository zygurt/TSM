function [ Ref, Test ] = fb_pre_pro( Ref_Model, Test_Model, General )
%UNTITLED Summary of this function goes here
%   Implementation of the PEAQ Pre-processing of excitation patterns
%   Implemented as per ITU-R BS.1387-1 Section 3
global debug_var

if debug_var
disp('Filterbank Pre-processing Excitation Patterns')
end
StepSize = General.AdvancedStepSize;

const = fb_constants();
fc = const.fc;
%Z is defined above
%All variables and recursive filters init to 0.

%Level and Pattern adaptation.

[Ref.E_L, Test.E_L] = level_adapt( Ref_Model.E, Test_Model.E, fc, StepSize, General.fs );
[Ref.E_P, Test.E_P] = pattern_adapt( Ref.E_L, Test.E_L, fc, StepSize, General.fs, 'advanced', General.model);


[Ref.Mod, Ref.Eline] = modulation(Ref_Model.E2, fc, General.fs, StepSize);
[Test.Mod, Test.Eline] = modulation(Test_Model.E2, fc, General.fs, StepSize);


[Ref.N, Ref.N_total] = loudness( Ref_Model.E, fc, General.version, General.model );
[Test.N, Test.N_total, Test.E_Thresh] = loudness( Test_Model.E, fc, General.version, General.model );


% [ Test.F_noise, Test.P_noise] = error_sig( Ref_Model.Fe, Test_Model.Fe, General.fs, Ref_Model.bands, General.model );

end

