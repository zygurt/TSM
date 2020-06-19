function DM = DM_Calc_OLD( Ref_Model, Test_Model)
%DM = DM_Calc( Ref_Model, Test_Model, General)
%   Measure of consistancy proposed by Laroche and Dolson
%   in "Improved Phase Vocoder Time-Scale Modification of Audio"
global debug_var

if debug_var
    disp('DM Calculation')
end

Z = Test_Model.X_MAG;
Y = Ref_Model.X_MAG;

DM = sum(sum(Z-Y).^2)/sum(sum(Y.^2));

end