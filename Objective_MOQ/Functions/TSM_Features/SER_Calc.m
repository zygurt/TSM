function SER = SER_Calc( Ref_Model, Test_Model)
%SER = SER_Calc( Ref_Model, Test_Model)
%   Measure of consistancy proposed by Griffin and Lim
%   in LSEE_MSTFTM paper, quoted by Roucos an Wilgus in
%   "High Quality Time-Scale Modification for Speech"
global debug_var

if debug_var
    disp('SER Calculation')
end

X = Ref_Model.X_MAG;
Y = Test_Model.X_MAG;

SER = 10*log10(sum(sum(Y.^2))/sum(sum(X-Y).^2));

%SER can go to infinity if files are identical
% Therefore, SER needs to be bounded
%Maximum value found experimentally was 80.5852
%For non-identical files, values above 20 are rare
if SER > 80
    SER=80;
end


end