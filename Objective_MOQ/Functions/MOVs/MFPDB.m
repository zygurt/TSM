function [ MOV ] = MFPDB(Pc, General)
%[ MOV ] = MFPDB(Pc, General)
%   As described by ITU-R BS.1387-1 Section 4.7.1
global debug_var

if debug_var
disp('    Maximum Filtered Probability of Detection')
end
c0 = 0.9^(General.BasicStepSize/1024);

Pc_tilda = zeros(size(Pc));
for n = 1:length(Pc)
    if n==1
        %Initial
        Pc_tilda(n) = (1-c0)*Pc(n);
    else
        Pc_tilda(n) = (1-c0)*Pc(n)+c0*Pc_tilda(n-1);
    end

end
PMc = zeros(size(Pc));
c1 = 0.99^(General.BasicStepSize/1024);
for n = 1:length(Pc)
    if n==1
        %Initial
        PMc(n) = max(0,Pc_tilda(n));
    else
        PMc(n) = max(c1*PMc(n-1),Pc_tilda(n));
    end

end

MOV = PMc(end);

    
end

