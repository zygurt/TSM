function [ MOV ] = ADBB(Pc, Qc)
%[ MOV ] = ADBB(Pc, Qc)
%   As described by ITU-R BS.1387-1 Section 4.7.2
global debug_var

if debug_var
disp('    Average Distorted Block(Frame)')
end

n_distorted = sum(Pc>0.5);

Qsum = sum(Qc);

if n_distorted == 0
    MOV = 0;
elseif n_distorted>0 && Qsum > 0
    MOV = log10(Qsum/n_distorted);
elseif n_distorted>0 && Qsum ==0
    MOV = -.05;
else
    disp('No ADBB calculation available');
    MOV = 0;
end



end

