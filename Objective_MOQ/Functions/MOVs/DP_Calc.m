function [ MOV ] = DP_Calc( Model_Ref, Model_Test, General )
%[ MOV ] = DP_Calc( Model_Ref, Model_Test, General )
%   Calculation of Detection Probability Features
%   As described by ITU-R BS.1387-1 Section 4.7
global debug_var

if debug_var
disp('  Detection Probability')
end
E_ref_db = 10*log10(Model_Ref.E);
E_test_db = 10*log10(Model_Test.E);

%Per channel.  Mono implementation, so ignore channels.

%Asymmetric average excitation
L = 0.3*max(E_ref_db,E_test_db)+0.7*E_test_db;
%Effective Detection Step Size s
N = size(L,1);
Z = size(L,2);
s = zeros(N,Z);

for n=1:N
    for k = 1:Z
        if L(n,k)>0
            s(n,k) = 5.95072*(6.39468./L(n,k)).^1.71332 + ...
                     (9.01033*10^-11)*L(n,k).^4 + ...
                     (5.05622*10^-6)*L(n,k).^3 - ...
                     0.00102438*L(n,k).^2 + ...
                     0.0550197*L(n,k) - 0.198719;
        else
            s(n,k) = 1*10^30;
        end
    end
end

%Signed Error
e = E_ref_db-E_test_db;

%Steepness of slope b
b = zeros(N,Z);
for n=1:N
    for k = 1:Z
        if(E_ref_db(n,k)>E_test_db(n,k))
            b(n,k) = 4;
        else
            b(n,k) = 6;
        end
    end
end

%Scale factor
a = (10.^(log10(log10(2.0))./b))./s;

%Probability of detection, Per channel.  Add this in the future maybe?
pc = 1-10.^(-a.*e.^b);

%Steps above threshold
qc = abs(round(e))./s;

%Binaural detection probability
%Not yet required.
% pbin = max(pleft,pright);
pbin = pc;

%Number of steps above threshold of the binaural channel
% qbin = max(qleft,qright);
qbin = qc;

%Total Probability Detection
Pc = 1-prod((1-pc),2);

%Total Steps above threshold
Qc = sum(qc,2);


MOV.MFPDB = MFPDB(Pc, General);
MOV.ADBB = ADBB(Pc, Qc);

end

