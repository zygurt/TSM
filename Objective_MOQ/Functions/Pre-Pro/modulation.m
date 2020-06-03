function [ Mod, Eline ] = modulation(E2, fc, fs, StepSize)
%[ Mod, Eline ] = modulation(E2, fc, fs, StepSize)
%   ITU-R BS.1387-1 Section 3.2
global debug_var

if debug_var
disp('  Modulation')
end
%Low pass the input signals
tau_100 = 0.05;
tau_0 = 0.008;
%Equation 41
tau = tau_0 + (100./fc) * (tau_100-tau_0);
%Equation 44
a = exp(-StepSize./(fs*tau));
E_der = zeros(size(E2));
Eline = zeros(size(E2));
for n = 1:size(E2,1)
    if n==1
        %Initial frame
        E_der(n,:) = (1-a).*(fs/StepSize).*abs(E2(n,:).^0.3);
        Eline(n,:) = (1-a).*E2(n,:).^0.3;
    else
        E_der(n,:) = a.*E_der(n-1,:) + (1-a).*(fs/StepSize).*abs(E2(n,:).^0.3-E2(n-1,:).^0.3);
        Eline(n,:) = a.*Eline(n-1,:) + (1-a).*E2(n,:).^0.3;
    end
end

Mod = E_der./(1+Eline/0.3);

end

