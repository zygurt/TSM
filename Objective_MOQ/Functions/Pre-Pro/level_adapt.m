function [ E_L_ref, E_L_test ] = level_adapt( E_ref, E_test, fc, StepSize, fs )
%[ E_L_ref, E_L_test ] = level_adapt( E_ref, E_test, fc, StepSize, fs )
%   ITU-R BS.1387-1 Section 3.1 and 3.1.1
global debug_var

if debug_var
disp('  Level Adaptation')
end
%Low pass the input signals
tau_100 = 0.05;
tau_min = 0.008;
%Equation 41
tau = tau_min + (100./fc) *(tau_100-tau_min);
%Equation 44
a = exp(-StepSize./(fs*tau));
%Equations 42-43
P_ref = zeros(size(E_ref));
P_test = zeros(size(E_test));
for n = 1:size(E_ref,1)
    if n == 1
        P_ref(n,:) = (1-a).* E_ref(n,:); %Removed the multiply by zeros
        P_test(n,:) = (1-a).* E_test(n,:); %Removed the multiply by zeros
    else
        P_ref(n,:) = a.*P_ref(n-1,:)+(1-a).* E_ref(n,:);
        P_test(n,:) = a.*P_test(n-1,:)+(1-a).* E_test(n,:);
    end
end

LevCorr = zeros(size(E_ref,1),1);
E_L_ref = zeros(size(E_ref));
E_L_test = zeros(size(E_ref));
for n = 1:size(E_ref,1)
    %Level Adaptation (Section 3.1.1 Equations 45-47)
    %Momentary Correction factor
    %Equation 45
    LevCorr(n) = (sum(sqrt(P_test(n,:).*P_ref(n,:)))/sum(P_test(n,:)))^2;
    
    if LevCorr(n) > 1
        E_L_ref(n,:) = E_ref(n,:)/LevCorr(n);
        E_L_test(n,:) = E_test(n,:);
    else
        E_L_test(n,:) = E_test(n,:)*LevCorr(n);
        E_L_ref(n,:) = E_ref(n,:);
    end
    
end

end

