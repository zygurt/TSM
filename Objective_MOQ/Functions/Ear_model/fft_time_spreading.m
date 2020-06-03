function [ E ] = fft_time_spreading( E2, fc )
%[ E ] = time_spreading( E2, E_f_prev, fc )
%   Time domain spreading
%   ITU-R BS.1387-1 Section 2.1.8
global debug_var

if debug_var
    disp('  Time Spreading')
end

tau_min = 0.008;
tau_100 = 0.03;
%Calculate time constants (Equation 21)
tau = tau_min + (100./fc) * (tau_100-tau_min);

%Compute a (Equation 24)
a = exp(-4./(187.5*tau));
%Init Variables
Ef = zeros(size(E2));
E = zeros(size(E2));

for n = 1:size(E2,1)
    if n == 1
        Ef_prev = zeros(1,size(E2,2));
    end
    %First order LPF (Equation 22)
    Ef(n,:) = a.*Ef_prev+(1-a).*E2(n,:);
    %Equation 23
    E(n,:) = max(Ef(n,:),E2(n,:));
    Ef_prev = E(n,:);
end

end

