function [ E_P_Ref, E_P_Test ] = pattern_adapt( E_L_Ref, E_L_Test, fc, StepSize, fs, version, model)
%[ E_P_Ref, E_P_Test ] = pattern_adapt( E_L_Ref, E_L_Test, fc, StepSize, fs, version, model)
%   ITU-R BS.1387-1 Section 3.1.2
global debug_var

if debug_var
disp('  Pattern Adaptation')
end
%Low pass the input signals
tau_100 = 0.05;
tau_min = 0.008;
%Equation 41
tau = tau_min + (100./fc) *(tau_100-tau_min);
%Equation 44
a = exp(-StepSize./(fs*tau));

%Pattern Adaptation
R_num = zeros(1,size(E_L_Ref,2));
R_den = zeros(1,size(E_L_Ref,2));
R = zeros(size(E_L_Ref));
R_Test = zeros(size(E_L_Ref));
R_Ref = zeros(size(E_L_Ref));
for n = 1:size(E_L_Ref,1) %For each of the frames
    for m = 0:n-1
        R_num = R_num + a.^m.*E_L_Test(n-m,:).*E_L_Ref(n-m,:);
        R_den = R_den + a.^m.*E_L_Ref(n-m,:).*E_L_Ref(n-m,:);
    end
    R(n,:) = R_num./R_den;

    for k = 1:size(E_L_Ref,2)
        normal = 1;
        if(R_den(k) == 0 && R_num(k) > 0)
            R_Test(n,k) = 0;
            R_Ref(n,k) = 1;
            normal = 0;
        end
        if (R_den(k) == 0 && R_num(k) == 0)
            if(k>1)
                R_Test(n,k) = R_Test(n,k-1);
                R_Ref(n,k) = R_Ref(n,k-1);
                normal = 0;
            else
                R_Test(n,k) = 1;
                R_Ref(n,k) = 1;
                normal = 0;
            end
        end
        if (R(n,k)>=1 && normal == 1)
            R_Test(n,k) = 1/R(n,k);
            R_Ref(n,k) = 1;
        else
            R_Test(n,k) = 1;
            R_Ref(n,k) = R(n,k);
        end
    end

    R_num = zeros(1,size(E_L_Ref,2));
    R_den = zeros(1,size(E_L_Ref,2));
end


%Average the correction factors (Equation 50-53)
%Calculate Pattern Correction factors (Equations 50-51)
PattCorr_Test = zeros(size(E_L_Ref));
PattCorr_Ref = zeros(size(E_L_Ref));


for n = 1:size(R_Test,1)
    for k = 1:size(R_Test,2)
        %Calculate M, M1 and M2
        if (strcmp(version, 'basic') || strcmp(version, 'Basic'))
            M = 8;
            Z = 109;
        elseif (strcmp(version, 'advanced') || strcmp(version, 'Advanced'))
            if (strcmp(model, 'filterbank') || strcmp(model, 'Filterbank'))
                M = 3;
                Z = 55;
            else
                M = 4;
                Z = 40;
            end
        else
            disp('Unknown version')
            M = 0;
            return
        end

        if(mod(M,2)==1) %Odd
            M1 = (M-1)/2;
            M2 = M1;
        else %Even
            M1 = M/2-1;
            M2 = M/2;
        end
        %Check to ensure within bounds
        if (k-M1<1 || k+M2 > size(R_Test,2)-1)
            M1 = min(M1,k);
            M2 = min(M2, Z-k-1);
            M = M1+M2+1;
        end
        %Calculate internal sum
        temp_test = 0;
        temp_ref = 0;
        for i = -M1:M2
%             disp(k+i+1)
%             if(k+i+1)==41
%                 disp('stop')
%             end
            temp_test = temp_test + R_Test(n,k+i+1);
            temp_ref = temp_ref + R_Ref(n,k+i+1);
        end

        if(n==1)
            PattCorr_Test(n,k) = ((1-a(k)).*temp_test)/M;
            PattCorr_Ref(n,k) = ((1-a(k)).*temp_ref)/M;
        else
            PattCorr_Test(n,k) = a(k).*PattCorr_Test(n-1,k)+((1-a(k)).*temp_test)/M;
            PattCorr_Ref(n,k) = a(k).*PattCorr_Ref(n-1,k)+((1-a(k)).*temp_ref)/M;
        end

    end
end


E_P_Ref = E_L_Ref.*PattCorr_Ref;
E_P_Test = E_L_Test.*PattCorr_Test;

end
