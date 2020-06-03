function [ E2 ] = fft_freq_spreading( Pp, fc, version )
%[ E2 ] = freq_spreading( Pp, fc, version )
%   Frequency Smearing of pitch patterns
%   ITU-R BS.1387-1 Section 2.1.7
%   Lower slope is 27 dB/Bark
%   Upper slope is frequency and energy dependent
global debug_var

if debug_var
    disp('  Frequency Spreading')
end
%Prepare for E2 calculation
if (strcmp(version, 'basic') || strcmp(version, 'Basic'))
    res = 0.25; %bark
    Z = 109;
elseif (strcmp(version, 'advanced') || strcmp(version, 'Advanced'))
    res = 0.5;  %bark
    Z = 55;
else
    disp('Unknown version')
    E2 = 0;
    return
end



%Calculate slopes (Equations 15,16)
L = 10*log10(Pp);
%Upper Slope
Su = -24 - repmat(230./fc,size(L,1),1) + 0.2*L;
%Lower Slope
Sl = 27; %dB/Bark
E_line_tilda = zeros(Z);
%Calculate E_line_tilda (Equation 20)
for k = 1:size(Pp,2)
    for j = 1:Z
        if k<j
            num = 10^((-res*(j-k)*Sl)/10);
            den1 = sum(10.^((-res*(j-(1:j))*Sl)/10));
            den2 = sum(10.^((res*((j+1:Z)-j)*Su(1,j))/10));
            E_line_tilda(k,j) = num./(den1+den2);
        else
            num = 10^((res*(k-j)*Su(1,j))/10);
            den1 = sum(10.^((-res*(j-(1:j))*Sl)/10));
            den2 = sum(10.^((res*((j+1:Z)-j)*Su(1,j))/10));
            E_line_tilda(k,j) = num./(den1+den2);
        end
    end
end
%Calculate Norm_SP (Equation 19)
Norm_SP = sum(E_line_tilda.^0.4,2).^(1/0.4);
Norm_SP_inv = 1./Norm_SP;
E2 =zeros(size(L));

for n = 1:size(L,1)
    E_line = zeros(Z);
    
    % Calculate the spreading
    %Calculate E_line (Equation 18)
    %Adjust the zero indexing
    
    for k = 1:size(Pp,2)
        for j = 1:Z
            %             E_line2(k,k<(1:Z)) = (10.^(L(n,(k<(1:Z)))/10)*10.^((-res*((k<(1:Z))-k)*Sl)/10)) / ...
            %                     (sum(10.^((-res*(j-(1:j))*Sl)/10)) + sum(10.^((res*((j+1:Z)-j)*Su(n,(k<(1:Z))))/10)));
            %             E_line2(k,k>=(1:Z)) = (10^(L(n,j)/10)*10^((res*(k-j)*Su(n,j))/10)) / ...
            %                     (sum(10.^((-res*(j-(1:j))*Sl)/10)) + sum(10.^((res*((j+1:Z)-j)*Su(n,j))/10)));
            %
            
            if k<j
                E_line(k,j) = (10^(L(n,j)/10)*10^((-res*(j-k)*Sl)/10)) / ...
                    (sum(10.^((-res*(j-(1:j))*Sl)/10)) + sum(10.^((res*((j+1:Z)-j)*Su(n,j))/10)));
            else
                E_line(k,j) = (10^(L(n,j)/10)*10^((res*(k-j)*Su(n,j))/10)) / ...
                    (sum(10.^((-res*(j-(1:j))*Sl)/10)) + sum(10.^((res*((j+1:Z)-j)*Su(n,j))/10)));
            end
        end
    end
    
    %Calculate E2 (Equation 17)
    E2(n,:) = Norm_SP_inv.*sum(E_line.^0.4,2).^(1/0.4);
    if debug_var
        fprintf('%.2f%%, ',100*n/size(Pp,1));
    end
end
if debug_var
    fprintf('\n');
end
end

