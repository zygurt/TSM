function [ E0 ] = fb_freq_spreading( x_ear, General )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

global debug_var

if debug_var
    disp('  Filterbank Frequency Spreading');
end

% const = fb_constants();
% fc = repmat(const.fc',1,size(x_ear,2),size(x_ear,3));
% L = 10*log10(abs(x_ear).^2);
% sl = 31;
% su = min(-4,-24-230./fc+0.2.*L);



%Pseudo code from ITU-R 1837-1
const = fb_constants();
z = hz2bark(const.fc);
dist = 0.1.^((z(40)-z(1))/(39*20));

%This should change to General.fs instead of 44100
t = 0.1;
fss = General.fs/32; %32 for downsampling
% a and b are currently as per ITU pseudocode.
a = exp(-1/(fss*t));
b = 1-a;

% for k = 1:40
A_re = x_ear(:,:,1);
A_im = x_ear(:,:,2);
% end

for k = 1:40
    cu = zeros(1,size(A_re,2));
    L = 10*log10(x_ear(k,:,1).^2+x_ear(k,:,2).^2+eps); %Sqrt and ^2 cancel
    s = max(4,(24+230./const.fc(k)-0.2*L));
%     s2 = min(-4,(-24-230./const.fc(k)+0.2*L));
    %Correct format of cu equation as per Kabal 2002
    %An Examination and Interpretation of ITU-R BS.1387: PEAQ
    for n = 2:size(s,2)
        cu(n) = b.*(s(n).^dist)+a*cu(n-1); %a and b are swapped in ITU psuedocode
    end
    
    d1 = x_ear(k,:,1);
    d2 = x_ear(k,:,2);
    
    for j = k:40
        d1 = d1.*cu;
        d2 = d2.*cu;
        A_re(j,:) = A_re(j,:)+d1;
        A_im(j,:) = A_im(j,:)+d2;
    end
end

cl = dist.^31;
d1 = 0;
d2 = 0;

for k=40:-1:1
    d1 = d1.*cl+A_re(k,:);
    d2 = d2.*cl+A_im(k,:);
    A_re(k,:) = d1;
    A_im(k,:) = d2;
    
    
end


%Rectification (Equation 34)
E0 = A_re.^2+A_im.^2;

end

