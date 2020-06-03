function [ X_w ] = fft_outer_middle_ear( X )
%[ X_w ] = outer_middle_ear( X )
%   Applies the weighting of the outer and middle ear as per
%   ITU-R BS.1387-1 Section 2.1.4
%   Equations 7-9
global debug_var

if debug_var
    disp('  Outer and Middle Ear')
end
fs = 44100;
N = 2048;
f = linspace(0,fs/2,N/2+1);

fkHz = f/1000;
%Rec ITU-R BS.1387-1 uses ^3.6 as final value
%Theide uses ^4 as final value
W_dB = -0.6*3.64*fkHz.^(-0.8) + ...
    6.5*exp(-0.6*(fkHz-3.3).^2) - ...
    (10^-3)*fkHz.^3.6;

% plot(f,W);
% W = -0.6*3.64*fkHz.^(-0.8) + ...
%     6.5*exp(-0.6*(fkHz-3.3).^2) - ...
%     (10^-3)*fkHz.^4;
% hold on
% plot(f,W);
% hold off
% legend('ITU-R','Theide')
% title('Outer and middle ear weighting function');
% xlabel('Frequency')
% ylabel('Weighting')

W = 10.^(W_dB/10);

X_w = abs(X(1:N/2+1,:)').*repmat((10.^(W/20)),size(X,2),1);

% figure
% plot(X(2:N/2+1))
% figure
% plot(X_w(2:end))


end

