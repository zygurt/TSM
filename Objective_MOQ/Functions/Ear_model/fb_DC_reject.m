function [ y ] = fb_DC_reject( x )
%[ y ] = fb_DC_reject( x )
%   Implemented as per ITU-R BS.1387-1 Section 2.2.4
%   4th order butterworth high-pass filter with fc of 20Hz
%   Cascade of two second order IIR filters
global debug_var

if debug_var
    disp('  Filter Bank DC Reject');
end

if(size(x,2)>size(x,1))
    x = x.';
end

% Pad with zeros
x = [0; 0; x; 0; 0];
y1 = zeros(size(x));
y = zeros(size(x));
%First second-order Butterworth filter
b = [1.99517, -0.995174];
for n = 3:length(x)-2
    y1(n) = x(n) - 2*x(n-1) + x(n-2) + b(1)*y1(n-1) + b(2)*y1(n-2);
end

%Second second-order Butterworth filter
b = [1.99799, -0.997998];
for n = 3:length(x)-2
    y(n) = x(n) - 2*x(n-1) + x(n-2) + b(1)*y(n-1) + b(2)*y(n-2);
end

%Remove padding
y = y(3:end-2);


%Checking against a single 4 pole butterworth filter
% [B,A] = butter(4,20/22050,'high');
% y_filter = filter(B,A,x);

% X = fft(x(2002:4050), 2048);
% Y = fft(y(2002:4050), 2048);
% Y_filter = fft(y_filter(2002:4050), 2048);
% semilogx(log10(abs(X(1:1029))));
% hold on
% semilogx(log10(abs(Y(1:1029))));
% semilogx(log10(abs(Y_filter(1:1029))));
% hold off
% legend('X','Y','Y Filter')


end

