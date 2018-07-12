%Scratchpad
close all
clear all
clc
% TSM = 0.1;
% ak = rand(10,1);
% a = 1/TSM;
% 
% old_points = (1:length(ak))-1;  %-1 to 0 index
% new_points = round(a*old_points)+1; %+1 to 1 index
% 
% ak_hat = zeros(1,ceil(length(ak)*a));
% ak_hat(new_points) = ak(old_points+1);
% 
% count = 0;
% for n = 1:length(ak_hat)
%     if(ak_hat(n)~=0) && (count == 0)
%         low = ak_hat(n);
%         low_n = n;
%         count = count+1;
%     elseif(ak_hat(n)~=0 && count > 0)
%         high = ak_hat(n);
%         high_n = n;
%         count = count+1;
%         ak_hat(low_n:high_n) = linspace(low, high,count);
%         low = high;
%         low_n = high_n;
%         count = 1;
%     else
%         count = count+1;
%     end
% end
% 
% subplot(211)
% plot(ak)
% title('Original')
% subplot(212)
% plot(ak_hat)
% title('Linear interpolation')

t = 0:0.001:1;
x = sin(200*t);
y_up = resample(x,6,1);
y_down = resample(y_up,1,6);

figure
subplot(311)
plot(x)
title('x')
subplot(312)
plot(y_up)
title('upsample')
subplot(313)
plot(y_down)
title('downsample')