function [ p ] = local_peak( x, N, s )
%[ p ] = local_peak( x, N )
%   Finds the local peaks within range N for signal x
%   x = signal
%   N = checking range
%   s = sensitivity

p = [];
n = floor(N/2);
sig_len = length(x);
if(size(x,1)>size(x,2))
    x = x';
end
x = [zeros(1,n) , x , zeros(1,n)];
for t = n+1:sig_len-n
    if(sum(x(t)>x(t-n:t+n))==N) && x(t)>0% && x(t)>(mean(x(t-n:t+n))+std(x(t-n:t+n)))%Maybe the mean should include the current value
%         disp(t)
        p = [p t];
        %Could increase the speed of this by incrementing by n if peak is
        %found
    end
end

% figure
% subplot(211)
% plot(x(p))
% title('Peak Values')
% subplot(212)
% hist(x(p))
% title('Peak value Histogram')

% p_mean = mean(x(p));
% p_std = std(x(p));
% x_mean = mean(x);
% x_std = std(x);

% figure(1)
% subplot(313)
% hold on
% line([0 1600],[x_mean+2*x_std x_mean+2*x_std])
% line([0 length(x)],[p_mean*s p_mean*s])
% hold off

% p = p(x(p)>(p_mean*s));%+p_std)); %Could replace this with finding first min between modes

%Remove Zero Padding
p = p-n;




end

