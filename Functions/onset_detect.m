function [ onsets ] = onset_detect( x, fs )
%[ onsets ] = onset_detect( x, fs )
%   HFC Onset detection based on:
%   "A Tutorial on Onset Detection in Music Signals" - Bello et al. 2005

num_strips = 4;




%Preprocessing
ms = 16;
overlap = 0.75;
N = 2^nextpow2(fs*ms*10^(-3));


x = [zeros(4*N,size(x,2));x;zeros(4*N,size(x,2))];

x_buf = buffer(x,N,overlap*N);
X = fft(x_buf,N);
X_mag = abs(X(1:end/2+1,:));

% %Convert to strips
% X_mag_strip = zeros(size(X_mag,2),4);
% 
% strips_top = (1:4)*N/(2*num_strips);
% strips_bottom = strips_top-(N/(2*num_strips)-1);
% strips_top(end) = strips_top(end)+1;
% for n = 1:num_strips
%     X_mag_strip(:,n) = sum(X_mag(strips_bottom(n):strips_top(n),:),1);
%    
% end
% plot(X_mag_strip);
% legend('Band 1','Band 2','Band 3','Band 4')

%Reduction

W = (1:N/2+1).^2;
W = (1:N/2+1);
W = repmat(W',1,size(X_mag,2));

E_tilda_part = W.*X_mag.^2;
E_tilda = sum(E_tilda_part,1);
E_tilda_log = log10(E_tilda);
E_tilda_log(E_tilda_log==-Inf) = 0;
E_log_tilda_diff = E_tilda_log(2:end)-E_tilda_log(1:end-1);

% E_tilda_log = E_tilda_log(2:end)+E_tilda_log(1:end-1);
% E_log_tilda_diff = E_tilda_log(2:end)-E_tilda_log(1:end-1);
E_log_tilda_diff(E_log_tilda_diff==-Inf) = 0;
E_log_tilda_diff(E_log_tilda_diff==Inf) = 0;
E_log_tilda_diff(isnan(E_log_tilda_diff)) = 0;



% %Plot the Original and Energy
% figure
% subplot(411)
% plot(x)
% title('Original')
% subplot(412)
% plot(E_tilda_log);
% title('Log10 E tilda');
% subplot(413)
% plot(E_log_tilda_diff);
% title('E log10 tilda diff')

s = 1;

%Peak Picking
%Remove mean
E_log_tilda_diff = E_log_tilda_diff-mean(E_log_tilda_diff);
%Divide by maximum absolute deviation
dev = max(abs(E_log_tilda_diff))-min(abs(E_log_tilda_diff));
E_log_tilda_diff = E_log_tilda_diff/dev;
%Low Pass filter
% E_log_tilda_diff = E_log_tilda_diff(2:end)+E_log_tilda_diff(1:end-1);



M = 10; %Should be half of 100ms according to paper
lambda = 1; %Set to 1 in the paper
delta = 0.1;  %Detection is sensitive to variations in delta
for n = M+1:length(E_log_tilda_diff)-M
    mov_median(n) = delta+lambda*median(abs(E_log_tilda_diff(n-M:n+M)));
end
mov_median(mov_median==0)=1;
% figure
% plot(mov_median)
E_log_tilda_diff = E_log_tilda_diff-[ones(size(mov_median,1),(length(E_log_tilda_diff)-length(mov_median))/2) mov_median ones(size(mov_median,1),(length(E_log_tilda_diff)-length(mov_median))/2)];
% figure
% plot(E_log_tilda_diff)
% subplot(414)
% plot(E_log_tilda_diff);
% title('Post-Processed')


p = local_peak(E_log_tilda_diff, 20, s); %Resolution of 125 ms
onsets = p*N*(1-overlap); %Convert frame to sample location

%Plot the transient locations
% subplot(411)
% hold on
% y_line = ones(size(onsets));
% line([onsets;onsets],[-1*y_line ; y_line],'Color','red');
% subplot(414)
% line([p;p],[-1*max(E_log_tilda_diff) ; max(E_log_tilda_diff)],'Color','red');
% hold off

%

onsets = onsets-4*N; %Remove the zero-padding


end

