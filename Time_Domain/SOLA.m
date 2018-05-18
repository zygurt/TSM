function [ y ] = SOLA( x, N, TSM )
%Synchronised Overlap Add (SOLA) Implementation
%   Roucos and Wilgus, High Quality Time-Scale Modification for Speech 1985
%   x is the input signal
%   N is the frame length.  Must be power of 2.
%   TSM is the TSM ratio 0.5 = 50%, 1 = 100% and 2.0 = 200% speed
% Minimum TSM value of 0.3  (This may be changed in the future, but is hard
% limited by the frame size.
% Tim Roberts - Griffith University 2018

% Time align successive windows with respect to signal similarity (magnitude and phase)
% before OLA by mximising the time-domain crosscorrelation between successive windows.
% Calculate cross correlation between current frame and the previous time
% adjusted frame.


alpha = 1/TSM;
Sa = N/4;
Ss = round(alpha*Sa);
if(Ss>=N)
    disp('TSM ratio too low');
    y = 0;
    return;
end
% 0. Pre-processing
xw = buffer(x,N,N-Sa);

% 1. Initialization:
% new_length = ceil(length(x)*alpha*1.1);
% y = zeros(new_length,size(x,2)); %Extra *1.1 is because time scale is not exact
y = xw(:,1);
% prev_end = N;
% 
% low_limit = N/2;
% high_limit = N/2;
% 
% k_min = N/2;
% k_max = N/2;

for m = 2:size(xw,2)
    %              Ss
    %         kmin km  kmax
    %y |=========(==|==)===|
    %x              |======|==========|
    %becomes
    %y_overlap (==|==)===|
    %x_overlap |======|
    
    recon_point = (m-1)*Ss;
    overlap_length = round((N-Ss)/2);
%     overlap_length = round(Ss/2)+mod(round(Ss/2),2);
   
    y_overlap = y(recon_point:recon_point+overlap_length-1);
    x_overlap = xw(1:overlap_length,m);
    
    [lag_x,lag_y, max_loc, xc_a] = maxcrosscorr(x_overlap,...
                                                y_overlap,...
                                                round(length(x_overlap)/4),...
                                                round(length(y_overlap)/4));
    fprintf('lag_x = %d, lag_y = %d, max_location = %d\n', lag_x, lag_y, max_loc);
    figure(1)
    plot(xc_a)
    adjusted_recon_point = recon_point-lag_x;
    total_overlap = length(y)-adjusted_recon_point;
  
    if(total_overlap > N)
        total_overlap = N;
    end
%     fade_in = 0:1/adjusted_overlap_length:1;
    fade_in = linspace(0,1,total_overlap)';
    fade_out = flipud(fade_in);
    
    y_fade = fade_out.*y(end-total_overlap+1:end);
    x_fade = fade_in.*xw(1:total_overlap,m);
    
    y = [y(1:end-total_overlap) ; y_fade+x_fade ; xw(total_overlap+1:end,m)];
    figure(2)
    plot(y)

%     overlap = N-Ss;
%     [lag_x, lag_y] = maxcrosscorr(x(prev_end-overlap+1:prev_end), xw(1:overlap,m), low_limit, high_limit); %lag of 0 means don't change anything
%     XCORRsegment = xcorr(xw(1:overlap,m),x(prev_end-overlap+1:prev_end));
%     [~,index]=max(XCORRsegment);
%     fprintf('lag_x = %d, lag_y = %d, MATLAB xcorr = %d\n', lag_x, lag_y, index);
%     %Extend estimate
% %     overlap_start_x = (m-1)*Ss+lag_x+1;
% %     overlap_end_x = (m-1)*Ss+lag_x+N;
% %     overlap_start_y = (m-1)*Ss+lag_y+1;
% %     overlap_end_y = (m-1)*Ss+lag_y+N;
%     overlap_start_y = prev_end-overlap+lag_y+1;
%     overlap_end_y = prev_end-overlap+lag_y+N;
%     
%     total_overlap = overlap-lag_y;
%     %Apply fade
%     fade_out = [ones(N-total_overlap,1);linspace(1,0,total_overlap)'];
%     if(overlap_start_y+N-1>length(y))
%         fprintf('STOP');
%     end
%     y(overlap_start_y:overlap_start_y+N-1) = y(overlap_start_y:overlap_start_y+N-1).*fade_out+xw(:,m).*flipud(fade_out);%.*w;
%     prev_end = m*Ss+N-1;

end

end



