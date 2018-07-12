function [ y ] = SOLA( x, N, TSM )
% [ y ] = SOLA( x, N, TSM )
% Synchronised Overlap Add (SOLA) Time-Scale Modification Implementation
%   Roucos and Wilgus, High Quality Time-Scale Modification for Speech 1985
%   x is the input signal
%   N is the frame length.  Must be power of 2. 4096 recommended for FS = 44.1kHz
%   TSM is the TSM ratio 0.5 = 50%, 1 = 100% and 2.0 = 200% speed
% Minimum TSM value of 0.3, for frame size of ~92 ms. Hard limited by the frame size.
% Time align successive windows with respect to signal similarity (magnitude and phase)
% before OLA by mximising the time-domain crosscorrelation between successive windows.
% Calculate cross correlation between current frame and the previous time
% adjusted frame.

% Tim Roberts - Griffith University 2018
if(size(x,2) > 1)
    disp('This SOLA method currently only works for mono signals.');
    disp('Max cross correlation lag is single channel only.');
    y = 0;
    return;
end
alpha = 1/TSM;
%Analysis shift size
Sa = N/4;
%Synthesis shift size
Ss = round(alpha*Sa);
%Synthesis Shift must be less than the frame length
if(Ss>=N)
    disp('TSM ratio too low. TSM>=0.3 for now.');
    y = 0;
    return;
end
%Split original signal into frames
xw = buffer(x,N,N-Sa);
%Copy the first frame to the output
y = xw(:,1);
%Process the remaining frames
for m = 2:size(xw,2)
    %Calculate the reconstruction point
    recon_point = (m-1)*Ss;
    %Calculate the length of overlap
    overlap_length = round((N-Ss)/2); %Half of recon to end
    %overlap_length = round(Ss/2)+mod(round(Ss/2),2); %From recon to end
    %Create the signals for cross correlation
    y_overlap = y(recon_point:recon_point+overlap_length-1);
    x_overlap = xw(1:overlap_length,m);
    %Calculate the offset for x for maximum cross correlation
    [lag_x, ~] = maxcrosscorrlag(x_overlap,...
                                 y_overlap,...
                                 round(length(x_overlap)/4),...
                                 round(length(y_overlap)/4));
    %Adjust the reconstruction point
    adjusted_recon_point = recon_point-lag_x;
    total_overlap = length(y)-adjusted_recon_point;
    %Limit the overlap to the length of the frame
    if(total_overlap > N)
        total_overlap = N;
    end
    %Create fade windows
    fade_in = linspace(0,1,total_overlap)';
    fade_out = flipud(fade_in);
    %Apply fades
    y_fade = fade_out.*y(end-total_overlap+1:end);
    x_fade = fade_in.*xw(1:total_overlap,m);
    %Overlap add to the output signal
    y = [y(1:end-total_overlap) ; y_fade+x_fade ; xw(total_overlap+1:end,m)];
    
end
    y = y/max(abs(y));
end



