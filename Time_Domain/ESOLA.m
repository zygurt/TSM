function [ y ] = ESOLA( x, N, TSM, fs )
%[ y ] = ESOLA( x, fs, TSM )
%   x = input signal
%   N = frame size (Normally 3 to 4 pitch periods) (20ms used in paper)
%   TSM = Time scale
%   fs = sampling frequency of signal
%   y = output signal
%   Implementation based on Rudresh et al., Epoch-Synchronous Overlap-Add
%   (ESOLA) for Time- and Pitch-Scale Modification of Speech Signals, 2018

% Tim Roberts - Griffith University 2018

num_chan = size(x,2);
if(num_chan > 1)
    disp('This ESOLA method currently only works for mono signals.');
    y = 0;
    return;
end
alpha = 1/TSM;
Ss = N/2;
Sa = round(Ss/alpha);

w = 0.5*(1 - cos(2*pi*(0:N-1)'/(N-1))); %hanning window

y = zeros(ceil(length(x)*alpha),num_chan); %1.1 is to allow for non exct time scaling
y_epochs = zeros(ceil(length(x)*alpha),num_chan);

epochs = ZFR(x, fs);

y(1:N,:) = x(1:N,:);
y_epochs(1:N,:) = epochs(1:N,:);
m = 2;
while m*Sa<length(x)-2*N
    %Create epoch frames
    in_epoch = epochs(m*Sa+1:m*Sa+N);
    out_epoch = y_epochs((m-1)*Ss+1:(m-1)*Ss+N);
    
    if(max(in_epoch) && max(out_epoch))
        %Must be at least one epoch in the current epoch frames
        first_out_epoch = 1;
        while(~out_epoch(first_out_epoch) && first_out_epoch<N)
            %Find the first epoch
            first_out_epoch = first_out_epoch+1;
        end
        
        km = 0;
        while(~in_epoch(first_out_epoch+km) && (first_out_epoch+km)<N)
            %Find the next closest epoch
            km = km+1;
        end
        if(first_out_epoch+km)==N
            km = 0;
        end
    else
        km = 0;
    end
    %Double check accessing bounds
    if(m*Sa+N+km)>length(x)
        disp('Accessed beyond original signal.  Returning signal processed thus far.')
        return;
    end
    %Select the final analysis frame
    in_grain = x(m*Sa+km+1:m*Sa+N+km).*w;
    %Overlap and add the new frame
    y(m*Ss+1:m*Ss+N,:) = y(m*Ss+1:m*Ss+N,:)+in_grain;
    %Overlap and add the epochs for the new frame
    y_epochs(m*Ss+1:m*Ss+N,:) = y_epochs(m*Ss+1:m*Ss+N,:)+epochs(m*Sa+km+1:m*Sa+N+km);
    %Increase the frame counter
    m = m+1;
end



end

