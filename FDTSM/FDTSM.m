function [ y ] = FDTSM( x, N, region_info )
%Frequency Dependent Time Scale Modification
% x is mono input signal
% N is the window size
% region_info should contain
    % region_info.TSM - Vector of TSM ratios
    % region_info.upper - Upper bounds of each region
                      % - max(region_info.upper) = N/2

if(max(region_info.upper) > N/2)
    fprintf('max(region_info.upper) > N/2. Please adjust region bounds\n');
end

                      
%Sum to mono if required
if (size(x,2) == 2)
    x = sum(x,2);
    num_chan = size(x,2);
else
    num_chan = 1;
end

%Calculate the lower bounds of each region
region_info.lower = [1 region_info.upper(1:end-1)+1];
region_info.num_regions = length(region_info.TSM);

%Expand the TSM vector
for r = 1:region_info.num_regions
    region_info.TSM_expanded(region_info.lower(r):region_info.upper(r)) = region_info.TSM(r);
end

%Set Synthesis hop size
Hs = N/4;
% alpha<1 = compression, alpha>1 = expansion
alpha = 1./region_info.TSM_expanded;
%Set Ha based Hs and time scale parameter
Ha = (Hs./alpha)';
%Create window function (Hann window)
wn = 0.5*(1 - cos(2*pi*(0:N-1)'/(N-1)));
%Zero pad and normalise the input
x = [zeros(N,num_chan); x; zeros(N,num_chan)] / max(max(abs(x)));
%Initialise the output
y = zeros(2*N+ceil(2*N+ceil(length(x)*max(alpha))),num_chan);
%Set omega k for looped calculations
temp = (0:N/2-1)';
omega_k = 2*pi*temp/N;
%Set the initial phases
last_input_phase = zeros (N/2,num_chan) ;
last_Y_phase = zeros (N/2,num_chan) ;
FRAME_COMP = zeros(N/2,1);
%Initialise input, output and end pointers
ptr_input = ones(region_info.num_regions,num_chan);
ptr_output = 1;
ptr_end = length(x)-N; %When ptr_in = ptr_end, stop
%Initialise time instance variable (frame)
frame = 1;

while(min(ptr_input)<ptr_end) %For the length of the file
    %Zero pad the end of the signal while waiting for low TSM bins to finish
    %!!!!!!!!!!!!This should be optimised!!!!!!!!!!!!!!!!
    while(max(ptr_input+N)>length(x))
        x = [x; zeros(N,num_chan)];
    end
    %Create the composite frame
    for r = 1:region_info.num_regions
        %Extract frame
        frame_current = x(ptr_input(r):ptr_input(r)+N-1,:).*wn;
        %Circular shift to place DC as first element
        %frame_shifted = circshift(frame_current,N/2); %Not req. in MATLAB
        %Calculate fft of current frame
        FRAME_CURRENT = fft(frame_current);
        %Colate only the appropriate frequency bins
        FRAME_COMP(region_info.lower(r):region_info.upper(r)) = FRAME_CURRENT(region_info.lower(r):region_info.upper(r));
    end
    
    %Calculate Magnitude and phase
    mag = abs(FRAME_COMP);
    phase = angle(FRAME_COMP);
    
    %Phase Vocoder
    if min(ptr_input) == 1
        %Copy first frame
        y_frame = real(ifft([FRAME_COMP;conj(FRAME_COMP(end:-1:1,:))])).*wn;
        Y_phase = phase;
    else
        %Phase Vocoder as per Laroche and Dolson 1999
        %Calculate Instantaneous phase
        delta_phi = phase - last_input_phase - Ha.*omega_k;
        %Adjust instantaneous phase to be between -pi and +pi
        k = round(delta_phi/(2*pi));
        delta_phi_adjust = delta_phi-k*2*pi;
        %Calculate the Instantaneous frequency
        inst_freq = omega_k + delta_phi_adjust./Ha;
        %Calculate the progression of phase
        Y_phase = last_Y_phase + Hs*inst_freq;
        %Create the output FFT
        Y = mag.*exp(1i*Y_phase);
        %iFFT Y
        y_frame = real(ifft([Y;conj(Y(end:-1:1,:))])).*wn;
        %Circular shift not needed in MATLAB
        %y_frame = circshift(real(ifft([Y;conj(Y(end:-1:1,:))])),N/2).*wn;
    end
    %Overlap add the output
    y(ptr_output:ptr_output+N-1,:) = y(ptr_output:ptr_output+N-1,:) + y_frame;
    %Advance the pointer locations
    frame = frame+1;
    ptr_input = round((frame-1)*Ha(region_info.lower(:)))+1;
    ptr_output = ptr_output+Hs;
    %Store input and output phases
    last_input_phase = phase;
    last_Y_phase = Y_phase;
end
%Normalise the output
y = y(N+1:end,:)/max(max(abs(y)));

end

