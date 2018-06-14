function [ y ] = PV_batch(x, N, TSM, filename)
%This standard phase vocoder processes each channel individually.
%   x is the input signal
%   N is the frame length.  Must be power of 2.
%   TSM is the TSM ratio 0.5 = 50%, 1 = 100% and 2.0 = 200% speed

% Tim Roberts - Griffith University 2018
for t = 1:length(TSM)
    tsm = TSM(t);
    %Calculate the number of channels
    num_chan = size(x,2);
    %Set time scale parameter
    alpha = 1/tsm;  % alpha<1 = compression, alpha>1 = expansion
    %Set Hop Factors
    Hs = N/4; %Synthesis Hop
    Ha = Hs/alpha; %Analysis Hop
    %Create window function (Hann) for correct number of channels
    w = 0.5*(1 - cos(2*pi*(0:N-1)'/(N-1)));
    wn = repmat(w,1,num_chan);
    %Zero pad to a multiple of N and normalise the input
    x = [zeros(N,num_chan); x; zeros(N-ceil(mod(length(x),Ha)),num_chan)]/max(max(abs(x)));
    %Initialise the output
    y = zeros(2*N+ceil(length(x)*alpha),num_chan);
    %Set omega for looped calculations
    omega_k = repmat(2*pi*(0:N/2)'/N,1,num_chan);
    %Set the initial phases
    last_input_phase = zeros (N/2+1,num_chan) ;
    last_Y_phase = zeros (N/2+1,num_chan) ;
    %using pointers set the analysis frame
    ptr_input = 1;
    ptr_output = 1;
    ptr_end = length(x)-N; %When ptr_in = ptr_end, stop
    %Initialise time instance variable (frame)
    frame = 1;
    
    while(ptr_input<ptr_end)
        frame_current = x(ptr_input:ptr_input+N-1,:).*wn;
        %Circular shift to place DC as first element (Not required in MATLAB)
        %frame_shifted = circshift(frame_current,length(frame_current)/2);
        FRAME_CURRENT = fft(frame_current);
        FRAME_CROP = FRAME_CURRENT(1:N/2+1,:); %Take only what's needed
        %Calculate Magnitude and phase
        mag = abs(FRAME_CROP);
        frame_phase = angle(FRAME_CROP);
        
        %Phase Vocoder
        if ptr_input == 1
            %Do initial setup
            y_frame = frame_current;
            Y_phase = frame_phase;
        else
            %Phase unwrapping as per Laroche and Dolson 1999
            %Calculate Instantaneous phase
            delta_phi = frame_phase-last_input_phase - Ha*omega_k;
            %Adjust instantaneous phase to be between -pi and +pi
            k = round(delta_phi/(2*pi)) ;
            delta_phi_adjust = delta_phi-k*2*pi;
            %Calculate the Instantaneous frequency
            inst_freq = omega_k + delta_phi_adjust/Ha;
            %Calculate the progression of phase
            Y_phase = last_Y_phase + Hs*inst_freq;
            %Create the output FFT
            Y = mag.*exp(1i*Y_phase);
            %iFFT Y (Circular shift not needed in MATLAB
            %y_frame = circshift(real(ifft([Y;conj(Y(end-1:-1:2,:))])),length(frame_current)/2).*wn;
            y_frame = real(ifft([Y;conj(Y(end-1:-1:2,:))])).*wn;
        end
        y(ptr_output:ptr_output+N-1,:) = y(ptr_output:ptr_output+N-1,:) + y_frame;
        %Increase the pointer location
        frame = frame+1;
        ptr_input = round((frame-1)*Ha)+1;
        ptr_output = ptr_output+Hs;
        %Store input and output phases
        last_input_phase = frame_phase;
        last_Y_phase = Y_phase;
    end
    %Normalise the output
    y = y(N+1:end,:)/max(max(abs(y)));
    f = [filename(1:end-4) sprintf('_PV_%g',tsm*100) '.wav'];
    audiowrite(f,y,fs);
end
end

