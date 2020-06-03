function [ Model ] = fft_Ear_Model( x, General, length_match, ref_test)
%[ E_f, E2 ] = Ear_Model_FFT( x, version )
%   Implementation of the PEAQ FFT Ear Model
%   Implemented as per ITU-R BS.1387-1 Section 2.1

global debug_var

if debug_var
    disp('FFT Ear Modelling')
end
N = General.N;
w = PEAQ_Hann(N)';
%Find the start of the audio file
% x_int = round(abs(x)*32767); %PEAQ uses -32768:32767 as signal range
thresh = 200/32767;
sample_total = 0;
n = 1;
while (sample_total<thresh && n<length(x-4))
    sample_total = sum(abs(x(n:n+4)));
    n = n+1;
end
Model.ref_start = n-1;
%Find the end of the audio file
n = length(x);
sample_total = 0;
while (sample_total<thresh && n>5)
    sample_total = sum(abs(x(n-4:n)));
    n = n-1;
end
Model.ref_end = n+1;

switch length_match
    case 'Framing_Ref'
        switch ref_test
            case 'ref'
                x_buf = vec_buffer(x, N, General.frames);
            case 'test'
%                 frames = 1:General.BasicStepSize:length(General.sig_ref_len)-General.N;
                frame_loc = [1 floor(General.frames(2:end)/General.TSM)];
                x_buf = vec_buffer(x, N, frame_loc);
            otherwise
                error('Denote file using ''ref'' or ''test''');
        end

        Model.X = fft(x_buf.*repmat(w,1,size(x_buf,2)));
        Model.X_MAG = abs(Model.X);
    case 'Framing_Test'
        switch ref_test
            case 'ref'
                frame_loc = [1 floor(General.Test_frames(2:end)*General.TSM)];
                x_buf = vec_buffer(x, N, frame_loc);
            case 'test'
                x_buf = vec_buffer(x, N, General.Test_frames);
            otherwise
                error('Denote file using ''ref'' or ''test''');
        end

        Model.X = fft(x_buf.*repmat(w,1,size(x_buf,2)));
        Model.X_MAG = abs(Model.X);

    case 'Interpolate_fd_up'
        x_buf = buffer(x,N,N/2);
        Model.X = fft(x_buf.*repmat(w,1,size(x_buf,2)));
        Model.X_MAG = abs(Model.X);
        switch ref_test
            case 'ref'
                if General.sig_ref_len<General.sig_test_len
                    %Create a buffer with the same dimensions as the longest file.
                    temp_buf = buffer(1:General.sig_test_len,General.N,General.BasicStepSize);

                    Model.X_MAG_interp = zeros(size(temp_buf));
                    for n = 1:size(Model.X_MAG_interp,1)
                        Model.X_MAG_interp(n,:) = interp1(linspace(0,1,size(Model.X_MAG,2)),Model.X_MAG(n,:),linspace(0,1,size(temp_buf,2)));
                    end
                    Model.X_MAG = Model.X_MAG_interp;
                end

            case 'test'
                if General.sig_test_len<General.sig_ref_len
                    %Create a buffer with the same dimensions as the longest file.
                    temp_buf = buffer(1:General.sig_ref_len,General.N,General.BasicStepSize);

                    Model.X_MAG_interp = zeros(size(temp_buf));
                    for n = 1:size(Model.X_MAG_interp,1)
                        Model.X_MAG_interp(n,:) = interp1(linspace(0,1,size(Model.X_MAG,2)),Model.X_MAG(n,:),linspace(0,1,size(temp_buf,2)));
                    end
                    Model.X_MAG = Model.X_MAG_interp;
                end
            otherwise
                error('Denote file using ''ref'' or ''test''');
        end

    case 'Interpolate_fd_down'
        x_buf = buffer(x,N,N/2);
        Model.X = fft(x_buf.*repmat(w,1,size(x_buf,2)));
        Model.X_MAG = abs(Model.X);
        switch ref_test
            case 'ref'
                if General.sig_test_len<General.sig_ref_len
                    %Create a buffer with the same dimensions as the longest file.
                    temp_buf = buffer(1:General.sig_test_len,General.N,General.BasicStepSize);

                    Model.X_MAG_interp = zeros(size(temp_buf));
                    for n = 1:size(Model.X_MAG_interp,1)
                        Model.X_MAG_interp(n,:) = interp1(linspace(0,1,size(Model.X_MAG,2)),Model.X_MAG(n,:),linspace(0,1,size(temp_buf,2)));
                    end
                    Model.X_MAG = Model.X_MAG_interp;
                end

            case 'test'
                if General.sig_ref_len<General.sig_test_len
                    %Create a buffer with the same dimensions as the longest file.
                    temp_buf = buffer(1:General.sig_ref_len,General.N,General.BasicStepSize);

                    Model.X_MAG_interp = zeros(size(temp_buf));
                    for n = 1:size(Model.X_MAG_interp,1)
                        Model.X_MAG_interp(n,:) = interp1(linspace(0,1,size(Model.X_MAG,2)),Model.X_MAG(n,:),linspace(0,1,size(temp_buf,2)));
                    end
                    Model.X_MAG = Model.X_MAG_interp;
                end
            otherwise
                error('Denote file using ''ref'' or ''test''');
        end

    case 'Interpolate_to_ref'
        x_buf = buffer(x,N,N/2);
        Model.X = fft(x_buf.*repmat(w,1,size(x_buf,2)));
        Model.X_MAG = abs(Model.X);
        if strcmp(ref_test,'test')
            %Create a buffer with the same dimensions ref.
            temp_buf = buffer(1:General.sig_ref_len,General.N,General.BasicStepSize);
            Model.X_MAG_interp = zeros(size(temp_buf));
            for n = 1:size(Model.X_MAG_interp,1)
                Model.X_MAG_interp(n,:) = interp1(linspace(0,1,size(Model.X_MAG,2)),Model.X_MAG(n,:),linspace(0,1,size(temp_buf,2)));
            end
            Model.X_MAG = Model.X_MAG_interp;
        end

    case 'Interpolate_to_test'
        x_buf = buffer(x,N,N/2);
        Model.X = fft(x_buf.*repmat(w,1,size(x_buf,2)));
        Model.X_MAG = abs(Model.X);
        if strcmp(ref_test,'ref')
            %Create a buffer with the same dimensions as test.
            temp_buf = buffer(1:General.sig_test_len,General.N,General.BasicStepSize);

            Model.X_MAG_interp = zeros(size(temp_buf));
            for n = 1:size(Model.X_MAG_interp,1)
                Model.X_MAG_interp(n,:) = interp1(linspace(0,1,size(Model.X_MAG,2)),Model.X_MAG(n,:),linspace(0,1,size(temp_buf,2)));
            end
            Model.X_MAG = Model.X_MAG_interp;
        end
    otherwise
        error('Denote length_match using ''Framing_Ref'', ''Framing_Test'', ''Interpolate_fd_up'', ''Interpolate_fd_down'', ''Interpolate_to_ref'', ''Interpolate_to_test''');
end



%Calculate the scaling factor for each frame
fac = fft_scaling_factor_calculation(Model.X_MAG);
Model.F = Model.X_MAG.*repmat(fac,size(Model.X_MAG,1),1);

%Ear Modelling
Model.Fe = fft_outer_middle_ear( Model.F );
%Calculate the critial bands
Model.bands = fft_critical_bands(General.version);
%Group into critical bands
Pe = fft_pitch_mapping( Model.Fe, General.fs, Model.bands );
fc = Model.bands(2,:);
Pp = fft_internal_noise( Pe, fc );

Model.E2 = fft_freq_spreading( Pp, fc, General.version );

if (strcmp(General.version, 'basic') || strcmp(General.version, 'Basic'))
    Z = 109;
elseif (strcmp(General.version, 'advanced') || strcmp(General.version, 'Advanced'))
    Z = 55;
else
    disp('Unknown General.version')
    Z = 0;
    return
end

%Frame processing
Model.E = fft_time_spreading( Model.E2, Model.bands(2,:) );
%Apply Masking
Model.M = fft_masking_threshold( Model.E, General.version );

end
