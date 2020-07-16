function [sig_ref, sig_test, General] = Signal_Prep(sig_ref, sig_test, General)
%function [sig_ref, sig_test, General] = Signal_Prep(sig_ref, sig_test, General)
%   Prepare the initial signal for objective processing.
global debug_var

if debug_var
    disp('Signal Preparation')
end

%Sum the stereo test file to mono
sig_ref = sum(sig_ref,2);
sig_test = sum(sig_test,2);  %Signals are already a single channel, but this is a double check.

%Remove any initial silence
n = 1;
while sig_ref(n)==0
    n = n+1;
end
first_sig_sample = n;
n = length(sig_ref);
while sig_ref(n)==0
    n = n-1;
end
last_sig_sample = n;
sig_ref = sig_ref(first_sig_sample:last_sig_sample);

n = 1;
while sig_test(n)==0
    n = n+1;
end
first_sig_sample = n;
n = length(sig_test);
while sig_test(n)==0
    n = n-1;
end
last_sig_sample = n;
sig_test = sig_test(first_sig_sample:last_sig_sample);

% %Remove any DC offset
sig_ref = sig_ref-mean(sig_ref);
sig_test = sig_test-mean(sig_test);

%Normalise the signals to account for normalisation during playback
sig_ref = sig_ref./max(abs(sig_ref));
sig_test = sig_test/max(abs(sig_test));

% Find the start of the reference signal
sample_ave = 0;
threshold = 0.0061; %Same threshold for data boundary (is 200/32767 from 5.2.4.4)
duration = 4;
r_start = 1;
while sample_ave<threshold && r_start<(length(sig_ref)-5)
    sample_ave = sum(abs(sig_ref(r_start:r_start+duration)));
    r_start = r_start+1;
end

%Find the end of the test signal
sample_ave = 0;
r_end = length(sig_ref)-duration;
while sample_ave<threshold && r_end>1
    sample_ave = sum(abs(sig_ref(r_end:r_end+duration)));
    r_end = r_end-1;
end

sig_ref = sig_ref(r_start-1:r_end-1);

%Find the start of the test signal
sample_ave = 0;
t_start = 1;
while sample_ave<threshold && t_start<(length(sig_test)-5)
    sample_ave = sum(abs(sig_test(t_start:t_start+duration)));
    t_start = t_start+1;
end

%Find the end of the test signal
sample_ave = 0;
t_end = length(sig_test)-duration;
while sample_ave<threshold && t_end>1
    sample_ave = sum(abs(sig_test(t_end:t_end+duration)));
    t_end = t_end-1;
end



sig_test = sig_test(t_start-1:t_end-1);

General.frames = 1:General.BasicStepSize:length(sig_ref)-General.N;
General.Test_frames = 1:General.BasicStepSize:length(sig_test)-General.N;
General.sig_ref_len = length(sig_ref);
General.sig_test_len = length(sig_test);

% if(ceil(length(sig_ref)/General.TSM)~=length(sig_test))
%
%     %Find the start of the reference signal
%     sample_ave = 0;
%     threshold = 0.0061; %Same threshold for data boundary
%     duration = 4;
%     n = 1;
%     while sample_ave<threshold && n<length(sig_ref-5)
%         sample_ave = sum(abs(sig_ref(n:n+duration)));
%         n = n+1;
%     end
%     sig_ref = sig_ref(n-1:end);
%
%     %Find the start of the test signal
%     sample_ave = 0;
%     n = 1;
%     while sample_ave<threshold && n<length(sig_test-5)
%         sample_ave = sum(abs(sig_test(n:n+duration)));
%         n = n+1;
%     end
%     sig_test = sig_test(n-1:end);
%
%
%     %Make the Signals the 'same' length
%     if(ceil(length(sig_ref)/General.TSM) < length(sig_test))
%         sig_test = sig_test(1:ceil(length(sig_ref)/General.TSM));
%
%     elseif(ceil(length(sig_ref)/General.TSM) > length(sig_test))
%         sig_test = [sig_test;zeros(ceil(length(sig_ref)/General.TSM)-length(sig_test),1)];
%
%     end
%
%     %Test the framing
%     ref_buf = buffer(sig_ref, General.N, General.N/2);
%     test_buf = buffer(sig_test, General.N, round(General.N-(General.N/2)/General.TSM));
%     General.ref_frames = size(ref_buf,2);
%     General.test_frames = size(test_buf,2);
%     if(size(ref_buf,2)~=size(test_buf,2))
%         fprintf('%s framed has a different length\n',General.Testname);
%     end
%
%
% else
%
%     %Test the framing
%     ref_buf = buffer(sig_ref, General.N, General.N/2);
%     test_buf = buffer(sig_test, General.N, round(General.N-(General.N/2)/General.TSM));
%     if(size(ref_buf,2)~=size(test_buf,2))
%         fprintf('%s framed has a different length\n',General.Testname);
%     end
% end



end
