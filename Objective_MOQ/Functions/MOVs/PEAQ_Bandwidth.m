function [MOV] = PEAQ_Bandwidth(Ref_F, Test_F, General)
%[MOV] = PEAQ_Bandwidth(Ref_F, Test_F)
%   As described by ITU-R BS.1387-1 Section 4.4.1
%Assumes that BW will be reduced due to processing
global debug_var

if debug_var
disp('    BandwidthRefB');
disp('    BandwidthTestB');
end
nf_sample_point = ceil((21000/(General.fs/2))*size(Ref_F,1)/2+1); %Noise floor above here. (21.5-24kHz when using fs=48kHz, using above 21k)
bw_cutoff_point = ceil((8000/(General.fs/2))*size(Ref_F,1)/2+1); %Only BW above 8 kHz

Ref_F_dB = 20*log10(Ref_F(1:size(Ref_F,1)/2+1,:));
Test_F_dB = 20*log10(Test_F(1:size(Test_F,1)/2+1,:));

BwRef = zeros(size(Ref_F_dB,2),1);
BwTest = zeros(size(Test_F_dB,2),1);
BwTest_new = zeros(size(Test_F_dB,2),1);
for n = 1:size(Ref_F_dB,2)
    ZeroThreshold = Test_F_dB(nf_sample_point,n);
    for k=nf_sample_point:size(Test_F_dB,1)
        ZeroThreshold = max(ZeroThreshold,Test_F_dB(k,n));
    end
    %First bin counting back that exceeds zero threshold+10dB
    for k = nf_sample_point-1:-1:1
        if Ref_F_dB(k,n)>= 10+ZeroThreshold
            BwRef(n) = k+1;
            break;
        end
    end
    %Next bin counting back that exceeds zero threshold + 5dB
    for k = BwRef(n):-1:1
        if Test_F_dB(k,n) >= 5+ZeroThreshold
            BwTest(n)=k+1;
            break
        end
    end
    %Next bin counting back that exceeds zero threshold + 10dB
    for k = nf_sample_point:-1:1
        if Test_F_dB(k,n) >= 10+ZeroThreshold
            BwTest_new(n)=k+1;
            break
        end
    end
end

MOV.BandwidthRefB = PEAQ_Temporal_Average(BwRef(BwRef>bw_cutoff_point),'Linear');
MOV.BandwidthTestB = PEAQ_Temporal_Average(BwTest(BwRef>bw_cutoff_point),'Linear');
MOV.BandwidthTestB_new = PEAQ_Temporal_Average(BwTest_new(BwRef>bw_cutoff_point),'Linear');
end

