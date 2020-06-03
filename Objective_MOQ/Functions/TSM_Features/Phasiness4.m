function [Ph_NW, Ph_MW] = Phasiness4(sig_ref, sig_test, General)
%[Ph_NW, Ph_MW] = Phasiness4(sig_ref, sig_test, General)
%   Returns a measure of Phasiness
global debug_var

if debug_var
    disp('Phasiness 4');
end

N = 2048;
L = 1*N/4;
fade_length = 4*N;
fade = 0.5*(1 - cos(2*pi*(0:fade_length-1)'/(fade_length-1)));

%----------Unwrap the phase of the reference signal-----
sig_ref = mean(sig_ref,2);
fade_signal = ones(size(sig_ref));
fade_signal(1:length(fade)/2) = fade(1:length(fade)/2);
fade_signal(end-length(fade)/2:end) = fade(end-length(fade)/2:end);
sig_ref = sig_ref.*fade_signal;
sig_ref_buf = buffer(sig_ref,N,N-L);
w = 0.5*(1 - cos(2*pi*(0:N-1)'/(N-1)));
wn = repmat(w,1,size(sig_ref_buf,2));
SIG_REF_BUF = fft(wn.*sig_ref_buf,N);
SIG_REF_BUF_ANGLE = angle(SIG_REF_BUF(1:N/2+1,:));

SIG_REF_BUF_ANGLE(SIG_REF_BUF_ANGLE<0) = 2*pi+SIG_REF_BUF_ANGLE(SIG_REF_BUF_ANGLE<0);

SIG_REF_BUF_ANGLE_UNWRAP = zeros(size(SIG_REF_BUF_ANGLE));
SIG_REF_BUF_ANGLE_UNWRAP(:,1) = SIG_REF_BUF_ANGLE(:,1);

for k = 1:size(SIG_REF_BUF_ANGLE,1)
    for n = 2:size(SIG_REF_BUF_ANGLE,2)
        temp = SIG_REF_BUF_ANGLE(k,n);
        while(temp<SIG_REF_BUF_ANGLE_UNWRAP(k,n-1))
            temp = temp+2*pi;
        end
        SIG_REF_BUF_ANGLE_UNWRAP(k,n) = temp;
    end
end

%----------Unwrap the phase of the test signal-----
fade_signal = ones(size(sig_test));
fade_signal(1:length(fade)/2) = fade(1:length(fade)/2);
fade_signal(end-length(fade)/2:end) = fade(end-length(fade)/2:end);
sig_test = sig_test.*fade_signal;
sig_test_buf = buffer(sig_test,N,N-L);
w = 0.5*(1 - cos(2*pi*(0:N-1)'/(N-1)));
wn = repmat(w,1,size(sig_test_buf,2));
SIG_TEST_BUF = fft(wn.*sig_test_buf,N);

SIG_TEST_BUF_ANGLE = angle(SIG_TEST_BUF(1:N/2+1,:));

SIG_TEST_BUF_ANGLE(SIG_TEST_BUF_ANGLE<0) = 2*pi+SIG_TEST_BUF_ANGLE(SIG_TEST_BUF_ANGLE<0);

SIG_TEST_BUF_ANGLE_UNWRAP = zeros(size(SIG_TEST_BUF_ANGLE));
SIG_TEST_BUF_ANGLE_UNWRAP(:,1) = SIG_REF_BUF_ANGLE(:,1);
for k = 1:size(SIG_TEST_BUF_ANGLE,1)
    for n = 2:size(SIG_TEST_BUF_ANGLE,2)
        temp = SIG_TEST_BUF_ANGLE(k,n);
        while(temp<SIG_TEST_BUF_ANGLE_UNWRAP(k,n-1))
            temp = temp+2*pi;
        end
        SIG_TEST_BUF_ANGLE_UNWRAP(k,n) = temp;
    end
end
%Phasiness2 version
% SIG_TEST_BUF_ANGLE_UNWRAP_DEC = General.TSM*resample(SIG_TEST_BUF_ANGLE_UNWRAP',size(SIG_REF_BUF_ANGLE_UNWRAP,2),size(SIG_TEST_BUF_ANGLE_UNWRAP,2))';
%Use Linear interpolation instead

% f = linspace(0,General.fs/2,N/2+1);

% fkHz = f/1000;
%Rec ITU-R BS.1387-1 uses ^3.6 as final value
%Theide uses ^4 as final value
% W_dB = -0.6*3.64*fkHz.^(-0.8) + ...
%     6.5*exp(-0.6*(fkHz-3.3).^2) - ...
%     (10^-3)*fkHz.^3.6;



% Check to see whether ref or test is longer
if size(SIG_REF_BUF_ANGLE_UNWRAP,2)<size(SIG_TEST_BUF_ANGLE_UNWRAP,2)
    %TSM ratio less than 100%
    t_target = 1:size(SIG_REF_BUF_ANGLE_UNWRAP,2);
    t_source = 1:size(SIG_TEST_BUF_ANGLE_UNWRAP,2);
    SIG_TEST_BUF_ANGLE_UNWRAP_INT = zeros(size(SIG_REF_BUF_ANGLE_UNWRAP));
    for k = 1:size(SIG_TEST_BUF_ANGLE)
        SIG_TEST_BUF_ANGLE_UNWRAP_INT(k,:) = interp1(t_source, SIG_TEST_BUF_ANGLE_UNWRAP(k,:), t_target);
    end
    
%     dur = length(sig_ref)/General.fs;
    
    
    ANGLE_DIFF = (SIG_REF_BUF_ANGLE_UNWRAP-General.TSM*SIG_TEST_BUF_ANGLE_UNWRAP_INT);%/dur;
    SIG_REF_BUF_MAG = abs(SIG_REF_BUF(1:N/2+1,:));
    SIG_REF_BUF_MAG_NORM = SIG_REF_BUF_MAG/max(max(SIG_REF_BUF_MAG));
%     SIG_REF_BUF_POW = SIG_REF_BUF_MAG_NORM.^2;
    ANGLE_DIFF_LOUD_MAG = ANGLE_DIFF.*SIG_REF_BUF_MAG_NORM;
%     ANGLE_DIFF_LOUD_POW = ANGLE_DIFF.*SIG_REF_BUF_POW;
%     W = 10.^(W_dB'/10);
%     W = W/max(W);
%     Wn = repmat(W,1,length(t_target));
%     ANGLE_DIFF_LOUD_FM = ANGLE_DIFF.*Wn;
    
%     if(debug_var)
%         figure(4)
%         subplot(411)
%         plot(SIG_REF_BUF_ANGLE_UNWRAP')
%         title('Reference')
%         subplot(412)
%         plot(SIG_TEST_BUF_ANGLE_UNWRAP_INT')
%         title('Test')
%         subplot(413)
%         plot((SIG_REF_BUF_ANGLE_UNWRAP-SIG_TEST_BUF_ANGLE_UNWRAP_INT)')
%         title('Unscaled Difference')
%         subplot(414)
%         plot((SIG_REF_BUF_ANGLE_UNWRAP-General.TSM*SIG_TEST_BUF_ANGLE_UNWRAP_INT)')
%         title('Scaled Difference')
%         REF_ANGLE = SIG_REF_BUF_ANGLE_UNWRAP;
%         TEST_ANGLE = General.TSM*SIG_TEST_BUF_ANGLE_UNWRAP_INT;
%     end
else
    t_target = 1:size(SIG_TEST_BUF_ANGLE_UNWRAP,2);
    t_source = 1:size(SIG_REF_BUF_ANGLE_UNWRAP,2);
    SIG_REF_BUF_ANGLE_UNWRAP_INT = zeros(size(SIG_TEST_BUF_ANGLE_UNWRAP));
    for k = 1:size(SIG_TEST_BUF_ANGLE)
        SIG_REF_BUF_ANGLE_UNWRAP_INT(k,:) = interp1(t_source, SIG_REF_BUF_ANGLE_UNWRAP(k,:), t_target);
    end
    
%     dur = length(sig_ref)/General.fs;
    
    
    ANGLE_DIFF = (General.TSM*SIG_REF_BUF_ANGLE_UNWRAP_INT-SIG_TEST_BUF_ANGLE_UNWRAP);%/dur;
    SIG_REF_BUF_MAG = abs(SIG_REF_BUF(1:N/2+1,:));
    SIG_REF_BUF_MAG_NORM = SIG_REF_BUF_MAG/max(max(SIG_REF_BUF_MAG));
    SIG_TEST_BUF_MAG = abs(SIG_TEST_BUF(1:N/2+1,:));
    SIG_TEST_BUF_MAG_NORM = SIG_TEST_BUF_MAG/max(max(SIG_TEST_BUF_MAG));
%     SIG_TEST_BUF_POW = SIG_TEST_BUF_MAG_NORM.^2;
    ANGLE_DIFF_LOUD_MAG = ANGLE_DIFF.*SIG_TEST_BUF_MAG_NORM;
%     ANGLE_DIFF_LOUD_POW = ANGLE_DIFF.*SIG_TEST_BUF_POW;
%     W = 10.^(W_dB'/10);
%     W = W/max(W);
%     Wn = repmat(W,1,length(t_target));
%     ANGLE_DIFF_LOUD_FM = ANGLE_DIFF.*Wn;
    
%     if(debug_var)
%         figure(4)
%         subplot(411)
%         plot(SIG_REF_BUF_ANGLE_UNWRAP_INT')
%         title('Reference')
%         subplot(412)
%         plot(SIG_TEST_BUF_ANGLE_UNWRAP')
%         title('Test')
%         subplot(413)
%         plot((SIG_REF_BUF_ANGLE_UNWRAP_INT-SIG_TEST_BUF_ANGLE_UNWRAP)')
%         title('Unscaled Difference')
%         subplot(414)
%         plot((General.TSM*SIG_REF_BUF_ANGLE_UNWRAP_INT-SIG_TEST_BUF_ANGLE_UNWRAP)')
%         title('Scaled Difference')
%         REF_ANGLE = General.TSM*SIG_REF_BUF_ANGLE_UNWRAP_INT;
%         TEST_ANGLE = SIG_TEST_BUF_ANGLE_UNWRAP;
%     end
end





%Calculate the mean absolute difference
Ph_mad_t = mean(abs(ANGLE_DIFF),1);
Ph_mad_f = mean(abs(ANGLE_DIFF),2);
% Ph.Ph_m_t = mean(mean(ANGLE_DIFF,1));
% Ph.ave_P_ave_t = mean(Ph_mad_t);
Ph_NW.ave_P_ave_f = mean(Ph_mad_f);
Ph_NW.std_P_ave_f = std(Ph_mad_f);
% Ph.P_ave_end = mean(abs(ANGLE_DIFF(:,end)));
% Ph.P_std_end = std(abs(ANGLE_DIFF(:,end)));
% Ph_mad_t_diff = Ph_mad_t(2:end)-Ph_mad_t(1:end-1); %First difference
% Ph.ave_Ph_mad_t_diff = mean(Ph_mad_t_diff);


% Ph_mad_t_mag = mean(abs(ANGLE_DIFF_LOUD_MAG),1);
Ph_mad_f_mag = mean(abs(ANGLE_DIFF_LOUD_MAG),2);
% Ph_mag.Ph_m_t_mag = mean(mean(ANGLE_DIFF_LOUD_MAG,1));
% Ph_mag.ave_P_ave_t_mag = mean(Ph_mad_t_mag);
Ph_MW.ave_P_ave_f_mag = mean(Ph_mad_f_mag);
Ph_MW.std_P_ave_f_mag = std(Ph_mad_f_mag);
% Ph_mag.P_ave_end_mag = mean(abs(ANGLE_DIFF_LOUD_MAG(:,end)));
% Ph_mag.P_std_end_mag = std(abs(ANGLE_DIFF_LOUD_MAG(:,end)));
% Ph_mad_t_diff_mag = Ph_mad_t_mag(2:end)-Ph_mad_t_mag(1:end-1);
% Ph_mag.ave_Ph_mad_t_diff_mag = mean(Ph_mad_t_diff_mag);

% Ph_mad_t_pow = mean(abs(ANGLE_DIFF_LOUD_POW),1);
% Ph_mad_f_pow = mean(abs(ANGLE_DIFF_LOUD_POW),2);
% Ph_pow.Ph_m_t_pow = mean(mean(ANGLE_DIFF_LOUD_POW,1));
% Ph_pow.ave_P_ave_t_pow = mean(Ph_mad_t_pow);
% Ph_pow.ave_P_ave_f_pow = mean(Ph_mad_f_pow);
% Ph_pow.std_P_ave_f_pow = std(Ph_mad_f_pow);
% Ph_pow.P_ave_end_pow = mean(abs(ANGLE_DIFF_LOUD_POW(:,end)));
% Ph_pow.P_std_end_pow = std(abs(ANGLE_DIFF_LOUD_POW(:,end)));
% Ph_mad_t_diff_pow = Ph_mad_t_pow(2:end)-Ph_mad_t_pow(1:end-1);
% Ph_pow.ave_Ph_mad_t_diff_pow = mean(Ph_mad_t_diff_pow);

% Ph_mad_t_fm = mean(abs(ANGLE_DIFF_LOUD_FM),1);
% Ph_mad_f_fm = mean(abs(ANGLE_DIFF_LOUD_FM),2);
% Ph_fm.Ph_m_t_fm = mean(mean(ANGLE_DIFF_LOUD_FM,1));
% Ph_fm.ave_P_ave_t_fm = mean(Ph_mad_t_fm);
% Ph_fm.ave_P_ave_f_fm = mean(Ph_mad_f_fm);
% Ph_fm.std_P_ave_f_fm = std(Ph_mad_f_fm);
% Ph_fm.P_ave_end_fm = mean(abs(ANGLE_DIFF_LOUD_FM(:,end)));
% Ph_fm.P_std_end_fm = std(abs(ANGLE_DIFF_LOUD_FM(:,end)));
% Ph_mad_t_diff_fm = Ph_mad_t_fm(2:end)-Ph_mad_t_fm(1:end-1);
% Ph_fm.ave_Ph_mad_t_diff_fm = mean(Ph_mad_t_diff_fm);






% if debug_var
%     
%     
%     
%     %         figure
%     %         plot(ANGLE_DIFF(:,end))
%     %         title('Final Difference Frame')
%     %         xlabel('Frequency (Bin)')
%     %         ylabel('Phase Difference')
%     %
%     %         figure
%     %         subplot(211)
%     %         plot(sig_ref)
%     %         title('Reference')
%     %         subplot(212)
%     %         plot(sig_test)
%     %         title('Test')
%     %         %
%     %         figure
%     %         subplot(311)
%     %         s = surf(SIG_REF_BUF_POW);%(:,300:400));
%     %         set(gca,'YScale','log')
%     %         s.EdgeColor = 'none';
%     %         title('Ref Power')
%     %         colormap(flipud(gray));
%     %         colorbar
%     %         view(2)
%     %         subplot(312)
%     %         s1 = surf(SIG_REF_BUF_ANGLE);%(:,300:400));
%     %         set(gca,'YScale','log')
%     %         s1.EdgeColor = 'none';
%     %         title('Ref Phase')
%     %         colorbar
%     %         view(2)
%     %         subplot(313)
%     %         s2 = surf(SIG_REF_BUF_POW.*SIG_REF_BUF_ANGLE);%(:,300:400));
%     %         set(gca,'YScale','log')
%     %         s2.EdgeColor = 'none';
%     %         title('Ref Power x Phase')
%     %         colorbar
%     %         view(2)
%     
%     %         figure
%     %         subplot(311)
%     %         s = surf(SIG_TEST_BUF_POW);%(:,300:400));
%     %         set(gca,'YScale','log')
%     %         s.EdgeColor = 'none';
%     %         title('Test Power')
%     %         colormap(flipud(gray));
%     %         colorbar
%     %         view(2)
%     %         subplot(312)
%     %         s1 = surf(SIG_TEST_BUF_ANGLE);%(:,300:400));
%     %         set(gca,'YScale','log')
%     %         s1.EdgeColor = 'none';
%     %         title('Test Phase')
%     %         colorbar
%     %         view(2)
%     %         subplot(313)
%     %         s2 = surf(SIG_TEST_BUF_POW.*SIG_TEST_BUF_ANGLE);%(:,300:400));
%     %         set(gca,'YScale','log')
%     %         s2.EdgeColor = 'none';
%     %         title('Test Power x Phase')
%     %         colorbar
%     %         view(2)
%     
%     figure(1)
%     subplot(411)
%     s = surf(REF_ANGLE);
%     set(gca,'YScale','log')
%     s.EdgeColor = 'none';
%     view(2)
%     % colormap(flipud(gray));
%     colorbar
%     title('Reference')
%     
%     subplot(412)
%     s = surf(TEST_ANGLE);
%     set(gca,'YScale','log')
%     s.EdgeColor = 'none';
%     view(2)
%     % colormap(flipud(gray));
%     colorbar
%     title('Test')
%     
%     subplot(413)
%     s = surf(ANGLE_DIFF);
%     set(gca,'YScale','log')
%     s.EdgeColor = 'none';
%     view(2)
%     % colormap(flipud(gray));
%     colorbar
%     title('Difference')
%     
%     subplot(414)
%     s = surf(SIG_REF_BUF_MAG_NORM);%(:,300:400));
%     set(gca,'YScale','log')
%     s.EdgeColor = 'none';
%     title('Ref Power')
%     %     colormap(flipud(gray));
%     colorbar
%     view(2)
%     title('Reference Magnitude Spectrum')
%     t = strrep(General.Testname,'_','\_');
%     suptitle(t)
%     a = split(General.Testname,'/');
%     fname = ['Plots/Phasiness/' char(a{3}) '1.png'];
%     %     print(fname, '-dpng');
%     
%     
%     %         figure
%     %         subplot(411)
%     %         plot(SIG_REF_BUF_ANGLE_UNWRAP_INT')
%     %         title('Reference Phase Unwrapped')
%     %
%     %         subplot(412)
%     %         plot(SIG_TEST_BUF_ANGLE_UNWRAP')
%     %         title('Test Phase Unwrapped')
%     %
%     %         subplot(413)
%     %         plot(SIG_TEST_BUF_ANGLE_UNWRAP_DEC')
%     %         title('Test Phase Unwrapped Resampled')
%     %
%     %         subplot(414)
%     %         plot(SIG_TEST_BUF_ANGLE_UNWRAP_INT')
%     %         title('Test Phase Unwrapped Linear Interp')
%     
%     %             figure
%     %             for n=1:size(SIG_REF_BUF_ANGLE_UNWRAP,1)
%     %     %         for n=47
%     %                 plot(SIG_REF_BUF_ANGLE_UNWRAP_SMOOTH(n,:))
%     %                 hold on
%     %                 plot(SIG_TEST_BUF_ANGLE_UNWRAP_DEC_SMOOTH(n,:))
%     %                 hold off
%     %                 t = sprintf('Bin %d',n);
%     %                 title(t)
%     %                 legend('Reference','Test','Location','southeast')
%     %             end
%     %         if(General.TSM>1.5)
%     
%     figure(2)
%     subplot(411)
%     plot(Ph_mad_t)
%     title('Phasiness (Time)')
%     xlabel('Time (Frame)')
%     ylabel('Phasiness')
%     subplot(412)
%     plot(Ph_mad_t_diff)
%     title('Phasiness 1st Diff (Time)')
%     xlabel('Time (Frame)')
%     ylabel('Phasiness 1st Diff')
%     
%     subplot(413)
%     semilogx(Ph_mad_f)
%     title('Phasiness (Frequency)')
%     ylabel('Phasiness')
%     xlabel('Frequency (Bin)')
%     subplot(414)
%     plot(ANGLE_DIFF')
%     title('All Differences')
%     xlabel('Time (Frame)')
%     ylabel('Frequency (Bin)')
%     t = strrep(General.Testname,'_','\_');
%     suptitle(t)
%     a = split(General.Testname,'/');
%     fname = ['Plots/Phasiness/' char(a{3}) '2No_weight.png'];
%     %     print(fname, '-dpng');
%     
%     
%     figure(3)
%     subplot(411)
%     plot(Ph_mad_t_mag)
%     title('Phasiness (Time)')
%     xlabel('Time (Frame)')
%     ylabel('Phasiness')
%     subplot(412)
%     plot(Ph_mad_t_diff_mag)
%     title('Phasiness 1st Diff (Time)')
%     xlabel('Time (Frame)')
%     ylabel('Phasiness 1st Diff')
%     
%     subplot(413)
%     semilogx(Ph_mad_f_mag)
%     title('Phasiness (Frequency)')
%     ylabel('Phasiness')
%     xlabel('Frequency (Bin)')
%     subplot(414)
%     plot(ANGLE_DIFF_LOUD_MAG')
%     title('All Differences')
%     xlabel('Time (Frame)')
%     ylabel('Frequency (Bin)')
%     t = strrep(General.Testname,'_','\_');
%     t = sprintf('%s \n Normalised Magnitude Weighting',t);
%     suptitle(t)
%     a = split(General.Testname,'/');
%     fname = ['Plots/Phasiness/' char(a{3}) '3mag_weight.png'];
%     %     print(fname, '-dpng');
%     
%     %     figure(4)
%     %     subplot(411)
%     %     plot(Ph_mad_t_pow)
%     %     title('Phasiness (Time)')
%     %     xlabel('Time (Frame)')
%     %     ylabel('Phasiness')
%     %     subplot(412)
%     %     plot(Ph_mad_t_diff_pow)
%     %     title('Phasiness 1st Diff (Time)')
%     %     xlabel('Time (Frame)')
%     %     ylabel('Phasiness 1st Diff')
%     %
%     %     subplot(413)
%     %     semilogx(Ph_mad_f_pow)
%     %     title('Phasiness (Frequency)')
%     %     ylabel('Phasiness')
%     %     xlabel('Frequency (Bin)')
%     %     subplot(414)
%     %     plot(ANGLE_DIFF_LOUD_POW')
%     %     title('All Differences')
%     %     xlabel('Time (Frame)')
%     %     ylabel('Frequency (Bin)')
%     %     t = strrep(General.Testname,'_','\_');
%     %     t = sprintf('%s \n Power Weighting',t);
%     %     suptitle(t)
%     %     a = split(General.Testname,'/');
%     %     fname = ['Plots/Phasiness/' char(a{3}) '4pow_weight.png'];
%     %     print(fname, '-dpng');
%     %
%     %     figure(5)
%     %     subplot(411)
%     %     plot(Ph_mad_t_fm)
%     %     title('Phasiness (Time)')
%     %     xlabel('Time (Frame)')
%     %     ylabel('Phasiness')
%     %     subplot(412)
%     %     plot(Ph_mad_t_diff_fm)
%     %     title('Phasiness 1st Diff (Time)')
%     %     xlabel('Time (Frame)')
%     %     ylabel('Phasiness 1st Diff')
%     %
%     %     subplot(413)
%     %     semilogx(Ph_mad_f_fm)
%     %     title('Phasiness (Frequency)')
%     %     ylabel('Phasiness')
%     %     xlabel('Frequency (Bin)')
%     %     subplot(414)
%     %     plot(ANGLE_DIFF_LOUD_FM')
%     %     title('All Differences')
%     %     xlabel('Time (Frame)')
%     %     ylabel('Frequency (Bin)')
%     %     t = strrep(General.Testname,'_','\_');
%     %     t = sprintf('%s \n Fletcher-Munson Weighting',t);
%     %     suptitle(t)
%     %     a = split(General.Testname,'/');
%     %     fname = ['Plots/Phasiness/' char(a{3}) '5FM_weight.png'];
%     %     print(fname, '-dpng');
%     
%     
%     
%     %     end
    
    
    
    
    
end



