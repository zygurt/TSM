function [es,et,en] = Fuzzy_Feat(x_ref,x_test,General)
%UNTITLED Summary of this function goes here
%   Implementation follows that of [1], and uses median filtering code of [2].

%References:
%[1] Fierro, L., & Välimäki, V. Towards Objective Evaluation of Audio Time-Scale Modification Methods.
%[2] Damskägg, E. P., & Välimäki, V. (2017). Audio time stretching using fuzzy classification of spectral bins. Applied Sciences, 7(12), 1293.

%Code for easy debugging of function
% [x_test,General.fs] = audioread('../../../Subjective_Testing/Sets/1/Alto_Sax_06_HPTSM_78.21_per.wav');
% [x_ref,~] = audioread('../../../Subjective_Testing/Source/Solo/Alto_Sax_06.wav');
% General.N = 2048;

global debug_var

if debug_var
    disp('Fuzzy Features')
end

% if debug_var
%     figure
%     subplot(211)
%     plot(x_ref)
%     title('Reference')
%     subplot(212)
%     plot(x_test)
%     title('Test')
% end


[Es_ref,Et_ref,En_ref] = Fuzzy_Energies(x_ref,General);
[Es_test,Et_test,En_test] = Fuzzy_Energies(x_test,General);

%Interpolate lengths to the length of the test file
% Es_ref_interp = interp1(linspace(0,1,size(Es_ref,2)),Es_ref,linspace(0,1,size(Es_test,2)));
% Et_ref_interp = interp1(linspace(0,1,size(Et_ref,2)),Et_ref,linspace(0,1,size(Et_test,2)));
% En_ref_interp = interp1(linspace(0,1,size(En_ref,2)),En_ref,linspace(0,1,size(En_test,2)));

Es_ref_interp = interp1(linspace(0,1,size(Es_ref,2)),Es_ref,linspace(0,1,size(Es_ref,2)/General.TSM));
Et_ref_interp = interp1(linspace(0,1,size(Et_ref,2)),Et_ref,linspace(0,1,size(Et_ref,2)/General.TSM));
En_ref_interp = interp1(linspace(0,1,size(En_ref,2)),En_ref,linspace(0,1,size(En_ref,2)/General.TSM));

if size(Es_ref_interp,2)>size(Es_test,2)
    Es_ref_interp = Es_ref_interp(1:size(Es_test,2));
    Et_ref_interp = Et_ref_interp(1:size(Et_test,2));
    En_ref_interp = En_ref_interp(1:size(En_test,2));
else
    Es_test = Es_test(1:size(Es_ref_interp,2));
    Et_test = Et_test(1:size(Et_ref_interp,2));
    En_test = En_test(1:size(En_ref_interp,2));
end

%Find the best lead or lag for alignment of the signals
k_arr = zeros(round(0.1*size(Es_test,2)),1);
k_arr2 = zeros(round(0.1*size(Es_test,2)),1);
for k = 1:round(0.1*size(Es_test,2))
    k_arr(k) = sum(Es_test(1:end-k+1).*Es_ref_interp(k:end));
    k_arr2(k) = sum(Es_test(k:end).*Es_ref_interp(1:end-k+1));
end
%This could be optimised for speed by removing fuzzyness and
%summing after element-wise AND operations

%This priorities the maximum for forwards and backwards
%Find overall max in each correlation, then find the location
[max_correlation, loc] = max([k_arr; k_arr2]);
if loc < length(k_arr)
    km = min(find(k_arr==max_correlation))-1;
else
    km = -1*(min(find(k_arr2==max_correlation))-1);
end
if(isempty(km))
    km = 0;
end


%Adjust the length to for ideal correlation
% if km<0
%     Es_ref_interp = [zeros(1,abs(km)),Es_ref_interp(1:end-abs(km))];
%     Et_ref_interp = [zeros(1,abs(km)),Et_ref_interp(1:end-abs(km))];
%     En_ref_interp = [zeros(1,abs(km)),En_ref_interp(1:end-abs(km))];
% elseif km>0
%     Es_ref_interp = [Es_ref_interp(km:end),zeros(1,abs(km))];
%     Et_ref_interp = [Et_ref_interp(km:end),zeros(1,abs(km))];
%     En_ref_interp = [En_ref_interp(km:end),zeros(1,abs(km))];
% end


%Adjust for the correlation lead/lag by truncating beginning of late signal
% and end of early signal
if km<0
    Es_ref_interp = Es_ref_interp(1:end-abs(km));
    Es_test = Es_test((abs(km)+1):end);
    Et_ref_interp = Et_ref_interp(1:end-abs(km));
    Et_test = Et_test((abs(km)+1):end);
    En_ref_interp = En_ref_interp(1:end-abs(km));
    En_test = En_test((abs(km)+1):end);
elseif km>0
    Es_ref_interp = Es_ref_interp((abs(km)+1):end);
    Es_test = Es_test(1:end-abs(km));
    Et_ref_interp = En_ref_interp((abs(km)+1):end);
    Et_test = Et_test(1:end-abs(km));
    En_ref_interp = En_ref_interp((abs(km)+1):end);
    En_test = En_test(1:end-abs(km));
end

% Es_ref_interp = Es_ref_interp(abs(km)+1:end-abs(km));
% Es_test = Es_test(abs(km):(length(Es_ref_interp)+1));
% Et_ref_interp = Et_ref_interp(abs(km)+1:end-abs(km));
% Et_test = Et_test(abs(km)+1:end-abs(km));
% En_ref_interp = En_ref_interp(abs(km)+1:end-abs(km));
% En_test = En_test(abs(km)+1:end-abs(km));


if debug_var
    figure
    subplot(311)
    hold on
    plot(Es_ref_interp/max(abs(Es_ref_interp)))
    plot(Es_test/max(abs(Es_test)))
    hold off
    title('Es')
    legend('Reference','Test')
    
    subplot(312)
    hold on
    plot(Et_ref_interp/max(abs(Et_ref_interp)))
    plot(Et_test/max(abs(Et_test)))
    hold off
    title('Et')
    legend('Reference','Test')
    
    subplot(313)
    hold on
    plot(En_ref_interp/max(abs(En_ref_interp)))
    plot(En_test/max(abs(En_test)))
    hold off
    title('En')
    legend('Reference','Test')
    hold off
end

%Energy Deviation Curves
Ls_test = 10*log10(Es_test);
Ls_test_bar = mean(Ls_test);
Ls_ref = 10*log10(Es_ref_interp);
% Ls_ref(Ls_ref==-Inf) = 0;
Ls_ref_bar = mean(Ls_ref);
DeltaEs = Ls_test - Ls_ref - (Ls_test_bar - Ls_ref_bar);

Lt_test = 10*log10(Et_test);
Lt_test_bar = mean(Lt_test);
Lt_ref = 10*log10(Et_ref_interp);
% Lt_ref(Lt_ref==-Inf) = 0;
Lt_ref_bar = mean(Lt_ref);
DeltaEt = Lt_test - Lt_ref - (Lt_test_bar - Lt_ref_bar);

Ln_test = 10*log10(En_test);
Ln_test_bar = mean(Ln_test);
Ln_ref = 10*log10(En_ref_interp);
% Ln_ref(Ln_ref==-Inf) = 0;
Ln_ref_bar = mean(Ln_ref);
DeltaEn = Ln_test - Ln_ref - (Ln_test_bar - Ln_ref_bar);

if debug_var
    figure
    subplot(311)
    plot(DeltaEs)
    title('\Delta Es')
    subplot(312)
    plot(DeltaEt)
    title('\Delta Et')
    subplot(313)
    plot(DeltaEn)
    title('\Delta En')
end


%Mean Squared Error of each fuzzy class
es = mean(DeltaEs(:,5:end-10).^2);
et = mean(DeltaEt(:,5:end-10).^2);
en = mean(DeltaEn(:,5:end-10).^2);


%[1] Now does linear regression using the es, et and en to MOS scores of [2]
%We will just return the features for use with the Neural Net.


end

