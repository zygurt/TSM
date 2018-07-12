%Stereo TSM script
close all
clear all
clc
tic
% pathInput = 'Confetti_Source/';
% pathOutput = 'Confetti_Out/';
pathInput = 'AudioIn/';
pathOutput = 'AudioOut/';
%Modify this path to the location of the MATLAB TSM Toolbox on your system
addpath('/home/tim/Documents/MATLAB/MATLAB_TSM-Toolbox_1.0');
addpath('../Frequency_Domain/');
addpath('../Time_Domain/');
addpath('../Functions');
d = dir(pathInput);
N=2048; %The larger this value, the smoother the averaged phase will be.
N_f = 2048; %Feature frame size
TSM = [0.3838 0.4427 0.5383 0.6524 0.7821 0.8258 0.9612 1.2570 1.4692 1.6961 1.8412];
mono_mode = 1;
for n = 3:numel(d)
    [x,fsAudio] = audioread([pathInput d(n).name]);
    disp(d(n).name);
    original(n-2).filename=[d(n).name];
    original(n-2).Fs = fsAudio;
    num_chan = size(x,2);
    %If the input signal is mono, create a stereo version
    if num_chan == 1
        switch mono_mode
            case 1
                %Multi_mono L=R
                x = [x , x];
                original(n-2).filename = ['centre_mono_' original(n-2).filename];
            case 2
                % Mono signal L/0
                x = [x , zeros(size(x,1), size(x,2))];
                original(n-2).filename = ['left_mono_' original(n-2).filename];
            case 3
                %Mono signal 0/R
                x = [zeros(size(x,1), size(x,2)), x];
                original(n-2).filename = ['right_mono_' original(n-2).filename];
            otherwise
                %something else
        end
        num_chan = 2;
    end
    %Store the stereo phase coherence, balance and width for the original signal
    [ original(n-2).C(1,:), original(n-2).C_o  ] = st_phase_coherence( x, N );
    [ original(n-2).B(1,:), original(n-2).B_o ] = st_balance(x, N);
    
    %Generate the Sum and Difference
    x_ms = zeros(length(x), 2);
    x_ms(:,1) = x(:,1)+x(:,2);
    x_ms(:,2) = x(:,1)-x(:,2);
    
    for k = 1:length(TSM)
        results(n-2,k).filename=d(n).name;
        results(n-2,k).Fs = fsAudio;
        results(n-2,k).TSM = TSM(k);
        
        yHP_MS =[];
        yWSOLA_MS = [];
        
        %Phase Vocoder methods
        fprintf('Naive, ');
        y_Naive = PV(x, N, TSM(k));
        fprintf('Bonada, ');
        y_Bonada = PV_Bonada(x, N, TSM(k));
        fprintf('Altoe, ');
        y_Altoe = PV_Altoe(x, N, TSM(k));
        fprintf('PV_MS_File, ');
        y_MS_File = PV_MS_File(x, N, TSM(k));
        fprintf('PV_MS_Frame, ');
        y_MS_Frame = PV_MS_Frame(x, N, TSM(k));
        
        %WSOLA
        fprintf('Naive WSOLA, ');
        yWSOLA = wsolaTSM(x, 1/TSM(k));
        fprintf('MS_LR WSOLA, ');
        yWSOLA_MS_temp = wsolaTSM(x_ms, 1/TSM(k));
        %Convert Back
        yWSOLA_MS(:,1) = (yWSOLA_MS_temp(:,1)+yWSOLA_MS_temp(:,2))/2;
        yWSOLA_MS(:,2) = (yWSOLA_MS_temp(:,1)-yWSOLA_MS_temp(:,2))/2;
        
        fprintf('Driedger HP, ');
        yHP = hpTSM(x, 1/TSM(k));
        fprintf('MS HP\n');
        yHP_MStemp = hpTSM(x_ms, 1/TSM(k));
        %Convert Back
        yHP_MS(:,1) = (yHP_MStemp(:,1)+yHP_MStemp(:,2))/2;
        yHP_MS(:,2) = (yHP_MStemp(:,1)-yHP_MStemp(:,2))/2;
        
        
        %         Ensure that all of the lengths are the same.
        if(length(yHP_MS)<length(y_Naive))
            disp('Driedger Shorter')
            yWSOLA = [ yWSOLA ; zeros(length(y_Naive)-length(yWSOLA), 2)];
            yWSOLA_MS = [ yWSOLA_MS ; zeros(length(y_Naive)-length(yWSOLA_MS), 2)];
            yHP = [ yHP ; zeros(length(y_Naive)-length(yHP), 2)];
            yHP_MS = [ yHP_MS ; zeros(length(y_Naive)-length(yHP_MS), 2)];
        else
            disp('Driedger Longer')
            y_Naive = [ y_Naive ; zeros(length(yWSOLA)-length(y_Naive), 2)];
            y_Bonada = [ y_Bonada ; zeros(length(yWSOLA)-length(y_Bonada), 2)];
            y_Altoe = [ y_Altoe ; zeros(length(yWSOLA)-length(y_Altoe), 2)];
            y_MS_File = [ y_MS_File ; zeros(length(yWSOLA)-length(y_MS_File), 2)];
            y_MS_Frame = [ y_MS_Frame ; zeros(length(yWSOLA)-length(y_MS_Frame), 2)];
        end

        fprintf('Audio Write, ');
        audiowrite([pathOutput sprintf('%0.2f',TSM(k)) '_Naive_' original(n-2).filename(1:end-4) '.wav'], y_Naive/max(max(abs(y_Naive))), fsAudio);
        audiowrite([pathOutput sprintf('%0.2f',TSM(k)) '_Bonada_' original(n-2).filename(1:end-4) '.wav'], y_Bonada/max(max(abs(y_Bonada))), fsAudio);
        audiowrite([pathOutput sprintf('%0.2f',TSM(k)) '_Alloe_' original(n-2).filename(1:end-4) '.wav'], y_Altoe/max(max(abs(y_Altoe))), fsAudio);
        audiowrite([pathOutput sprintf('%0.2f',TSM(k)) '_MS_File_' original(n-2).filename(1:end-4) '.wav'], y_MS_File/max(max(abs(y_MS_File))), fsAudio);
        audiowrite([pathOutput sprintf('%0.2f',TSM(k)) '_MS_Frame_' original(n-2).filename(1:end-4) '.wav'], y_MS_Frame/max(max(abs(y_MS_Frame))), fsAudio);
        audiowrite([pathOutput sprintf('%0.2f',TSM(k)) '_WSOLA_' original(n-2).filename(1:end-4) '.wav'], yWSOLA/max(max(abs(yWSOLA))), fsAudio);
        audiowrite([pathOutput sprintf('%0.2f',TSM(k)) '_WSOLA_MS_' original(n-2).filename(1:end-4) '.wav'], yWSOLA_MS/max(max(abs(yWSOLA_MS))), fsAudio);
        audiowrite([pathOutput sprintf('%0.2f',TSM(k)) '_HP_' original(n-2).filename(1:end-4) '.wav'], yHP/max(max(abs(yHP))), fsAudio);
        audiowrite([pathOutput sprintf('%0.2f',TSM(k)) '_HP_MS_' original(n-2).filename(1:end-4) '.wav'], yHP_MS/max(max(abs(yHP_MS))), fsAudio);
        
        fprintf('Calculate Features, ');
        [ results(n-2,k).C(1,:), results(n-2,k).C_o(1,1) ] = st_phase_coherence( y_Naive, N_f );
        [ results(n-2,k).B(1,:), results(n-2,k).B_o(1,1) ] = st_balance(y_Naive, N_f);
        
        [ results(n-2,k).C(2,:), results(n-2,k).C_o(2,1)  ] = st_phase_coherence( y_Bonada, N_f );
        [ results(n-2,k).B(2,:), results(n-2,k).B_o(2,1) ] = st_balance(y_Bonada, N_f);
        
        [ results(n-2,k).C(3,:), results(n-2,k).C_o(3,1)  ] = st_phase_coherence( y_Altoe, N_f );
        [ results(n-2,k).B(3,:), results(n-2,k).B_o(3,1) ] = st_balance(y_Altoe, N_f);
        
        [ results(n-2,k).C(4,:), results(n-2,k).C_o(4,1) ] = st_phase_coherence( y_MS_File, N_f );
        [ results(n-2,k).B(4,:), results(n-2,k).B_o(4,1) ] = st_balance(y_MS_File, N_f);
        
        [ results(n-2,k).C(5,:), results(n-2,k).C_o(5,1) ] = st_phase_coherence( y_MS_Frame, N_f );
        [ results(n-2,k).B(5,:), results(n-2,k).B_o(5,1) ] = st_balance(y_MS_Frame, N_f);
        
        [ results(n-2,k).C(6,:), results(n-2,k).C_o(6,1) ] = st_phase_coherence( yWSOLA, N_f );
        [ results(n-2,k).B(6,:), results(n-2,k).B_o(6,1) ] = st_balance(yWSOLA, N_f);
        
        [ results(n-2,k).C(7,:), results(n-2,k).C_o(7,1) ] = st_phase_coherence( yWSOLA_MS, N_f );
        [ results(n-2,k).B(7,:), results(n-2,k).B_o(7,1) ] = st_balance(yWSOLA_MS, N_f);
        
        [ results(n-2,k).C(8,:), results(n-2,k).C_o(8,1) ] = st_phase_coherence( yHP, N_f );
        [ results(n-2,k).B(8,:), results(n-2,k).B_o(8,1) ] = st_balance(yHP, N_f);
        
        [ results(n-2,k).C(9,:), results(n-2,k).C_o(9,1) ] = st_phase_coherence( yHP_MS, N_f );
        [ results(n-2,k).B(9,:), results(n-2,k).B_o(9,1) ] = st_balance(yHP_MS, N_f);
        
        fprintf('Stereo Results. \n');
        fprintf('TSM = %g complete\n', TSM(k));
    end
end
save('subjective_files_results.mat','results');
save('subjective_files_original.mat','original');