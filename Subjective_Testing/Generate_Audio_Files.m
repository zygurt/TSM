close all
clear all
clc

addpath('../Functions');
addpath('../Batch');
addpath('../Frequency_Domain');
addpath('../Time_Domain');
addpath('../../../Documents/MATLAB/MATLAB_TSM-Toolbox_1.0');
% addpath('../External/NMFTSM');

% pathInput = 'AudioIn/';
pathInput = 'Source/Objective';
pathOutput = 'AudioOut/';

%Create a string containing all the folders
allSubFolders = genpath(pathInput);
% Parse into a cell array.
remain = allSubFolders;
listOfFolderNames = {};
%Break the string into individual folder names
while true
    [singleSubFolder, remain] = strtok(remain, ':');
    if isempty(singleSubFolder)
        break;
    end
    listOfFolderNames = [listOfFolderNames singleSubFolder];
end
numberOfFolders = length(listOfFolderNames);

%Iterate through the folders and put the location into the filelist struct
%Also set the remaining fields to zero.
% a = 0.2;
% b = 1.5;
% n = 5;
% TSM = a + (b-a).*rand(1,n)
% hist(TSM)
% pause()
% TSM = [0.3268,0.5620,0.7641,0.8375,0.9109,1,1.241,1.4543]; %Eval Values
% TSM = [0.3838 0.4427 0.5383 0.6524 0.7821 0.8258 0.9961 1.381 1.667 1.924]; %Training Values

tic

for k = 1:numberOfFolders
    directory = dir(listOfFolderNames{k});
    for l = 1:numel(directory)
        if directory(l).isdir == 0
            location = sprintf('%s/%s',listOfFolderNames{k},directory(l).name);
            [x,fs] = audioread(location);
            x = sum(x,2);
            x = x/max(abs(x));
            
            out_filename = sprintf('%sPV/%s_PV',pathOutput,directory(l).name(1:end-4));
            [ ~ ] = PV_batch(x, 2048, TSM, fs, out_filename);

            out_filename = sprintf('%sIPL/%s_IPL',pathOutput,directory(l).name(1:end-4));
            [ ~ ] = PL_PV_batch( x, 2048, TSM, 1, fs, out_filename );
            
            out_filename = sprintf('%sSPL/%s_SPL',pathOutput,directory(l).name(1:end-4));
            [ ~ ] = PL_PV_batch( x, 2048, TSM, 2, fs, out_filename );
            
            out_filename = sprintf('%sPhavorit_IPL/%s_Phavorit_IPL',pathOutput,directory(l).name(1:end-4));
            [ ~ ] = Phavorit_PV_batch( x, 2048, TSM, 0, fs, out_filename );
            
            out_filename = sprintf('%sPhavorit_SPL/%s_Phavorit_SPL',pathOutput,directory(l).name(1:end-4));
            [ ~ ] = Phavorit_PV_batch( x, 2048, TSM, 1, fs, out_filename );
            
%Using Driedger's TSM library for this instead
%             out_filename = sprintf('%sWSOLA/%s_WSOLA',pathOutput,directory(l).name(1:end-4));
%             [ ~ ] = WSOLA_batch( x, 2048, TSM, fs, out_filename );

            out_filename = sprintf('%sFESOLA/%s_FESOLA',pathOutput,directory(l).name(1:end-4));
            [ ~ ] = FESOLA_batch( x, 1024, TSM, fs, out_filename );

             out_filename = sprintf('%suTVS/%s_uTVS',pathOutput,directory(l).name(1:end-4));
             [ ~ ] = uTVS_batch( x, fs, TSM, out_filename );
%             
            for t = 1:length(TSM)
%                 %Add the non-batch methods in here.
%                 %Audio write happens in the batch methods
%                 fprintf('%s, SliceTSM, %g%%\n',location, TSM(t)*100);
%                 [ y_Slice ] = SliceTSM( x, fs, TSM(t) );
%                 out_filename = sprintf('%sSlice/%s_Slice_%g_per.wav',pathOutput,directory(l).name(1:end-4),TSM(t)*100);
%                 audiowrite(out_filename,y_Slice/max(abs(y_Slice)),fs);

                fprintf('%s, ESOLA, %g%%\n',location, TSM(t)*100);
                [ y_ESOLA ] = ESOLA( x, 1024, TSM(t), fs );
                out_filename = sprintf('%sESOLA/%s_ESOLA_%g_per.wav',pathOutput,directory(l).name(1:end-4),TSM(t)*100);
                audiowrite(out_filename,y_ESOLA/max(abs(y_ESOLA)),fs);

                fprintf('%s, WSOLA, %g%%\n',location, TSM(t)*100);
                y_WSOLA = wsolaTSM(x,1/TSM(t));
                out_filename = sprintf('%sWSOLA/%s_WSOLA_%g_per.wav',pathOutput,directory(l).name(1:end-4),TSM(t)*100);
                audiowrite(out_filename,y_WSOLA/max(abs(y_WSOLA)),fs);
                
                fprintf('%s, HPTSM, %g%%\n',location, TSM(t)*100);
                y_HPTSM = hpTSM(x,1/TSM(t));
                out_filename = sprintf('%sHPTSM/%s_HPTSM_%g_per.wav',pathOutput,directory(l).name(1:end-4),TSM(t)*100);
                audiowrite(out_filename,y_HPTSM/max(abs(y_HPTSM)),fs);

                %If it crashes here asking for curl, sudo apt install curl
                fprintf('%s, Elastique, %g%%\n',location, TSM(t)*100);
                parameter.pitchShift = 0;
                parameter.formantShift = 0;
                parameter.fsAudio = fs;
                [y_Elastique,~] = elastiqueTSM(x,1/TSM(t),parameter);
                out_filename = sprintf('%sElastique/%s_Elastique_%g_per.wav',pathOutput,directory(l).name(1:end-4),TSM(t)*100);
                audiowrite(out_filename,y_Elastique/max(abs(y_Elastique)),fs);
            end
        end
    end
end
t = toc
