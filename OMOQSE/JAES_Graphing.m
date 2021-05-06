% JAES additional graphs script
close all
clear all
clc

load('DE_SE_RESULTS.mat');

%Get difference between DE and SE results
%ignore TSM<0.25 and TSM==1;
diff = results_OMOQDE(:,[2:12,14:end],:)-results_OMOQSE(:,[2:12,14:end],:);

abs_diff = abs(diff);

%Aim is to show performance for each file in evaluation set per method
% results(file,TSM,method)

MAD = squeeze(mean(abs_diff,2));

MD = squeeze(mean(diff,2));

Methods = {'DIPL','ESOLA','EL','FESOLA','FuzzyPV','HPTSM','IPL','NMFTSM','PV','PIPL','PSPL','SPL','WSOLA','SuTVS','uTVS'};

Filenames = {'Alto Sax 15';'Ardour 2';'Brass and perc 9';'Child 4';'Female 2';'Female 4';'Jazz 3';'Male 16';'Male 22';'Male 6';'Mexican Flute 02';'Oboe piano 1';'Ocarina 02';'Rock 4';'Saxophones 6';'Solo flute 2';'Synth Bass 2';'Triangle 02';'Woodwinds 4';'You mean this one'};
TSM = [0.2257,0.2635,0.3268,0.4444,0.5620,0.6631,0.7641,0.8008,0.8375,0.8742,0.9109,0.9555,1,1.1205,1.241,1.3477,1.4543,1.6272,1.8042,2.1632]; %All Eval Values


figure
surf(MAD)
title('MAD DE-SE MOS')
colorbar
xticks((1:15)+0.5);
xticklabels(Methods);
xtickangle(45);
yticks((1:20)+0.5);
yticklabels(Filenames);
axis tight
view(-180,90);

figure
surf(MD);
title('MD DE-SE MOS');
colorbar;
xticks((1:15)+0.5);
xticklabels(Methods);
xtickangle(45);
yticks((1:20)+0.5);
yticklabels(Filenames);
axis tight
view(-180,90);
