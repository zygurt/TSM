%Combine Features

%The aim of this script is to combine features generated with
%different alignment methods

% %Combine Interpolate to Test with Anchor Test
% 
% load('Features/MOVs_20200620Interpolate_to_test.mat');
% M = MOVs;
% O = OMOV;
% load('Features/MOVs_20200620Framing_Test.mat');
% Comb_MOV = [M,MOVs(:,[6:9,13:17,22,23])];
% Comb_OMOV = [O,OMOV(:,[6:9,13:17,22,23])];
% MOVs = Comb_MOV;
% OMOV = Comb_OMOV;
% save('Features/MOVs_20200620Combine_ToTest_AnchorTest.mat','MOVs','OMOV','-v7')
% 
% %Combine all the different features
% 
% close all
% clear all
% clc
% 
% addpath('Functions/');
% Comb_MOV = [];%MOVs(:,1:5);
% Comb_OMOV = [];%OMOV(:,1:5);
% load('Features/MOVs_20200620Interpolate_to_test.mat');
% [Comb_MOV,Comb_OMOV] = Cat_Unique(Comb_MOV,Comb_OMOV,MOVs,OMOV,'ToTest');
% load('Features/MOVs_20200620Interpolate_to_ref.mat');
% [Comb_MOV,Comb_OMOV] = Cat_Unique(Comb_MOV,Comb_OMOV,MOVs,OMOV,'ToRef');
% load('Features/MOVs_20200620Interpolate_fd_up.mat');
% [Comb_MOV,Comb_OMOV] = Cat_Unique(Comb_MOV,Comb_OMOV,MOVs,OMOV,'Up');
% load('Features/MOVs_20200620Interpolate_fd_down.mat');
% [Comb_MOV,Comb_OMOV] = Cat_Unique(Comb_MOV,Comb_OMOV,MOVs,OMOV,'Down');
% load('Features/MOVs_20200620Framing_Test.mat');
% [Comb_MOV,Comb_OMOV] = Cat_Unique(Comb_MOV,Comb_OMOV,MOVs,OMOV,'AnchorTest');
% load('Features/MOVs_20200620Framing_Ref.mat');
% [Comb_MOV,Comb_OMOV] = Cat_Unique(Comb_MOV,Comb_OMOV,MOVs,OMOV,'AnchorRef');
% 
% MOVs = Comb_MOV;
% OMOV = Comb_OMOV;
% save('Features/MOVs_20200620Combine_Unique.mat','MOVs','OMOV','-v7')



%Create Incl Source features
% close all
% clear all
% clc
% 
% load('Features/MOVs_20200620Interpolate_to_test.mat');
% M = MOVs;
% load('Features/MOVs_Source_20200620Interpolate_to_test.mat');
% Comb_MOV = [MOVs;M];
% MOVs = Comb_MOV;
% save('Features/MOVs_20200620ToTest_Incl_Source.mat','MOVs','OMOV','-v7')
% 
% load('Features/MOVs_20200620Framing_Test.mat');
% M = MOVs;
% load('Features/MOVs_Source_20200620Framing_Test.mat');
% Comb_MOV = [MOVs;M];
% MOVs = Comb_MOV;
% save('Features/MOVs_20200620Framing_Test_Incl_Source.mat','MOVs','OMOV','-v7')

%Combine Interpolate to Test with Anchor Test Including Source

% load('Features/MOVs_20200620ToTest_Incl_Source.mat');
% M = MOVs;
% O = OMOV;
% load('Features/MOVs_20200620Framing_Test_Incl_Source.mat');
% Comb_MOV = [M,MOVs(:,[6:9,13:17,22,23])];
% Comb_OMOV = [O,OMOV(:,[6:9,13:17,22,23])];
% MOVs = Comb_MOV;
% OMOV = Comb_OMOV;
% save('Features/MOVs_20200620Combine_ToTest_AnchorTest_Incl_Source.mat','MOVs','OMOV','-v7')


%Combine Evaluation Features
close all
clear all
clc

load('Features/MOVs_Eval_20200622Interpolate_to_test.mat');
M = MOVs;
O = OMOV;
load('Features/MOVs_Eval_20200622Framing_Test.mat');
Comb_MOV = [M,MOVs(:,[6:9,13:17,22,23])];
Comb_OMOV = [O,OMOV(:,[6:9,13:17,22,23])];
MOVs = Comb_MOV;
OMOV = Comb_OMOV;
save('Features/MOVs_Eval_20200622Combine_ToTest_AnchorTest.mat','MOVs','OMOV','-v7')

