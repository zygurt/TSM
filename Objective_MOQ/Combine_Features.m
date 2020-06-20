%Combine Features

%The aim of this script is to combine features generated with
%different alignment methods

%Combine Interpolate to Test with Anchor Test

load('Features/MOVs_20200620Interpolate_to_test.mat');
M = MOVs;
O = OMOV;
load('Features/MOVs_20200620Framing_Test.mat');
Comb_MOV = [M,MOVs(:,[6:9,13:17,22,23])];
Comb_OMOV = [O,OMOV(:,[6:9,13:17,22,23])];
MOVs = Comb_MOV;
OMOV = Comb_OMOV;
save('Features/MOVs_20200620Combine_ToTest_AnchorTest.mat','MOVs','OMOV','-v7')

%Combine all the different features

close all
clear all
clc

addpath('Functions/');
Comb_MOV = [];%MOVs(:,1:5);
Comb_OMOV = [];%OMOV(:,1:5);
load('Features/MOVs_20200620Interpolate_to_test.mat');
[Comb_MOV,Comb_OMOV] = Cat_Unique(Comb_MOV,Comb_OMOV,MOVs,OMOV,'ToTest');
load('Features/MOVs_20200620Interpolate_to_ref.mat');
[Comb_MOV,Comb_OMOV] = Cat_Unique(Comb_MOV,Comb_OMOV,MOVs,OMOV,'ToRef');
load('Features/MOVs_20200620Interpolate_fd_up.mat');
[Comb_MOV,Comb_OMOV] = Cat_Unique(Comb_MOV,Comb_OMOV,MOVs,OMOV,'ToTest');
load('Features/MOVs_20200620Interpolate_fd_down.mat');
[Comb_MOV,Comb_OMOV] = Cat_Unique(Comb_MOV,Comb_OMOV,MOVs,OMOV,'ToRef');
load('Features/MOVs_20200620Framing_Test.mat');
[Comb_MOV,Comb_OMOV] = Cat_Unique(Comb_MOV,Comb_OMOV,MOVs,OMOV,'AnchorTest');
% load('Features/MOVs_20200620Framing_Ref.mat');
% [Comb_MOV,Comb_OMOV] = Cat_Unique(Comb_MOV,Comb_OMOV,MOVs,OMOV,'AnchorRef');


MOVs = Comb_MOV;
OMOV = Comb_OMOV;
save('Features/MOVs_20200620Combine_Unique.mat','MOVs','OMOV','-v7')







