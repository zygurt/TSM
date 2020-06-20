function [Comb_MOV,Comb_OMOV] = Cat_Unique(Comb_MOV,Comb_OMOV,MOVs,OMOV,feat)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
for n = 1:size(MOVs,2)
    if isempty(Comb_MOV)
        Comb_MOV = MOVs(:,1);
        Comb_OMOV = OMOV(:,1);
    else
        diff = Comb_MOV-repmat(MOVs(:,n),1,size(Comb_MOV,2));
        col_sum = sum(diff,1);
        if(sum(abs(col_sum)<1e-12)==0)
            %Concatenate the feature
            Comb_MOV = [Comb_MOV,MOVs(:,n)];
            Comb_OMOV = [Comb_OMOV,strcat(char(OMOV(n)),'_',feat)];
        end
    end
end
end

