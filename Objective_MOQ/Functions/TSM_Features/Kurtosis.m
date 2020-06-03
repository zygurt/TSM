function [ kurt ] = Kurtosis( X )
%[ kurt ] = Kurtosis( X )
%   Calculation of Kurtosis based on
  % H. Yu and T. Fingscheidt, "A Figure of Merit for Instrumental
  % Optimization of Noise Reduction Algorithms", in Proc. 5th Biennial
  % Workshop on DSP for In-Vehicle Systems, Kiel, Germany, Sep. 2011, pp.
  % 140-147.

K = size(X,1);
X_bar = (1/K)*sum(X,1);

num = 1/K*sum((X-X_bar).^4,1);

den = (1/K*sum((X-X_bar).^2,1)).^2;

kurt = num./den;



end
