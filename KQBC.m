function [selection, ret_w] = KQBC(X_train, Y_train, T, kernel, queries, ...
                                           parameter1,parameter2)
% function [selection, coefs, errors] = KQBC(X_train, Y_train, queries, ...
%                                            T,kernel, parameter1,parameter2)
%
% Runs the kernelized version of the Query By Committee (QBC) algorithm.
%
% Inputs:
%   X_train - Training instances.
%   Y_train - Training labels (values +1 and -1).
%   T       - Number of random walk steps to make when selecting a
%             random hypothesis.
%   kernel  - Type of kernel to be used. Possible values are
%             'Linear', 'Poly' and 'Gauss'.
%   queries  - Number of queries indicate for the loop to break.
%   parameter1, parameter2 -
%             parameters for the chosen kernel.
%
% Outputs:
%   selection - The instances for which the algorithm queried for
%               label.
%   coefs     - The coefficients of the hypotheses used in each of
%               the steps of the algorithm.
%   errors    - The training error at each step.


% Copyright (C) 2005 Ran Gilad-Bachrach
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
%
% Change Log
% ----------
% Version 1.0 - May 4 2005



tol = 1e-10;   %tolerance for the factor function


  switch kernel
    case 'Linear'
      K = X_train*X_train';
    case 'Poly'
      K = (X_train*X_train' + parameter1).^parameter2;
    case 'Gauss'
      nrm = sum(X_train.^2,2);
      K = exp(-(repmat(nrm,1,length(Y_train))-2*X_train*X_train'+...
		repmat(nrm',length(Y_train),1))/(2*parameter1.^2));
    otherwise
      disp(['unknown kernel']);
      return;
  end

  coefs=[];
  errors=[];
  ret_w=[];

  samp_num = length(Y_train);


  selection = [1];    % initialization: select the first sample point
  selected = 1;

  coef = zeros(length(Y_train),1);
  coef(1) = Y_train(1)/sqrt(K(1,1));
  preds = K*coef;
  errate = sum(Y_train.*preds<=0);

  for ii= 2:samp_num;
    if length(selection) >= queries
        break;
    end
    extension = [selection ii];
    % the following steps correspond to finding an orthonormal
    % basis to the sample points. We generate a matrix A such that
    % A'*K(extension,extension)*A is the identity matrix with rank
    % equals to the rank of K.
    [u,s] = schur(K(extension,extension));
    s = diag(s);
    I = (s>tol);
    A = u(:,I)*diag(s(I).^-0.5);

    restri = diag(Y_train(selection))*K(selection,extension)*A;

    co1 = pinv(A)*coef(extension);


    co2 = hit_n_run(co1,restri,T);
    co1 = hit_n_run(co2,restri,T);

    pred1 = K(ii,extension)*(A*co1);
    pred2 = K(ii,extension)*(A*co2);

    if (pred1*pred2<=0)   % the classifiers disagree
      selection = extension;
      if (Y_train(ii)*pred1>=0)
	    coef(extension) = A*co1;
        ret_w = co1;
      else
	    coef(extension) = A*co2;
        ret_w = co2;
      end
      coefs = [coefs,coef];
      errors = [errors;sum(Y_train.*(K*coef)<=0)/length(Y_train)];
      %disp([ii,length(selection),errors(end)]);
    end
  end
%end

