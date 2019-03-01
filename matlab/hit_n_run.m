function [x]=hit_n_run(x,A,T)
% function [x]=hit_n_run(x,A,T)
%
%   Returns a random point using the hit and run algorithm from the
% convex body defined by Ax>=0 and ||x||<=1.
%   The random walks begins from the point x which is assumed
% to be an internal point (i.e. satisfies the constraints Ax>=0
% and ||x||<=1. The number of steps the algorithms will perform
% is T.
%
% Inputs:
%   x - A starting point for the random walk. Must be internal
%       point.
%
%   A - A set of constraints defining the convex body: Ax>=0
%
%   T - Number of steps to perform.
%
% Outputs:
%   x - A random point in the convex body {x: Ax>=0, ||x||<=1}.
%

% Copyright (C) 2005 Ran Gilad-Bachrach
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
%
% Change Log
% ----------
% Version 1.0 - May 4 2005


dim=length(x);
x=x(:);
u=randn(T,dim); % at step t the algorithm will pick a random point
                % on the line through x and x+u(t,:)
Au=u*A';
nu=sum(u.^2,2);
l=rand(T,1);

for t=1:T
  Ax=A*x;
  ratio=-Ax./Au(t,:)';
  I=(Au(t,:)>0);
  mn=max([ratio(I);-realmax]);
  I=(Au(t,:)<0);
  mx=min([ratio(I);realmax]);

  disc=(x'*u(t,:)')^2-nu(t)*(norm(x)^2-1);
  if (disc<0)
    warning(['negative disc ' num2str(disc) '. Probably x is not a ' ...
                   'feasable point.']);

    disc=0;
  end
  hl=(-(x'*u(t,:)')+sqrt(disc))/nu(t);
  ll=(-(x'*u(t,:)')-sqrt(disc))/nu(t);

  xx=min(hl,mx);
  nn=max(ll,mn);
  x=x+u(t,:)'*(nn+l(t)*(xx-nn));
end