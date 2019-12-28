function [end_iter, w_estimate] = pan(Output, Dic, lambda, MAXITER)
%%
% This is a VANILA implementation of the following paper
% 
% The algorithm solves the inverse problem 
%   $y = Aw + \xi$
% ============= Inputs ===============
% y      : output, in the paper $y(t) = (x(t+\delta)-x(t))/\delta$;
% A      : dictionary matrix;
% lambda : the tradeoff parameter you should use; 
% basically, it is proportional to the invese of the variance, e.g. 1;
% MAXITER: maximum number of the iterations, e.g. 5
% ============= Reference =============
% W. Pan, Y. Yuan, J. Goncalves, and G.-B. Stan, 
% A Sparse Bayesian Approach to the Iden- tification of Nonlinear State-Space Systems,
% IEEE Transaction on Automatic Control, 2015 (to appear). arXiv:1408.3549
% http://arxiv.org/abs/1408.3549
% ============= Author =============
%  Wei Pan (w.pan11@imperial.ac.uk, panweihit@gmail.com)
% ============= Version =============
%   1.0 (Sep, 2012)
%%

delta = 1e-6;

[N,n]=size(Dic);
% initialisation of the variables
U=ones(n, MAXITER);
Gamma=zeros(n, MAXITER);
UU=zeros(n, MAXITER);
w_estimate=zeros(n, MAXITER);
WWW=ones(n, MAXITER);
fprintf(1, 'Finding a sparse feasible point using l1-norm heuristic ...\n');
end_iter = 0;
for iter=1:1:MAXITER
    end_iter = iter;
    fprintf('This is round %d \n', iter);
    cvx_begin quiet
    cvx_solver sedumi   %sdpt3
    variable W(n)
    minimize    (lambda*norm( U(:,iter).*W, 1 )+ 0.5*sum((Dic* W-Output).^2) )
    %                 subject to
    %                           W.^2-ones(101,1)<=0;
    cvx_end
    % find W(n)
    
    w_estimate(:,iter)=W;
    WWW(:,iter)=W;
    Gamma(:,iter)=U(:,iter).^-1.*W;
    Dic0=lambda*eye(N)+Dic*diag(Gamma(:,iter))*Dic';
    UU(:, iter)=diag(Dic'*(Dic0\Dic));
    U(:,iter+1)=abs(sqrt(UU(:, iter)));
    
    for i=1:n
        if   w_estimate(i,iter).^2/norm(w_estimate(:,iter))^2<delta
            w_estimate(i,iter)=0;
        end
    end
        % stopping criterion
    
    if(iter>1 && abs(max(w_estimate(:,iter)-w_estimate(:,iter-1)))<1e-3)
        w_estimate(:,end)=w_estimate(:,iter);
        break;
    end
end

