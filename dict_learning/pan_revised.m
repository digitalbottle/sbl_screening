function [screen_ratio, end_iter, w_estimate] = pan_revised(Output, Dic, lambda, MAXITER)
%%
% The algorithm solves the inverse problem 
%   $y = Aw + \xi$
% ============= Inputs ===============
% y      : output, in the paper $y(t) = (x(t+\delta)-x(t))/\delta$;
% A      : dictionary matrix;
% lambda : the tradeoff parameter you should use, 
% basically, it is proportional to the invese of the variance, e.g. 1;
% MAXITER: maximum number of the iterations, e.g. 5
%%
    screen_ratio = 0;
    delta = 1e-6;
    [N,n]=size(Dic);
    % initialisation of the variables
    U=ones(n, MAXITER);
    Gamma=zeros(n, MAXITER);
    UU=zeros(n, MAXITER);
    w_estimate=zeros(n, MAXITER);
    WWW=ones(n, MAXITER);
%     fprintf(1, 'Finding a sparse feasible point using l1-norm heuristic ...\n');
    Dic_bak = Dic;
    end_iter = 0;
    for iter=1:1:MAXITER
        end_iter = iter;
%         fprintf('This is round %d \n', iter);
        u = U(:,iter);
        [v,lam_max] = screen(Dic,Output,lambda,u); 
        Dic(:,v==1) = [];
        % screening
%         fprintf('screen ratio for round %d = %f%%\n', iter, (1-size(Dic,2)/n )*100);
        screen_ratio = 1-size(Dic,2)/n;
        % solve the reduced problem
        cvx_begin quiet
        cvx_solver sedumi   %sdpt3
        variable W(size(Dic,2))
        minimize    (lambda*norm( U(v~=1,iter).*W, 1 )+ 0.5*sum((Dic* W-Output).^2) )
        %                 subject to
        %                           W.^2-ones(101,1)<=0;
        cvx_end
        % recover W and Dic
        tol = 1e-8;
        W = recover(v,W,tol);
        Dic = Dic_bak;
        % recover finished
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
            w_estimate(:,end)=w_estimate(:,iter-1);
            w_estimate(w_estimate(:,end) < 1e-6, end) = 0;
            break;
        end
    end
%     fprintf('lambda_max=%f\n',lam_max);
end