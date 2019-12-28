clc;clear;
addpath(genpath('..'));
%% prepare dataset
[dict_img, dict_pt, target_img, target_pts] = load_data('data');
for i=1:length(target_img)
    temp = target_img{i};
    target_img{i} = temp(:);
end
Y = target_img{1};
y = Y(:);
dict_img = dict_img(:,:);
A = dict_img';
% Normalization
y = y / sqrt(sum(y .* y));
A = A .* (1 ./ sqrt(sum(A .* A, 1)));
%% ordinary least squares solution
%-----------------------------------
[w_ls, ~, MSE_ls] = lscov(A,y);
%-----------------------------------
%% Lasso
fprintf("Solving Lasso\n");
%-----------------------------------
[w_lasso, lasso_res] = lasso(A,y);
%-----------------------------------
figure('Name','Lasso');
subplot(3,1,1);
plot(lasso_res.DF, 'LineWidth',2);
title('non-zero parameter number')
subplot(3,1,2);
plot(lasso_res.Lambda, 'LineWidth',2);
title('Lambda')
subplot(3,1,3);
plot(lasso_res.MSE, 'LineWidth',2);
title('MSE')
%% pan method
lambda_max = max(y' * A);
ratio = 0.7;
lambda = lambda_max*ratio;
fprintf("Solving pan\n");
MAXITER = 100;
tic;
[end_iter_pan, w_pan] =  pan(y, A, lambda,MAXITER);
toc;
figure('Name','pan');
plot(sum(w_pan(:, 1:end_iter_pan)>0), 'LineWidth',2);
%% pan revised method
fprintf("Solving pan revised\n");
MAXITER = 100;
tic;
[screen_ratio, end_iter_pan_re, w_screen] = pan_revised(y, A, lambda,MAXITER);
toc;
figure('Name','pan revised');
plot(sum(w_screen(:, 1:end_iter_pan_re)>0), 'LineWidth',2);
% compare = [w_pan(:,end), w_screen(:,end), w_lasso, w_ls]
fprintf('max |w_screen-w_pan| =%f\n',max(abs(w_pan(:,end)-w_screen(:,end))));
%% PLOT
% stem(w_pan(:,end));
% stem(w_screen(:,end));
% imshow(reshape(A(:,100)./ max(A(:,1), [], 'all'), [28, 28]));