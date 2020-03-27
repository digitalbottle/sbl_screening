clc;clear;
addpath(genpath('.'));
output_file = './data/result';
if exist(output_file,'dir') == 0
    mkdir(output_file);
end
%% Load dataset
train_image = loadMNISTImages('./data/mnist/train-images-idx3-ubyte');
train_label = loadMNISTLabels('./data/mnist/train-labels-idx1-ubyte');
test_image = loadMNISTImages('./data/mnist/t10k-images-idx3-ubyte');
test_label = loadMNISTLabels('./data/mnist/t10k-labels-idx1-ubyte');
%% Sparse representation: test data [i] = Sparse_representation * train data
% dict
dict_img = train_image(:,:);
A = dict_img;
A = A .* (1 ./ sqrt(sum(A .* A, 1)));
% target
target_img = test_image(:, :);
B = target_img;
B = B .* (1 ./ sqrt(sum(B .* B, 1)));
%-----------------------------------
% Lasso
%-----------------------------------
fprintf("Solving Lasso\n");
% for i=1:size(B, 1)
%     [w_lasso, lasso_res] = lasso(A,B(:, i));
% end
i = 1;
[w_lasso, lasso_res] = lasso(A,B(:, i));
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
saveas(gcf, [output_file '/Lasso.png']);
close(gcf)
close 
%% Borda Count