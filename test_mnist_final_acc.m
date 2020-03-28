clc;clear;
addpath(genpath('.'));
output_file = './data/result/acc/';
if exist(output_file,'dir') == 0
    mkdir(output_file);
end
%% Load dataset
train_image_base = loadMNISTImages('./data/mnist/train-images-idx3-ubyte');
train_label_base = loadMNISTLabels('./data/mnist/train-labels-idx1-ubyte');
test_image_base = loadMNISTImages('./data/mnist/t10k-images-idx3-ubyte');
test_label_base = loadMNISTLabels('./data/mnist/t10k-labels-idx1-ubyte');
%% Accuracy -- lambda / lambda_max
exp_time = 3;
lambda_num = 15;
dict_set_num = 100;
test_set_num = 100;
% record
lasso_acc = zeros(exp_time, lambda_num);
lasso_times = zeros(exp_time, 1);
lasso_lambda = zeros(exp_time, lambda_num);
Pan_acc = zeros(exp_time, lambda_num);
Pan_times = zeros(exp_time, 1);
Pan_lambda = zeros(exp_time, lambda_num);
Pan_revised_acc = zeros(exp_time, lambda_num);
Pan_revised_times = zeros(exp_time, 1);
Pan_revised_lambda = zeros(exp_time, lambda_num);
% main loop
for exp_t=1:exp_time
    train_num_list = randperm(size(train_image_base, 2), dict_set_num);
    test_num_list = randperm(size(test_image_base, 2), test_set_num);
    train_image = train_image_base(:, train_num_list);
    train_label = train_label_base(train_num_list);
    test_image = test_image_base(:, test_num_list);
    test_label = test_label_base(test_num_list);
    %% Sparse representation: test data [i] = Sparse_representation * train data
    % dict
    A = train_image;
    A = A .* (1 ./ sqrt(sum(A .* A, 1)));
    % target
    B = test_image;
    B = B .* (1 ./ sqrt(sum(B .* B, 1)));
    %-----------------------------------
    % Lasso
    %-----------------------------------
    lasso_result = zeros(size(test_image, 2), lambda_num);
    t1=clock;
    for i=1:size(test_image, 2)
        [w_lasso, lasso_res] = lasso(A, B(:, i), 'NumLambda', lambda_num+1);
        %% Correlation classification
        score = Correlation(w_lasso(:,1:end-1), train_label);
        [~, argmax] = max(score, [], 1);
        lasso_result(i, :) = argmax - 1;
    end
    t2=clock;
    lasso_acc(exp_t, :) = sum(lasso_result == test_label, 1) ./ size(test_image, 2);
    fprintf('Lasso Test Exp %d/%d, time=%.2fs, acc=%.2f\n', ...
                 exp_t, exp_time, etime(t2,t1), mean(lasso_acc(exp_t, :)));
end
lambda_ratios = linspace(0.01, 0.99, size(lasso_acc, 2));
h_fig = figure('Name', 'Accuracy', 'Visible', 'off');
lasso_acc_mean = mean(lasso_acc, 1);
lasso_acc_std = std(lasso_acc, 1, 1);
errorbar(lambda_ratios, lasso_acc_mean, lasso_acc_std, '-s', 'LineWidth', 2, 'Color', 'r', ...
                  'MarkerSize',10, 'MarkerEdgeColor','r','MarkerFaceColor','w')
title('Accuracy -- \lambda / \lambda_{max}')
xlabel('\lambda / \lambda_{max}')
ylabel('Accuracy')
legend('Lasso');
saveas(h_fig, [output_file 'Accuracy.png']);
close(h_fig)