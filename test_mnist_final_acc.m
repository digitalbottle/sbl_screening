clc;clear;
addpath('dict_learning', 'utils')
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
exp_time = 10;
lambda_num = 10;
lambda_ratios = linspace(0.1, 0.9, lambda_num);
dict_set_num = 1000;
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
%     %-----------------------------------
%     % Lasso
%     %-----------------------------------
%     lasso_result = zeros(size(test_image, 2), lambda_num);
%     t1=clock;
%     parfor i=1:size(test_image, 2)
%         [~, lasso_res] = lasso(A, B(:, i), 'NumLambda', lambda_num+1);
%         lambda_max = lasso_res.Lambda(end-1);
%         [w_lasso, lasso_res] = lasso(A, B(:, i), 'Lambda', lambda_ratios * lambda_max);
%         %% Correlation classification
%         score = Correlation(w_lasso(:,1:end), train_label);
%         [~, argmax] = max(score, [], 1);
%         lasso_result(i, :) = argmax - 1;
%     end
%     t2=clock;
%     lasso_acc(exp_t, :) = sum(lasso_result == test_label, 1) ./ size(test_image, 2);
%     fprintf('Lasso Test Exp %d/%d, time=%.2fs, acc=%.2f\n', ...
%                  exp_t, exp_time, etime(t2,t1), mean(lasso_acc(exp_t, :)));
    %-----------------------------------
    % Pan Wei
    %-----------------------------------
    Pan_result = zeros(size(test_image, 2), lambda_num);
    t1=clock;
    parfor i=1:size(test_image, 2)
        lambda_max = max(B(:, i)' * A);
        w_pan = zeros(size(A, 2), lambda_num);
        lambda = lambda_max * lambda_ratios;
        for ratio = 1:lambda_num
            fprintf('Pan Test Exp ratio %d/%d, img %d/%d exp %d/%d\n', ...
                           ratio, lambda_num, i, size(test_image, 2), exp_t, exp_time);
            MAXITER = 100;
            % [end_iter_pan, w_pan_iter] =  pan(B(:, i), A, lambda(ratio), MAXITER);
            [~, end_iter_pan_re, w_screen] = pan_revised(B(:, i), A, lambda(ratio), MAXITER);
            w_pan(:, ratio) = w_screen(:, end);
        end
        %% Correlation classification
        score = Correlation(w_pan(:,1:end), train_label);
        [~, argmax] = max(score, [], 1);
        Pan_result(i, :) = argmax - 1;
    end
    t2=clock;
    Pan_acc(exp_t, :) = sum(Pan_result == test_label, 1) ./ size(test_image, 2);
    fprintf('Pan Test Exp %d/%d, time=%.2fs, acc=%.2f\n', ...
                 exp_t, exp_time, etime(t2,t1), mean(Pan_acc(exp_t, :)));
end

h_fig = figure('Name', 'Accuracy', 'Visible', 'off');
% hold on
% lasso_acc_mean = mean(lasso_acc, 1);
% lasso_acc_std = std(lasso_acc, 1, 1);
% errorbar(lambda_ratios, lasso_acc_mean, lasso_acc_std, '-s', 'LineWidth', 2, 'Color', 'r', ...
%                   'MarkerSize',10, 'MarkerEdgeColor','r','MarkerFaceColor','w')
Pan_acc_mean = mean(Pan_acc, 1);
Pan_acc_std = std(Pan_acc, 1, 1);
errorbar(lambda_ratios, Pan_acc_mean, Pan_acc_std, '-s', 'LineWidth', 2, 'Color', 'b', ...
                  'MarkerSize',10, 'MarkerEdgeColor','g','MarkerFaceColor','w')

title('Accuracy -- \lambda / \lambda_{max}')
xlabel('\lambda / \lambda_{max}')
ylabel('Accuracy')
% legend('Lasso', 'Pan Wei Screen Test');
saveas(h_fig, [output_file 'Accuracy.png']);
close(h_fig)