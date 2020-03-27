clc;clear;
addpath(genpath('.'));
output_file = './data/result/';
if exist(output_file,'dir') == 0
    mkdir(output_file);
end
%% Load dataset
train_image = loadMNISTImages('./data/mnist/train-images-idx3-ubyte');
train_label = loadMNISTLabels('./data/mnist/train-labels-idx1-ubyte');
test_image = loadMNISTImages('./data/mnist/t10k-images-idx3-ubyte');
test_label = loadMNISTLabels('./data/mnist/t10k-labels-idx1-ubyte');
%------------------------
train_image = train_image(:, 1:1000);
train_label = train_label(1:1000);
test_image = test_image(:, 1:3);
test_label = test_label(1:3);
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
if exist([output_file 'lasso/'],'dir') == 0
    mkdir([output_file 'lasso/']);
end
for i=1:size(test_image, 2)
    t1=clock;
    [w_lasso, lasso_res] = lasso(A, B(:, i));
    %-----------------------------------
    h_fig = figure('Name','Lasso', 'Visible', 'off');
    subplot(3,1,1);
    plot(lasso_res.DF, 'LineWidth',2);
    title('non-zero parameter number')
    xlabel('\lambda')
    ylabel('Number')
    subplot(3,1,2);
    plot(lasso_res.Lambda, 'LineWidth',2);
    title('Lambda')
    subplot(3,1,3);
    plot(lasso_res.MSE, 'LineWidth',2);
    title('MSE')
    saveas(h_fig, [output_file 'lasso/Lasso_' num2str(i) '.png']);
    close(h_fig)
    %% Correlation classification
    score = Correlation(w_lasso(:,1:end-1), train_label);
    score_norm = normalize(score,1,'norm');
    h_fig = figure('Name','Lasso', 'Visible', 'off');
    image('XData',lasso_res.Lambda(1:end-1),...
          'YData',0:(size(score_norm, 1) - 1),...
          'CData',score_norm,'CDataMapping','scaled');colorbar;
    hold on
    plot(lasso_res.Lambda(1:end-1), ones(size(score_norm, 2), 1) * test_label(i), 'LineWidth', 5, 'Color', 'r')
    legend('Truth')
    xlabel('\lambda')
    ylabel('classification')
    saveas(h_fig, [output_file 'lasso/score_norm_lasso_' num2str(i) '.png']);
    close(h_fig)
    t2=clock;
    fprintf(['Lasso Test Image ' num2str(i) ', time=%.2fs' '\n'], etime(t2,t1));
end
%-----------------------------------
% Pan Wei
%-----------------------------------
if exist([output_file 'PanWei/'],'dir') == 0
    mkdir([output_file 'PanWei/']);
end
for i=1:size(test_image, 2)
    t1=clock;
    lambda_max = max(B(:, i)' * A);
    w_pan = zeros(size(A, 2), 9);
    for ratio = 1:9
        lambda = lambda_max * ratio / 10;
        MAXITER = 100;
        [end_iter_pan, w_pan_iter] =  pan(B(:, i), A, lambda, MAXITER);
        w_pan(:, ratio) = w_pan_iter(:, end);
    end
    %-----------------------------------
    lambdas = lambda_max * (1:9) ./ 10;
    h_fig = figure('Name', 'PanWei', 'Visible', 'off');
    plot(lambdas, sum(w_pan > 0, 1), 'LineWidth',2);
    title('non-zero parameter number')
    xlabel('\lambda')
    ylabel('Number')
    saveas(h_fig, [output_file 'PanWei/PanWei_' num2str(i) '.png']);
    close(h_fig)
    %% Correlation classification
    score = Correlation(w_pan, train_label);
    score_norm = normalize(score,1,'norm');
    h_fig = figure('Visible', 'off');
    image('XData', lambdas,...
          'YData',0:(size(score_norm, 1) - 1),...
          'CData',score_norm,'CDataMapping','scaled');colorbar;
    hold on
    plot(lambdas, ones(size(score_norm, 2), 1) * test_label(i), 'LineWidth',5, 'Color', 'r')
    legend('Truth')
    xlabel('\lambda')
    ylabel('classification')
    saveas(h_fig, [output_file 'PanWei/score_norm_PanWei_' num2str(i) '.png']);
    close(h_fig)
    t2=clock;
    fprintf(['PanWei Test Image ' num2str(i) ', time=%.2fs' '\n'], etime(t2,t1));
end

%-----------------------------------
% Pan Wei revised
%-----------------------------------
if exist([output_file 'PanWeiRevised/'],'dir') == 0
    mkdir([output_file 'PanWeiRevised/']);
end
for i=1:size(test_image, 2)
    t1=clock;
    lambda_max = max(B(:, i)' * A);
    w_pan_revised = zeros(size(A, 2), 9);
    for ratio = 1:9
        lambda = lambda_max * ratio / 10;
        MAXITER = 100;
        [~, end_iter_pan_re, w_screen] = pan_revised(B(:, i), A, lambda,MAXITER);
        w_pan_revised(:, ratio) = w_screen(:, end);
    end
    %-----------------------------------
    lambdas = lambda_max * (1:9) ./ 10;
    h_fig = figure('Name', 'PanWeiRevised', 'Visible', 'off');
    plot(lambdas, sum(w_pan_revised > 0, 1), 'LineWidth',2);
    title('non-zero parameter number')
    xlabel('\lambda')
    ylabel('Number')
    saveas(h_fig, [output_file 'PanWeiRevised/PanWeiRevised' num2str(i) '.png']);
    close(h_fig)
    %% Correlation classification
    score = Correlation(w_pan_revised, train_label);
    score_norm = normalize(score,1,'norm');
    h_fig = figure('Visible', 'off');
    image('XData', lambdas,...
          'YData',0:(size(score_norm, 1) - 1),...
          'CData',score_norm,'CDataMapping','scaled');colorbar;
    hold on
    plot(lambdas, ones(size(score_norm, 2), 1) * test_label(i), 'LineWidth',5, 'Color', 'r')
    legend('Truth')
    xlabel('\lambda')
    ylabel('classification')
    saveas(h_fig, [output_file 'PanWeiRevised/score_norm_PanWeiRevised_' num2str(i) '.png']);
    close(h_fig)
    t2=clock;
    fprintf(['PanWeiRevised Test Image ' num2str(i) ', time=%.2fs' '\n'], etime(t2,t1));
end