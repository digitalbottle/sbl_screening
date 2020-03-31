clc;clear;
addpath('dict_learning')
denoise_file = './data/result/reconstruction/denoise/';
para_es_file = './data/result/reconstruction/para_es/';
%% prepare dataset
[dict_img, dict_pt, target_imgs, clean_imgs, target_pts] = load_data('./data');
dict_set_num = 100;
test_set_num = 2;
lambda_num = 3;
lambda_ratios = linspace(0.1, 0.9, lambda_num);

dict_img_flatten = dict_img(:, :)';
train_num_list = randperm(size(dict_img, 1), dict_set_num);
test_num_list = randperm(size(target_imgs, 1), test_set_num);

dict_set = dict_img_flatten(:, train_num_list);
clean_set = clean_imgs(test_num_list);
target_set = target_imgs(test_num_list);
%% Record
%% main loop
for i=1:test_set_num
    clean_img = clean_set{i};
    clean_img_flatten = clean_img(:);
    target_img = target_set{i};
    target_img_flatten = target_img(:);
    % save target and clean image    
    target_output_file = [denoise_file num2str(i) '.png'];
    clean_output_file = [denoise_file num2str(i) '_clean.png'];
    if exist(denoise_file,'dir') == 0
        mkdir(denoise_file);
    end
    target_img_norm = (target_img - min(target_img, [], 'all')) ./ (max(target_img, [], 'all')- min(target_img, [], 'all'));
    clean_img_norm = (clean_img - min(clean_img, [], 'all')) ./ (max(clean_img, [], 'all')- min(clean_img, [], 'all'));
    target_img_norm = imresize(target_img_norm, 10);
    clean_img_norm = imresize(clean_img_norm, 10);
    imwrite(target_img_norm, target_output_file);
    imwrite(clean_img_norm, clean_output_file);
    %% Sparse representation: test data [i] = Sparse_representation * train data
    % dict
    A = dict_set;
    A = A .* (1 ./ sqrt(sum(A .* A, 1)));
    % target
    B = target_img_flatten;
    B = B .* (1 ./ sqrt(sum(B .* B, 1)));
    %-----------------------------------
    % Pan Wei Revised
    %-----------------------------------
    lambda_max = max(B' * A);
    w_pan = zeros(size(A, 2), lambda_num);
    lambda = lambda_max * lambda_ratios;
    for ratio = 1:lambda_num
        fprintf('Pan Test ratio %d/%d, img %d/%d\n', ...
                       ratio, lambda_num, i, test_set_num);
        MAXITER = 100;
        [~, end_iter_pan_re, w_screen] = pan_revised(B, A, lambda(ratio), MAXITER);
        w_pan(:, ratio) = w_screen(:, end);
        %-----------------------------------
        % Denoise
        %-----------------------------------
        output_file = [denoise_file num2str(i)];
        if exist(output_file,'dir') == 0
            mkdir(output_file);
        end
        output = [output_file '/' num2str(ratio) '.png'];
        Denoise_img_flatten = A * w_screen(:, end);
        Denoise_img = reshape(Denoise_img_flatten, size(target_img));
        Denoise_img_norm = (Denoise_img - min(Denoise_img, [], 'all')) ./ (max(Denoise_img, [], 'all')- min(Denoise_img, [], 'all'));
        Denoise_img_norm = imresize(Denoise_img_norm, 10);
        imwrite(Denoise_img_norm, output);
        %-----------------------------------
        % Parameter Estimation
        %-----------------------------------
    end
    
end