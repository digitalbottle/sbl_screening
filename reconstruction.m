clc;clear;
addpath('dict_learning', 'MSSIM_PSNR')
denoise_file = './data/result/reconstruction/denoise/';
Wavelet_file = './data/result/reconstruction/Wavelet/';
if exist(denoise_file, 'dir') ~= 0
    rmdir(denoise_file, 's');
end
if exist(Wavelet_file, 'dir') ~= 1
    rmdir(Wavelet_file, 's');
end
%% prepare dataset
[dict_img, dict_pt, target_imgs, clean_imgs, target_pts] = load_data('./data');
dict_set_num = 1000;
test_set_num = 50;
lambda_num = 10;
lambda_ratios = linspace(0.01, 0.30, lambda_num);
Wavelet_ratios = linspace(1, 4, 4);
%-----------------------------------
dict_img_flatten = dict_img(:, :)';
train_num_list = randperm(size(dict_img, 1), dict_set_num);
test_num_list = randperm(size(target_imgs, 1), test_set_num);
%-----------------------------------
dict_set = dict_img_flatten(:, train_num_list);
clean_set = clean_imgs(test_num_list);
target_set = target_imgs(test_num_list);
%% Record
denoise_acc = zeros(test_set_num, lambda_num);
psnr_acc = zeros(test_set_num, lambda_num);
ssim_acc = zeros(test_set_num, lambda_num);
% Wavelet
Wavelet_denoise_acc = zeros(test_set_num, 4);
Wavelet_psnr_acc = zeros(test_set_num, 4);
Wavelet_ssim_acc = zeros(test_set_num, 4);
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
    target_img_norm = (target_img - min(min(target_img))) ./ (max(max(target_img)) - min(min(target_img)));
    clean_img_norm = (clean_img - min(min(clean_img))) ./ (max(max(clean_img)) - min(min(clean_img)));
    target_img_norm = imresize(target_img_norm, 10);
    clean_img_norm = imresize(clean_img_norm, 10);
    imwrite(target_img_norm, target_output_file);
    imwrite(clean_img_norm, clean_output_file);
    
    target_output_file = [Wavelet_file num2str(i) '.png'];
    clean_output_file = [Wavelet_file num2str(i) '_clean.png'];
    if exist(Wavelet_file,'dir') == 0
        mkdir(Wavelet_file);
    end
    imwrite(target_img_norm, target_output_file);
    imwrite(clean_img_norm, clean_output_file);
    %% Sparse representation: test data [i] = Sparse_representation * train data
    % dict
    A = dict_set;
    A = A .* (1 ./ sqrt(sum(A .* A, 1)));
    % target
    B = target_img_flatten;
    B = B .* (1 ./ sqrt(sum(B .* B, 1)));
    % clean
    C = clean_img_flatten;
    C = C .* (1 ./ sqrt(sum(C .* C, 1)));
    %-----------------------------------
    % Pan Wei Revised
    %-----------------------------------
    lambda_max = max(B' * A);
    w_pan = zeros(size(A, 2), lambda_num);
    lambda = lambda_max * lambda_ratios;
    parfor ratio = 1:lambda_num
        fprintf('Pan Test ratio %d/%d, img %d/%d\n', ...
                       ratio, lambda_num, i, test_set_num);
        MAXITER = 100;
        [~, end_iter_pan_re, w_screen] = pan_revised(B, A, lambda(ratio), MAXITER);
        w_pan(:, ratio) = w_screen(:, end);
        % [end_iter_pan, w_pan_iter] =  pan(B(:, i), A, lambda(ratio), MAXITER);
        % w_pan(:, ratio) = w_pan_iter(:, end);
        %-----------------------------------
        % Denoise
        %-----------------------------------
        output_file = [denoise_file num2str(i)];
        if exist(output_file,'dir') == 0
            mkdir(output_file);
        end
        output = [output_file '/' num2str(ratio) '.png'];
        Denoise_img_flatten = A * w_screen(:, end);
        % save denoise result
        Denoise_img = reshape(Denoise_img_flatten, size(target_img));
        Denoise_img_norm = (Denoise_img - min(min(Denoise_img))) ./ (max(max(Denoise_img)) - min(min(Denoise_img)));
        Denoise_img_norm = imresize(Denoise_img_norm, 10);
        imwrite(Denoise_img_norm, output);
        % denoise acc
        denoise_acc(i, ratio) = 1 - sum((Denoise_img_flatten - C) .^ 2) / sum(C .^ 2);
        psnr_acc(i, ratio) = getPSNR(Denoise_img, clean_img);
        ssim_acc(i, ratio) = getMSSIM(Denoise_img, clean_img);
    end
    for ratio = 1:4
        %-----------------------------------
        % Wavelet signal denoising result
        %-----------------------------------
        Denoise_img = wdenoise(target_img, Wavelet_ratios(ratio),'Wavelet','sym4');
        Denoise_img_norm = (Denoise_img - min(min(Denoise_img))) ./ (max(max(Denoise_img)) - min(min(Denoise_img)));
        Denoise_img_norm = imresize(Denoise_img_norm, 10);
        output_file = [Wavelet_file num2str(i)];
        if exist(output_file,'dir') == 0
            mkdir(output_file);
        end
        output = [output_file '/' num2str(ratio) '.png'];
        imwrite(Denoise_img_norm, output);
        Wavelet_denoise_acc(i, ratio) = 1 - sum(sum((Denoise_img - clean_img) .^ 2)) / sum(sum(clean_img .^ 2));
        Wavelet_psnr_acc(i, ratio) = getPSNR(Denoise_img, clean_img);
        Wavelet_ssim_acc(i, ratio) = getMSSIM(Denoise_img, clean_img);
    end   
end
% Draw result
%-----------------------------------
% Show acc
%-----------------------------------
draw_denoise([denoise_file 'Accuracy.png'], ...
   lambda_ratios, denoise_acc, '\lambda / \lambda_{max}', 'Accuracy');
%-----------------------------------
% Show PSNR
%-----------------------------------
draw_denoise([denoise_file 'PSNR.png'], ...
   lambda_ratios, psnr_acc, '\lambda / \lambda_{max}', 'PSNR');
%-----------------------------------
% Show SSIM
%-----------------------------------
draw_denoise([denoise_file 'SSIM.png'], ...
   lambda_ratios, ssim_acc, '\lambda / \lambda_{max}', 'SSIM');
%% Draw Wavelet result
%-----------------------------------
% Show acc
%-----------------------------------
draw_denoise([Wavelet_file 'Accuracy.png'], ...
   Wavelet_ratios, Wavelet_denoise_acc, 'Level', 'Accuracy');
%-----------------------------------
% Show PSNR
%-----------------------------------
draw_denoise([Wavelet_file 'PSNR.png'], ...
   Wavelet_ratios, Wavelet_psnr_acc, 'Level', 'PSNR');
%-----------------------------------
% Show SSIM
%-----------------------------------
draw_denoise([Wavelet_file 'SSIM.png'], ...
   Wavelet_ratios, Wavelet_ssim_acc, 'Level', 'SSIM');