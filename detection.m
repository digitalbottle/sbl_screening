clc;clear;
addpath('dict_learning', 'utils')
detection_file = './data/result/reconstruction/detection/';
% Clean old results
if exist(detection_file, 'dir') ~= 0
    rmdir(detection_file, 's');
end
%% prepare dataset
[dict_img, dict_pt, target_imgs, clean_imgs, target_pts] = load_data('./data');
dict_set_num = 1000;
test_set_num = 10;
lambda_num = 10;
lambda_ratios = linspace(0.01, 0.09, lambda_num);
nms_overlap = 0.5;
%-----------------------------------
dict_img_flatten = dict_img(:, :)';
train_num_list = randperm(size(dict_img, 1), dict_set_num);
test_num_list = randperm(size(target_imgs, 1), test_set_num);
%-----------------------------------
dict_set = dict_img_flatten(:, train_num_list);
dict_pt_set = dict_pt(train_num_list, :);

clean_set = clean_imgs(test_num_list);
target_set = target_imgs(test_num_list);
target_pt_set = target_pts(test_num_list);
%% Record
IOU_res = zeros(test_set_num, lambda_num);
%% main loop
parfor i=1:test_set_num
    clean_img = clean_set{i};
    clean_img_flatten = clean_img(:);
    target_img = target_set{i};
    target_img_flatten = target_img(:);
    target_pt = target_pt_set{i};
    % save target and clean image    
    target_output_file = [detection_file num2str(i) '.png'];
    clean_output_file = [detection_file num2str(i) '_clean.png'];
    if exist(detection_file,'dir') == 0
        mkdir(detection_file);
    end
    target_img_norm = (target_img - min(min(target_img))) ./ (max(max(target_img)) - min(min(target_img)));
    clean_img_norm = (clean_img - min(min(clean_img))) ./ (max(max(clean_img)) - min(min(clean_img)));
    draw_bbox(target_output_file, target_img_norm, target_pt(:, 2:end), [], 10);
    draw_bbox(clean_output_file, clean_img_norm, target_pt(:, 2:end), [], 10);
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
    %-------------lasso-----------------
    [w_lasso, lasso_res] = lasso(A, B, 'NumLambda', lambda_num+1);
    %-------------lasso-----------------
    lambda = lambda_max * lambda_ratios;
    for ratio = 1:lambda_num
        fprintf('Pan Test ratio %d/%d, img %d/%d\n', ...
                       ratio, lambda_num, i, test_set_num);
        %------------PanWei-----------------
        MAXITER = 100;
        [~, end_iter_pan_re, w_screen] = pan_revised(B, A, lambda(ratio), MAXITER);
        w_pan(:, ratio) = w_screen(:, end);
        %------------lasso------------------
%         w_pan(:, ratio) = w_lasso(:, ratio);
        %-----------------------------------
        % Detection
        %-----------------------------------
        output_file = [detection_file num2str(i)];
        if exist(output_file,'dir') == 0
            mkdir(output_file);
        end
        % detection result
        result_pt = dict_pt_set(abs(w_pan(:, ratio)) > (0.1*max(w_pan(:, ratio))), :);
        %% NMS
        x = result_pt(:, 1);
        y = result_pt(:, 2);
        w = result_pt(:, 3);
        h = result_pt(:, 3);
        nms_bbox = zeros(size(result_pt, 1), 5);
        nms_bbox(:, 1) = x-w/2+0.5;
        nms_bbox(:, 2) = y-h/2+0.5;
        nms_bbox(:, 3) = x+w/2+0.5;
        nms_bbox(:, 4) = y+h/2+0.5;
        nms_bbox(:, 5) = w_pan(abs(w_pan(:, ratio)) > (0.1*max(w_pan(:, ratio))), ratio);
        top_bbox = nms(nms_bbox, nms_overlap);
        result_pt_nms = zeros(size(top_bbox, 1), 3);
        result_pt_nms(:, 1) = (top_bbox(:, 1) + top_bbox(:, 3) - 1) ./ 2;
        result_pt_nms(:, 2) = (top_bbox(:, 2) + top_bbox(:, 4) - 1) ./ 2;
        result_pt_nms(:, 3) = top_bbox(:, 3) - top_bbox(:, 1);
        %% draw
        draw_bbox([output_file '/' num2str(ratio) '.png'], ...
                       target_img_norm, target_pt(:, 2:end), result_pt_nms, 10);
        draw_bbox([output_file '/' num2str(ratio) '_c.png'], ...
                       clean_img_norm, target_pt(:, 2:end), result_pt_nms, 10);
        % detection iou
        bboxA = zeros(size(result_pt, 1), 4);
        bboxA(:, 1) = result_pt(:, 1) - result_pt(:, 3) ./ 2 + 0.5;
        bboxA(:, 2) = result_pt(:, 2) - result_pt(:, 3) ./ 2 + 0.5;
        bboxA(:, 3) = result_pt(:, 3);
        bboxA(:, 4) = result_pt(:, 3);
        %-----------------------------------
        bboxB = zeros(size(target_pt, 1), 4);
        bboxB(:, 1) = target_pt(:, 2) - target_pt(:, 4) ./ 2 + 0.5;
        bboxB(:, 2) = target_pt(:, 3) - target_pt(:, 4) ./ 2 + 0.5;
        bboxB(:, 3) = target_pt(:, 4);
        bboxB(:, 4) = target_pt(:, 4);
        if size(bboxA, 1) > size(bboxB, 1)
            IOU_res(i, ratio) = mean(max(bboxOverlapRatio(bboxA,bboxB), [], 2));
        else
            IOU_res(i, ratio) = mean(max(bboxOverlapRatio(bboxA,bboxB), [], 1));
        end  
    end
end
h_fig = figure('Name', 'IOU', 'Visible', 'off');
IOU_acc_mean = mean(IOU_res, 1);
IOU_acc_std = std(IOU_res, 1, 1);
errorbar(lambda_ratios, IOU_acc_mean, IOU_acc_std, '-s', 'LineWidth', 2, 'Color', 'black', ...
                  'MarkerSize',10, 'MarkerEdgeColor','r','MarkerFaceColor','w')

title('IOU -- \lambda / \lambda_{max}')
xlabel('\lambda / \lambda_{max}')
ylabel('mean-IOU')
% legend('Lasso', 'Pan Wei Screen Test');
saveas(h_fig, [detection_file 'IOU.png']);
close(h_fig)