clc;clear;
%% prepare dataset
[dict_img, dict_pt, target_img, target_pts] = load_data('./data');
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
%% pan revised method
lambda_max = max(y' * A);
screen_ratio = zeros(11,1);
pan_revised_times = ones(11,1);
parfor ratio = 0:10
    lambda = lambda_max * ratio / 10;
    MAXITER = 100;
    t1=clock;
    [screen_ratio(ratio+1), end_iter_pan_re, w_screen] = pan_revised(y, A, lambda,MAXITER);
    t2=clock;
    fprintf('pan revised time = %.2fs\n', etime(t2,t1));
    pan_revised_times(ratio+1) = etime(t2,t1);
end
%% Plot
figure('Name','Screen Ratio');
plot(0:0.1:1, screen_ratio(:), '-o', 'MarkerFaceColor','r', 'LineWidth',2, 'color', [0.6350 0.0780 0.1840]);
xlim([0 1])
ylim([0 1])
xlabel('\lambda/\lambda_{max}')
ylabel('Rejection Percentage')
%% time cost
pan_times = ones(11,1);
parfor ratio = 0:10
    lambda = lambda_max * ratio / 10;
    MAXITER = 100;
    t1=clock;
    [end_iter_pan, w_pan] =  pan(y, A, lambda,MAXITER);
    t2=clock;
    fprintf('pan revised time = %.2fs\n', etime(t2,t1));
    pan_times(ratio+1) = etime(t2,t1);
end

figure('Name','Speed up');
plot(0:0.1:1, log10(pan_times(:)./pan_revised_times(:)), '-o', 'MarkerFaceColor','r', 'LineWidth',2, 'color', [0.6350 0.0780 0.1840]);
xlim([0 1])
xlabel('\lambda/\lambda_{max}')
ylabel('Speed up 10\^ times')