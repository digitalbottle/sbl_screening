function [dict_img, dict_pt, target_img, target_pts] = load_data(path)
    dict_img = permute(h5read(fullfile(path, 'dictionary.h5'), '/dictionary'), [3, 2, 1]);
    dict_pt = permute(h5read(fullfile(path, 'dictionary.h5'), '/point'), [2, 1]);
    target_num = h5read(fullfile(path, 'target.h5'), '/target_num');
    target_img = {};
    target_pts = {};
    for i = 1:target_num
        target_img{i} = permute(h5read(fullfile(path, 'target.h5'), ['/targets_' num2str(i-1, '%02d')]), [2, 1]);
        target_pts{i} = permute(h5read(fullfile(path, 'target.h5'), ['/point_' num2str(i-1, '%02d')]), [2, 1]);
    end
end