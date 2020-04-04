function draw_bbox(output_file, image, gt, bbox, enlarge)
    h_fig = figure('Name', 'Bounding Box', 'Visible', 'off');
    img = imresize(image, enlarge);
    image_new = cat(3, img, img, img);
    imshow(image_new, 'border', 'tight');
    hold on;
    for i=1:size(gt, 1)
        x = gt(i, 1);
        y = gt(i, 2);
        w = gt(i, 3);
        h = gt(i, 3);
        rectangle('Position', enlarge * [x-w/2+0.5 y-h/2+0.5 w h], ...
                         'EdgeColor','r', 'Curvature',[1 1], 'LineWidth',2);
    end
    for i=1:size(bbox, 1)
        x = bbox(i, 1);
        y = bbox(i, 2);
        w = bbox(i, 3);
        h = bbox(i, 3);
        rectangle('Position', enlarge * [x-w/2+0.5 y-h/2+0.5 w h], ...
                         'EdgeColor','b', 'Curvature',[1 1], 'LineWidth',2);
    end
    saveas(h_fig, output_file);
end

