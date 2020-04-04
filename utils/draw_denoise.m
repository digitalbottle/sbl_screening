function draw_denoise(output_file, lambda_ratios, denoise_acc, x_label_name, y_label_name)
    h_fig = figure('Name', y_label_name, 'Visible', 'off');
    Pan_acc_mean = mean(denoise_acc, 1);
    Pan_acc_std = std(denoise_acc, 1, 1);
    errorbar(lambda_ratios, Pan_acc_mean, Pan_acc_std, '-s', 'LineWidth', 2, 'Color', 'b', ...
                      'MarkerSize',10, 'MarkerEdgeColor','b','MarkerFaceColor','w')
    title([y_label_name ' -- ' x_label_name])
    xlabel(x_label_name)
    ylabel(y_label_name)
    % legend('Lasso', 'Pan Wei Screen Test');
    saveas(h_fig, output_file);
    close(h_fig)
end

