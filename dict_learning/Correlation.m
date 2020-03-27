function score = Correlation(w_lasso, train_label)
    one_hot = full(sparse(1:numel(train_label + 1), train_label + 1,1));
    score = one_hot' * abs(w_lasso);
end

