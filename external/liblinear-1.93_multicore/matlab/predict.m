function [predicted_label, accuracy, decision_values] = predict(label, data, model)
    if model.bias > 0
        data = [data, ones(size(data, 1), 1) * model.bias];
    end
    predicted_label = data * model.w';
    
    if model.Parameters < 10                   % classification
        assert(size(label, 2) == 1);
        [decision_values, idx] = max(predicted_label, [], 2);
        predicted_label = model.Label(idx);
        accuracy = sum(predicted_label == label) / length(label);
        fprintf('Accuracy = %f\n', accuracy);
    else                                       % regression  
        error = sum(sum((predicted_label - label).^2)) / size(label, 1);
        fprintf('Mean squared error = %f\n', error);
    end
end