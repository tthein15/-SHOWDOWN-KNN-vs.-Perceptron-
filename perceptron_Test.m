function i_sig = perceptron_Test(weight,bias,trial, PS, stdev)
% transpose test data
trial = trial.'; 

if stdev == 1
     % applying standardization mapping(training data)
    New_test = mapstd('apply',trial,PS);
else
    New_test = trial;
end

% predict outcome through the activation function
prediction = bias + dot(weight, New_test); 
% return sign
i_sig = sign(prediction);
% return sign
end