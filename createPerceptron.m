function ptron = createPerceptron(trainDev, maxIter, kfolds, stdev)
ptron(1:kfolds) ={'temp'};
% Importing and Cleaning 
clc 
import currencytrain.csv
cleaned = currencytrain(:,[2:end 1]); % moving real/fake label to last column
[trainDev, testdata] = splitTrainTest(cleaned , 0.8); % splits Data into Testing and Training/Development

% Splitting and Training
[training, dev] = kfoldDataSplit(trainDev, kfolds);
for kfoldCounter = 1 : kfolds
    % exchange train/dev if kfolds = 1
    if kfolds == 1 
         % trains perceptron(gives weight, bias, test/dev data normalization)
        [weight, bias, std_MapCode] = train_Perceptron(dev{kfoldCounter}, maxIter, stdev);
    else
         % trains perceptron(gives weight, bias, test/dev data normalization)
        [weight, bias, std_MapCode] = train_Perceptron(training{kfoldCounter}, maxIter, stdev); 
    end


    % Testing Developmental Data 
    % initialize
    num_Correct = 0;  
    for i_row = 1:height(dev{kfoldCounter})
        % run test data through perceptron
        sign = perceptron_Test(weight,bias,dev{kfoldCounter}{i_row,1:end-1}, std_MapCode, stdev); 
        % result interpretator
        if dev{kfoldCounter}{i_row,end} == 'real' 
            result = 1;
        else 
            result = -1;
        end
        
        if sign*result > 0
            % count of correct classification
            num_Correct = num_Correct + 1; 
        end
    end
    
     % mean absolute error calculation
    class_error = 1 - num_Correct/height(dev{kfoldCounter});
    if kfolds == 1
        performance_Eval = NaN;
    else
        performance_Eval = sprintf('Accuracy: %d out of %d (%.1f%%), or %.4f classification error (Absolute Mean Error/L1).', num_Correct, height(dev{kfoldCounter}),round(100*(num_Correct/height(dev{kfoldCounter})),0), class_error); % summary of performance in words
    end
    
    % adding data to perceptron object
    obj = Trained_Perceptron(weight, bias, std_MapCode, class_error, kfolds, performance_Eval,stdev);
    % adding perceptron object into a cell array of other perceptrons (for kfolds)
    ptron{kfoldCounter} = obj; 
    %fprintf('Accuracy: %d out of %d, or %d %% \n', num_Correct, height(dev{kfolds}),round(100*(num_Correct/height(dev{kfolds})),0));
end