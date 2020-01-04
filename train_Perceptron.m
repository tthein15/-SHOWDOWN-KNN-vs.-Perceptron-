function [weight, bias, PS] = train_Perceptron(Data,maxCouter,stdev)

% initialize weight for independent variables
weight = zeros(min(size(Data{:,1:(end-1)})),1); 
% initialize bias 
bias = 0;

% hyperparameter loop iteration constraint
for iter = 1:maxCouter 
    % create data matrix(no height label)
    Data_Matrix = Data{:,1:(end-1)}; 
    % transpose Data Matrix (switch row/columns)
    Data_Matrix = Data_Matrix.'; 
    if stdev ==1
        % create standardized data matrix and standardization mapping structure
        [StandardizedDataMatrix,PS] = mapstd(Data_Matrix, 0, 1); 
    else
        StandardizedDataMatrix = Data_Matrix;
    end
    
    % loop through all rows of matrix
    for i_row = 1:max(size((StandardizedDataMatrix)))
        % activation function
        sum = dot(weight,StandardizedDataMatrix(:,i_row)) + bias;
        
        if sum > 0
            active = 1;
        else
            active = -1;
        end
        
        % read height label
        if Data{i_row,end} == 'real' 
            label = 1;
        else
            label = -1;
        end
        
        % height label/activation agreement
        if active*label <= 0 
           % weight correction
           weight = weight + label*StandardizedDataMatrix(:,i_row); 
           % bias correction
           bias = bias + label; 
        end
    end
end
end