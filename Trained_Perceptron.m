classdef Trained_Perceptron
    % Class Object:Holds data for this class and predictor function   
    
    properties
        weight
        bias
        std_MapCode
        class_error
        kfolds
        performance_Eval
        stdev
    end
    
    methods
        function obj = Trained_Perceptron(weight,bias,std_MapCode, class_error, kfolds, performance_Eval, stdev)
            % class constructor method
            obj.weight = weight;
            obj.bias = bias;
            obj.std_MapCode = std_MapCode;
            obj.class_error = class_error;
            obj.kfolds = kfolds;
            obj.performance_Eval = performance_Eval;
            obj.stdev= stdev;
        end
        
        function predictions = Perceptron_prediction(obj, test)
            % predictor method
            predictions = test(:,end);
            for i_row = 1:height(test)
                 % run test data through perceptron
                i_sign = perceptronTest(obj.weight,obj.bias,test{i_row,1:end-1},obj.std_MapCode, obj.stdev);
                
                % result interpretator
                if i_sign == 1 
                    predictions{i_row,1} = "real";
                else
                    predictions{i_row,1} = "false";
                end
            end
        end
    end
end

